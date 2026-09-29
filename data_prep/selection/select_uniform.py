#!/usr/bin/env python3
"""Freeze nested uniform-random OPD supports and balanced training schedules."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[2] / "outputs" / "opd_project_root"))
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCHEMA = "opd_uniform_selection_schedules_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def prompt_id(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping) or "index" not in extra:
        raise ValueError("every candidate must have extra_info.index")
    return str(extra["index"])


def prompt_characters(row: Mapping[str, Any]) -> int:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"candidate {prompt_id(row)} has no prompt")
    return sum(
        len(str(message.get("content", "")))
        for message in prompt
        if isinstance(message, Mapping)
    )


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def annotate(
    row: Mapping[str, Any],
    *,
    selection_seed: int,
    training_seed: int,
    support_size: int,
    schedule_index: int,
    presentation_index: int,
    batch_size: int,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(row))
    extra = dict(result["extra_info"])
    extra["selection_benchmark"] = {
        "schema_version": SCHEMA,
        "selection_method": "uniform_random_without_replacement",
        "selection_rng": "numpy.random.Generator(PCG64)",
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "support_size": support_size,
        "schedule_index": schedule_index,
        "presentation_index": presentation_index,
        "optimizer_step": schedule_index // batch_size,
        "position_in_step": schedule_index % batch_size,
    }
    result["extra_info"] = extra
    return result


def materialize_schedule(
    support: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    total = steps * batch_size
    base, remainder = divmod(total, len(support))
    extras = {
        prompt_id(row)
        for row in sorted(
            support,
            key=lambda row: stable_int(
                training_seed, "presentation-remainder", prompt_id(row)
            ),
        )[:remainder]
    }
    targets = {
        prompt_id(row): base + int(prompt_id(row) in extras)
        for row in support
    }
    counts: Counter[str] = Counter()
    schedule = []
    for presentation in range(max(targets.values())):
        eligible = [
            row
            for row in support
            if counts[prompt_id(row)] < targets[prompt_id(row)]
        ]
        eligible.sort(
            key=lambda row: stable_int(
                training_seed,
                "schedule-round",
                presentation,
                prompt_id(row),
            )
        )
        for row in eligible:
            identifier = prompt_id(row)
            schedule.append(
                annotate(
                    row,
                    selection_seed=selection_seed,
                    training_seed=training_seed,
                    support_size=len(support),
                    schedule_index=len(schedule),
                    presentation_index=counts[identifier],
                    batch_size=batch_size,
                )
            )
            counts[identifier] += 1
    if len(schedule) != total:
        raise AssertionError(f"schedule has {len(schedule)} rows, expected {total}")
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("support presentations differ by more than one")
    return schedule, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--candidate-ids-json",
        type=Path,
        default=None,
        help="optional frozen JSON containing a problem_ids list",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-sizes", type=int, nargs="+", default=(8, 48))
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output_dir.resolve()
    sizes = sorted(set(args.support_sizes))
    if not source.is_file():
        raise FileNotFoundError(source)
    if not sizes or sizes[0] <= 0:
        raise ValueError("support sizes must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {output}")

    import pyarrow.parquet as pq
    from opd_code.data import atomic_write_parquet

    source_rows = pq.read_table(source).to_pylist()
    identifiers = [prompt_id(row) for row in source_rows]
    if len(set(identifiers)) != len(source_rows):
        raise ValueError("candidate pool has duplicate extra_info.index values")
    candidate_provenance: dict[str, Any]
    if args.candidate_ids_json is None:
        rows = source_rows
        candidate_provenance = {
            "definition": "all source rows",
            "rows": len(rows),
        }
    else:
        candidate_path = args.candidate_ids_json.resolve()
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        requested = [str(value) for value in payload["problem_ids"]]
        if len(set(requested)) != len(requested):
            raise ValueError("candidate-id artifact contains duplicates")
        requested_set = set(requested)
        rows = [row for row in source_rows if prompt_id(row) in requested_set]
        found = {prompt_id(row) for row in rows}
        if found != requested_set:
            raise ValueError(
                f"candidate IDs missing from source: {sorted(requested_set - found)[:10]}"
            )
        candidate_provenance = {
            "definition": "frozen problem_ids subset of source",
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "rows": len(rows),
            "upstream_selection_method": payload.get("selection_method"),
            "upstream_selection_seed": payload.get("selection_seed"),
        }
    if sizes[-1] > len(rows):
        raise ValueError("support exceeds candidate pool")

    rng = np.random.Generator(np.random.PCG64(args.selection_seed))
    permutation = rng.permutation(len(rows))
    ordered = [rows[int(index)] for index in permutation]
    output.mkdir(parents=True)
    support_dir = output / "supports"
    schedule_dir = output / "schedules"
    support_dir.mkdir()
    schedule_dir.mkdir()

    support_manifest = {}
    schedule_manifest = {}
    previous_ids: set[str] = set()
    for size in sizes:
        support = ordered[:size]
        ids = {prompt_id(row) for row in support}
        if not previous_ids <= ids:
            raise AssertionError("uniform supports are not nested")
        previous_ids = ids
        support_path = support_dir / f"support_{size}.parquet"
        atomic_write_parquet(support, support_path)
        schedule, counts = materialize_schedule(
            support,
            steps=args.steps,
            batch_size=args.batch_size,
            selection_seed=args.selection_seed,
            training_seed=args.training_seed,
        )
        schedule_path = (
            schedule_dir / f"support_{size}_s{args.steps}_b{args.batch_size}.parquet"
        )
        atomic_write_parquet(schedule, schedule_path)
        lengths = [prompt_characters(row) for row in support]
        sources = Counter(str(row.get("data_source", "missing")) for row in support)
        support_manifest[str(size)] = {
            "path": str(support_path),
            "sha256": sha256(support_path),
            "rows": size,
            "prompt_ids": sorted(ids),
            "prompt_characters_mean": statistics.fmean(lengths),
            "prompt_characters_median": statistics.median(lengths),
            "data_sources": dict(sorted(sources.items())),
        }
        schedule_manifest[str(size)] = {
            "path": str(schedule_path),
            "sha256": sha256(schedule_path),
            "rows": len(schedule),
            "unique_prompts": len(counts),
            "presentations_min": min(counts.values()),
            "presentations_max": max(counts.values()),
        }

    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_before_training",
        "selection_method": "uniform_random_without_replacement",
        "selection_rng": {
            "implementation": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "numpy_version": np.__version__,
        },
        "selection_seed": args.selection_seed,
        "training_seed": args.training_seed,
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "rows": len(source_rows),
        },
        "candidate_pool": candidate_provenance,
        "support_sizes": sizes,
        "supports": support_manifest,
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "total_presentations": args.steps * args.batch_size,
            "schedule_order": (
                "balanced repeated presentation; SHA256 ranking within each round "
                "using training_seed, independent of selection_seed"
            ),
        },
        "schedules": schedule_manifest,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "SCHEDULES_VALIDATED.json",
        {
            "schema_version": SCHEMA,
            "status": "validated",
            "manifest_sha256": sha256(output / "manifest.json"),
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
