#!/usr/bin/env python3
"""Materialize nested OPD supports from a preregistered candidate ranking."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[2] / "outputs" / "opd_project_root"))
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCHEMA = "opd_ranked_selection_schedules_v1"


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
    return str(row["extra_info"]["index"])


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def materialize_schedule(
    support: Sequence[Mapping[str, Any]],
    *,
    method: str,
    selection_seed: int,
    training_seed: int,
    steps: int,
    batch_size: int,
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
        for source_row in eligible:
            identifier = prompt_id(source_row)
            row = copy.deepcopy(dict(source_row))
            extra = dict(row["extra_info"])
            extra["selection_benchmark"] = {
                "schema_version": SCHEMA,
                "selection_method": method,
                "selection_seed": selection_seed,
                "training_seed": training_seed,
                "support_size": len(support),
                "schedule_index": len(schedule),
                "presentation_index": counts[identifier],
                "optimizer_step": len(schedule) // batch_size,
                "position_in_step": len(schedule) % batch_size,
            }
            row["extra_info"] = extra
            schedule.append(row)
            counts[identifier] += 1
    if len(schedule) != total:
        raise AssertionError("schedule row count mismatch")
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("support presentation counts differ by more than one")
    return schedule, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-sizes", type=int, nargs="+", default=(8, 48))
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    source = args.source.resolve()
    ranking_path = args.ranking.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {output}")
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    method = str(ranking["selection_method"])
    selection_seed = int(ranking["selection_seed"])
    ordered_ids = [str(value) for value in ranking["ordered_prompt_ids"]]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("ranking contains duplicate prompt IDs")

    import pyarrow.parquet as pq
    from opd_code.data import atomic_write_parquet

    rows = pq.read_table(source).to_pylist()
    by_id = {prompt_id(row): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("source contains duplicate extra_info.index values")
    if set(ordered_ids) != set(by_id):
        raise ValueError("ranking and source candidate sets differ")
    sizes = sorted(set(args.support_sizes))
    if not sizes or sizes[0] < 1 or sizes[-1] > len(rows):
        raise ValueError("invalid support sizes")

    output.mkdir(parents=True)
    (output / "supports").mkdir()
    (output / "schedules").mkdir()
    supports = {}
    schedules = {}
    previous: set[str] = set()
    for size in sizes:
        selected_ids = ordered_ids[:size]
        if not previous <= set(selected_ids):
            raise AssertionError("ranked supports are not nested")
        previous = set(selected_ids)
        support = [by_id[identifier] for identifier in selected_ids]
        support_path = output / "supports" / f"support_{size}.parquet"
        atomic_write_parquet(support, support_path)
        schedule, counts = materialize_schedule(
            support,
            method=method,
            selection_seed=selection_seed,
            training_seed=args.training_seed,
            steps=args.steps,
            batch_size=args.batch_size,
        )
        schedule_path = (
            output
            / "schedules"
            / f"support_{size}_s{args.steps}_b{args.batch_size}.parquet"
        )
        atomic_write_parquet(schedule, schedule_path)
        supports[str(size)] = {
            "path": str(support_path),
            "sha256": sha256(support_path),
            "rows": size,
            "prompt_ids_in_rank_order": selected_ids,
        }
        schedules[str(size)] = {
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
        "selection_method": method,
        "selection_seed": selection_seed,
        "training_seed": args.training_seed,
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "rows": len(rows),
        },
        "ranking": {
            "path": str(ranking_path),
            "sha256": sha256(ranking_path),
            "schema_version": ranking.get("schema_version"),
        },
        "selection_cost": ranking.get("selection_cost"),
        "support_sizes": sizes,
        "supports": supports,
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "total_presentations": args.steps * args.batch_size,
        },
        "schedules": schedules,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "SCHEDULES_VALIDATED.json",
        {
            "schema_version": SCHEMA,
            "status": "validated",
            "manifest": str(output / "manifest.json"),
            "manifest_sha256": sha256(output / "manifest.json"),
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
