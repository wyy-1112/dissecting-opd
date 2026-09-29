#!/usr/bin/env python3
"""Extend the frozen Qwen3-4B Math supports to a 100-step OPD budget.

M=1, M=48, and M=3840 retain exactly the prompt identities selected for the
batch-256 Qwen3-4B student / GRPO-500 teacher experiment.  Only presentation
counts and within-round order are rematerialized for 25,600 presentations.
The full arm points at the original 57,046-row DeepMath population and relies
on the training seed and shuffle contract used by the original full run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Directory laid out like the original project (data/opd, data/rl, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[1] / "outputs" / "opd_project_root"))
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_UPSTREAM = (
    ROOT / "data/opd/qwen3_4b_math_diversity_b256_s15_v1"
)
DEFAULT_SOURCE = Path(
    "/path/to/code/"
    "G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "data/opd/deepseek_1p5b_justrl_math_diversity_b256_s100_v1"
)
SUPPORT_SIZES = (1, 48, 3840)
SCHEMA_VERSION = "deepseek_1p5b_justrl_math_diversity_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prompt_index(row: Mapping[str, Any]) -> int:
    return int(row["extra_info"]["index"])


def annotated_row(
    row: Mapping[str, Any],
    *,
    support_size: int,
    total_rollouts: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
    presentation_index: int,
    schedule_index: int,
    upstream_manifest: Path,
) -> dict[str, Any]:
    result = dict(row)
    extra = dict(row["extra_info"])
    previous = dict(extra.get("math_prompt_diversity") or {})
    previous.update(
        {
            "schema_version": SCHEMA_VERSION,
            "selection_method": "frozen_qwen3_4b_grpo500_support",
            "selection_seed": selection_seed,
            "training_seed": training_seed,
            "support_size": support_size,
            "total_rollouts": total_rollouts,
            "presentation_index": presentation_index,
            "schedule_index": schedule_index,
            "optimizer_step": schedule_index // batch_size,
            "position_in_step": schedule_index % batch_size,
            "upstream_manifest": str(upstream_manifest),
        }
    )
    extra["math_prompt_diversity"] = previous
    result["extra_info"] = extra
    return result


def materialize_schedule(
    support: Sequence[Mapping[str, Any]],
    *,
    total_rollouts: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
    upstream_manifest: Path,
) -> tuple[list[dict[str, Any]], Counter[int]]:
    support_size = len(support)
    base_presentations, remainder = divmod(total_rollouts, support_size)
    extra_indices = {
        prompt_index(row)
        for row in sorted(
            support,
            key=lambda row: stable_int(
                training_seed,
                "presentation-remainder",
                support_size,
                prompt_index(row),
            ),
        )[:remainder]
    }
    target_counts = {
        prompt_index(row): base_presentations
        + (prompt_index(row) in extra_indices)
        for row in support
    }

    counts: Counter[int] = Counter()
    schedule: list[dict[str, Any]] = []
    for presentation in range(max(target_counts.values())):
        eligible = [
            row
            for row in support
            if target_counts[prompt_index(row)] > presentation
        ]
        eligible.sort(
            key=lambda row: stable_int(
                training_seed,
                "schedule-round",
                support_size,
                presentation,
                prompt_index(row),
            )
        )
        for row in eligible:
            index = prompt_index(row)
            schedule.append(
                annotated_row(
                    row,
                    support_size=support_size,
                    total_rollouts=total_rollouts,
                    batch_size=batch_size,
                    selection_seed=selection_seed,
                    training_seed=training_seed,
                    presentation_index=counts[index],
                    schedule_index=len(schedule),
                    upstream_manifest=upstream_manifest,
                )
            )
            counts[index] += 1

    if len(schedule) != total_rollouts:
        raise AssertionError("schedule has the wrong row count")
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("prompt presentation counts differ by more than one")
    return schedule, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-rollouts", type=int, default=25_600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    upstream_root = args.upstream_root.expanduser().resolve()
    upstream_manifest = upstream_root / "manifest.json"
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.total_rollouts % args.batch_size:
        raise ValueError("total rollouts must be divisible by batch size")
    if not upstream_manifest.is_file():
        raise FileNotFoundError(upstream_manifest)
    if not source.is_file():
        raise FileNotFoundError(source)

    manifest_path = output_dir / "manifest.json"
    marker_path = output_dir / "DIVERSITY_SCHEDULES_VALIDATED.json"
    if (manifest_path.exists() or marker_path.exists()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite frozen schedules")
    schedule_dir = output_dir / "schedules" / f"seed_{args.training_seed}"
    schedule_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    from opd_code.data import atomic_write_parquet

    supports: dict[int, list[dict[str, Any]]] = {}
    support_provenance: dict[str, object] = {}
    schedule_provenance: dict[str, object] = {}
    for size in SUPPORT_SIZES:
        support_path = (
            upstream_root
            / "supports"
            / f"seed_{args.selection_seed}"
            / f"support_{size}.parquet"
        )
        if not support_path.is_file():
            raise FileNotFoundError(support_path)
        rows = pq.read_table(support_path).to_pylist()
        if len(rows) != size:
            raise AssertionError(f"{support_path} has {len(rows)} rows, expected {size}")
        indices = [prompt_index(row) for row in rows]
        if len(set(indices)) != size:
            raise AssertionError(f"{support_path} contains duplicate prompt indices")
        supports[size] = rows
        support_provenance[str(size)] = {
            "path": str(support_path),
            "rows": size,
            "sha256": sha256(support_path),
        }

        schedule, counts = materialize_schedule(
            rows,
            total_rollouts=args.total_rollouts,
            batch_size=args.batch_size,
            selection_seed=args.selection_seed,
            training_seed=args.training_seed,
            upstream_manifest=upstream_manifest,
        )
        steps = args.total_rollouts // args.batch_size
        schedule_path = (
            schedule_dir / f"support_{size}_s{steps}_b{args.batch_size}.parquet"
        )
        atomic_write_parquet(schedule, schedule_path)
        schedule_provenance[str(size)] = {
            "path": str(schedule_path),
            "rows": pq.read_metadata(schedule_path).num_rows,
            "unique_prompts": size,
            "presentations_min": min(counts.values()),
            "presentations_max": max(counts.values()),
            "sha256": sha256(schedule_path),
        }

    for smaller, larger in zip(SUPPORT_SIZES, SUPPORT_SIZES[1:]):
        smaller_indices = {prompt_index(row) for row in supports[smaller]}
        larger_indices = {prompt_index(row) for row in supports[larger]}
        if not smaller_indices <= larger_indices:
            raise AssertionError(f"M={smaller} is not nested in M={larger}")

    source_rows = pq.read_metadata(source).num_rows
    if source_rows != 57_046:
        raise AssertionError(f"expected 57,046 DeepMath rows, found {source_rows}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "selection_method": "frozen_qwen3_4b_grpo500_support",
        "selection_seed": args.selection_seed,
        "training_seed": args.training_seed,
        "batch_size": args.batch_size,
        "optimizer_steps": args.total_rollouts // args.batch_size,
        "total_rollouts_per_arm": args.total_rollouts,
        "support_sizes": list(SUPPORT_SIZES),
        "nested": True,
        "upstream_manifest": {
            "path": str(upstream_manifest),
            "sha256": sha256(upstream_manifest),
        },
        "supports": support_provenance,
        "schedules": schedule_provenance,
        "full_arm": {
            "path": str(source),
            "rows": source_rows,
            "sha256": sha256(source),
            "shuffle": True,
            "note": (
                "All 57,046 prompts are eligible; 100 batches consume 25,600 "
                "presentations under training seed 42."
            ),
        },
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        marker_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "validated",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
