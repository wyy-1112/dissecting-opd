#!/usr/bin/env python3
"""Extend the validated M=48 schedule to a 60-update rollout-budget curve.

The first 3,840 rows are copied logically from the completed B=256 diversity
arm.  Later rows continue the same deterministic 48-prompt round-robin order,
so every milestone is a prefix of one training trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
import os
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

# Directory laid out like the original project (data/opd, data/rl, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[1] / "outputs" / "opd_project_root"))
SRC = Path(__file__).resolve().parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepmath_nested_supports import (  # noqa: E402
    atomic_json,
    materialize_schedule,
    sha256,
)
from opd_code.data import atomic_write_parquet  # noqa: E402

DEFAULT_DATA_ROOT = (
    ROOT / "data/opd/qwen3_4b_math_diversity_b256_s15_v1"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "data/opd/qwen3_4b_math_m48_rollout_curve_b256_s60_v1"
)
DEFAULT_SUPPORT = DEFAULT_DATA_ROOT / "supports/seed_42/support_48.parquet"
DEFAULT_REFERENCE = (
    DEFAULT_DATA_ROOT
    / "schedules/seed_42/support_48_s15_b256.parquet"
)


def prompt_index(row: Mapping[str, Any]) -> int:
    return int(row["extra_info"]["index"])


def baseline_metadata(
    support: list[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for row in support:
        index = prompt_index(row)
        diversity = row["extra_info"]["math_prompt_diversity"]
        metadata[index] = {
            "baseline_step": int(diversity["baseline_step"]),
            "baseline_batch_index": int(
                diversity["baseline_batch_index"]
            ),
            "baseline_position_in_batch": int(
                diversity["baseline_position_in_batch"]
            ),
            "score": float(diversity["baseline_score"]),
            "prompt_token_count": int(
                diversity["baseline_prompt_token_count"]
            ),
            "response_token_count": int(
                diversity["baseline_response_token_count"]
            ),
        }
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument(
        "--reference-schedule", type=Path, default=DEFAULT_REFERENCE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--support-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimizer-steps", type=int, default=60)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--milestones", type=int, nargs="+", default=(3, 6, 15, 30, 60)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.support.is_file():
        raise FileNotFoundError(args.support)
    if not args.reference_schedule.is_file():
        raise FileNotFoundError(args.reference_schedule)
    if args.optimizer_steps <= 0 or args.batch_size <= 0:
        raise ValueError("optimizer steps and batch size must be positive")

    total_rollouts = args.optimizer_steps * args.batch_size
    support = pq.read_table(args.support).to_pylist()
    reference = pq.read_table(args.reference_schedule).to_pylist()
    if len(support) != args.support_size:
        raise ValueError(
            f"support has {len(support)} rows, expected {args.support_size}"
        )
    if len(reference) != 15 * args.batch_size:
        raise ValueError("reference schedule is not the completed 15-step arm")
    if total_rollouts < len(reference):
        raise ValueError("extended schedule cannot be shorter than reference")
    if total_rollouts % args.support_size:
        raise ValueError("rollout budget must divide evenly over M=48")
    if any(step <= 0 or step > args.optimizer_steps for step in args.milestones):
        raise ValueError("milestone outside the training horizon")

    generated, counts = materialize_schedule(
        support,
        baseline_metadata(support),
        total_rollouts=total_rollouts,
        batch_size=args.batch_size,
        selection_seed=args.selection_seed,
        training_seed=args.training_seed,
    )
    reference_indices = [prompt_index(row) for row in reference]
    generated_prefix_indices = [
        prompt_index(row) for row in generated[: len(reference)]
    ]
    if reference_indices != generated_prefix_indices:
        raise AssertionError("extended prompt-index prefix changed")

    # Preserve every field consumed by the completed first 15 updates.
    generated[: len(reference)] = reference
    if generated[: len(reference)] != reference:
        raise AssertionError("reference rows were not preserved exactly")

    final_counts = Counter(prompt_index(row) for row in generated)
    expected_presentations = total_rollouts // args.support_size
    if set(final_counts.values()) != {expected_presentations}:
        raise AssertionError("M=48 presentations are not exactly balanced")
    if counts != final_counts:
        raise AssertionError("materializer and final presentation counts differ")

    schedule_dir = args.output_root / "schedules/seed_42"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    output = (
        schedule_dir
        / f"support_48_s{args.optimizer_steps}_b{args.batch_size}.parquet"
    )
    atomic_write_parquet(generated, output)
    written = pq.read_table(output).to_pylist()
    if written[: len(reference)] != reference:
        raise AssertionError("written parquet did not preserve reference prefix")

    milestones = {
        str(step): {
            "optimizer_step": step,
            "rollouts": step * args.batch_size,
            "presentations_per_prompt": (step * args.batch_size)
            / args.support_size,
        }
        for step in args.milestones
    }
    manifest = {
        "schema_version": "opd_qwen3_4b_math_m48_rollout_curve_v1",
        "status": "validated",
        "support_size": args.support_size,
        "batch_size": args.batch_size,
        "optimizer_steps": args.optimizer_steps,
        "total_rollouts": total_rollouts,
        "selection_seed": args.selection_seed,
        "training_seed": args.training_seed,
        "milestones": milestones,
        "support": {
            "path": str(args.support.resolve()),
            "rows": len(support),
            "sha256": sha256(args.support),
        },
        "reference_schedule": {
            "path": str(args.reference_schedule.resolve()),
            "rows": len(reference),
            "sha256": sha256(args.reference_schedule),
        },
        "schedule": {
            "path": str(output.resolve()),
            "rows": len(written),
            "sha256": sha256(output),
            "prefix_rows": len(reference),
            "prefix_prompt_indices_exact": True,
            "prefix_rows_logically_exact": True,
            "presentations_per_prompt": expected_presentations,
        },
    }
    manifest_path = args.output_root / "manifest.json"
    atomic_json(manifest_path, manifest)
    marker = args.output_root / "ROLLOUT_CURVE_SCHEDULE_VALIDATED.json"
    atomic_json(
        marker,
        {
            "status": "validated",
            "manifest": str(manifest_path.resolve()),
            "schedule_sha256": manifest["schedule"]["sha256"],
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
