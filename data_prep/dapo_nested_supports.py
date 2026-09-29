#!/usr/bin/env python3
"""Freeze nested uniform-random supports and 100-step OPD schedules for a math pool.

Default target is the cleaned DAPO pool: the existing DeepSeek arms distill on DeepMath
level 6, which is *not* what JustRL was RL-trained on; the teacher saw DAPO-Math-17k.
These arms hold the pair, template, batch size, step budget, and seeds fixed and swap
only the prompt population, so the M-scaling curve can be read against a pool that
matches the teacher's own RL distribution.

The pool, support sizes, and labels are all overridable, so the identical draw procedure
can be applied to DeepMath level 6 for a same-procedure cross-pool control at equal M.

Supports are drawn by uniform random nesting (each support is the prefix of the next
under one seeded permutation), with no difficulty or gradient-based selection, so they
serve as the unbiased control for the criterion-based M=8 arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

# Directory laid out like the original project (data/opd, data/rl, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[1] / "outputs" / "opd_project_root"))
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_POOL = ROOT / "data/opd/dapo_math_clean_v1/pool.parquet"
DEFAULT_OUTPUT = ROOT / "data/opd/deepseek_1p5b_justrl_dapo_diversity_b256_s100_v1"
SUPPORT_SIZES = (8, 384)
SCHEMA_VERSION = "deepseek_1p5b_justrl_dapo_diversity_v1"
SELECTION_METHOD = "uniform random nested draw from the cleaned DAPO-Math pool"


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
    pool_manifest: Path,
    schema_version: str,
    selection_method: str,
    extra_key: str,
) -> dict[str, Any]:
    result = dict(row)
    extra = dict(row["extra_info"])
    previous = dict(extra.get(extra_key) or {})
    previous.update(
        {
            "schema_version": schema_version,
            "selection_method": selection_method,
            "selection_seed": selection_seed,
            "training_seed": training_seed,
            "support_size": support_size,
            "total_rollouts": total_rollouts,
            "presentation_index": presentation_index,
            "schedule_index": schedule_index,
            "optimizer_step": schedule_index // batch_size,
            "position_in_step": schedule_index % batch_size,
            "pool_manifest": str(pool_manifest),
        }
    )
    extra[extra_key] = previous
    result["extra_info"] = extra
    return result


def materialize_schedule(
    support: Sequence[Mapping[str, Any]],
    *,
    total_rollouts: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
    pool_manifest: Path,
    schema_version: str,
    selection_method: str,
    extra_key: str,
) -> tuple[list[dict[str, Any]], Counter[int]]:
    """Balanced-exposure schedule, identical in construction to the DeepMath arms."""
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
        prompt_index(row): base_presentations + (prompt_index(row) in extra_indices)
        for row in support
    }

    counts: Counter[int] = Counter()
    schedule: list[dict[str, Any]] = []
    for presentation in range(max(target_counts.values())):
        eligible = [
            row for row in support if target_counts[prompt_index(row)] > presentation
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
                    pool_manifest=pool_manifest,
                    schema_version=schema_version,
                    selection_method=selection_method,
                    extra_key=extra_key,
                )
            )
            counts[index] += 1

    if len(schedule) != total_rollouts:
        raise AssertionError("schedule has the wrong row count")
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("prompt presentation counts differ by more than one")
    return schedule, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-rollouts", type=int, default=25_600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--support-sizes",
        type=int,
        nargs="+",
        default=list(SUPPORT_SIZES),
        help="nested support sizes, ascending",
    )
    parser.add_argument("--schema-version", default=SCHEMA_VERSION)
    parser.add_argument("--selection-method", default=SELECTION_METHOD)
    parser.add_argument("--extra-key", default="dapo_prompt_diversity")
    parser.add_argument(
        "--allow-unvalidated-pool",
        action="store_true",
        help="accept a source pool that carries no POOL_VALIDATED.json marker",
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    support_sizes = tuple(sorted(arguments.support_sizes))
    if len(set(support_sizes)) != len(support_sizes):
        raise ValueError("support sizes must be distinct")

    pool_path = arguments.pool.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    if arguments.total_rollouts % arguments.batch_size:
        raise ValueError("total rollouts must be divisible by batch size")
    if not pool_path.is_file():
        raise FileNotFoundError(pool_path)

    pool_marker = pool_path.parent / "POOL_VALIDATED.json"
    if pool_marker.is_file():
        pool_validation = json.loads(pool_marker.read_text())
        if pool_validation["pool_sha256"] != sha256(pool_path):
            raise AssertionError("pool.parquet does not match its validated digest")
    elif arguments.allow_unvalidated_pool:
        pool_validation = {"pool_sha256": sha256(pool_path), "audit": None}
    else:
        raise FileNotFoundError(f"pool is not frozen/validated: {pool_marker}")

    manifest_path = output_dir / "manifest.json"
    marker_path = output_dir / "DIVERSITY_SCHEDULES_VALIDATED.json"
    if (manifest_path.exists() or marker_path.exists()) and not arguments.overwrite:
        raise FileExistsError("refusing to overwrite frozen schedules")

    import pyarrow.parquet as pq

    from opd_code.data import atomic_write_parquet

    rows = pq.read_table(pool_path).to_pylist()
    indices = [prompt_index(row) for row in rows]
    if len(set(indices)) != len(rows):
        raise AssertionError("pool contains duplicate prompt indices")
    largest = max(support_sizes)
    if len(rows) < largest:
        raise AssertionError(f"pool has {len(rows)} rows, need at least {largest}")

    # One seeded permutation; every support is a prefix of it, which makes nesting exact
    # by construction rather than by post-hoc check.
    permutation = sorted(
        rows,
        key=lambda row: stable_int(
            arguments.selection_seed, "dapo-support-order", prompt_index(row)
        ),
    )

    support_dir = output_dir / "supports" / f"seed_{arguments.selection_seed}"
    schedule_dir = output_dir / "schedules" / f"seed_{arguments.training_seed}"
    support_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir.mkdir(parents=True, exist_ok=True)

    steps = arguments.total_rollouts // arguments.batch_size
    supports: dict[int, list[Mapping[str, Any]]] = {}
    support_provenance: dict[str, object] = {}
    schedule_provenance: dict[str, object] = {}

    for size in support_sizes:
        support = permutation[:size]
        supports[size] = support
        support_path = support_dir / f"support_{size}.parquet"
        atomic_write_parquet([dict(row) for row in support], support_path)
        support_provenance[str(size)] = {
            "path": str(support_path),
            "rows": size,
            "sha256": sha256(support_path),
            "prompt_indices": sorted(prompt_index(row) for row in support),
        }

        schedule, counts = materialize_schedule(
            support,
            total_rollouts=arguments.total_rollouts,
            batch_size=arguments.batch_size,
            selection_seed=arguments.selection_seed,
            training_seed=arguments.training_seed,
            pool_manifest=pool_marker,
            schema_version=arguments.schema_version,
            selection_method=arguments.selection_method,
            extra_key=arguments.extra_key,
        )
        schedule_path = (
            schedule_dir / f"support_{size}_s{steps}_b{arguments.batch_size}.parquet"
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

    for smaller, larger in zip(support_sizes, support_sizes[1:]):
        if not {prompt_index(r) for r in supports[smaller]} <= {
            prompt_index(r) for r in supports[larger]
        }:
            raise AssertionError(f"M={smaller} is not nested in M={larger}")

    smallest = min(support_sizes)
    manifest = {
        "schema_version": arguments.schema_version,
        "status": "validated",
        "selection_method": arguments.selection_method,
        "selection_seed": arguments.selection_seed,
        "training_seed": arguments.training_seed,
        "batch_size": arguments.batch_size,
        "optimizer_steps": steps,
        "total_rollouts_per_arm": arguments.total_rollouts,
        "support_sizes": list(support_sizes),
        "nested": True,
        "pool": {
            "path": str(pool_path),
            "rows": len(rows),
            "sha256": pool_validation["pool_sha256"],
            "audit": pool_validation["audit"],
        },
        "supports": support_provenance,
        "schedules": schedule_provenance,
        "full_arm": {
            "path": str(pool_path),
            "rows": len(rows),
            "sha256": pool_validation["pool_sha256"],
            "shuffle": True,
            "note": (
                f"All {len(rows):,d} prompts are eligible; {steps} batches consume "
                f"{arguments.total_rollouts:,d} presentations under the training seed."
            ),
        },
        f"support_{smallest}_prompts": [
            {
                "index": prompt_index(row),
                "problem": row["prompt"][0]["content"],
                "answer": row["reward_model"]["ground_truth"],
            }
            for row in supports[smallest]
        ],
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        marker_path,
        {
            "schema_version": arguments.schema_version,
            "status": "validated",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
    )
    for size in support_sizes:
        entry = schedule_provenance[str(size)]
        print(
            f"[ok] M={size:<5d} {entry['rows']:,d} rows  "
            f"presentations {entry['presentations_min']}-{entry['presentations_max']}  "
            f"-> {entry['path']}"
        )
    print(f"[ok] M=full  {len(rows):,d} prompts, shuffled -> {pool_path}")
    print(f"[ok] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
