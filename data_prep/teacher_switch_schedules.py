#!/usr/bin/env python3
"""Freeze support/training-seed schedules for the 4B teacher-switch grid."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


# Directory laid out like the original project (data/opd, data/rl, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[1] / "outputs" / "opd_project_root"))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from deepmath_nested_supports import (  # noqa: E402
    load_baseline_metadata,
    materialize_schedule as math_schedule,
)
from eurus_code_nested_supports import (  # noqa: E402
    construct_nested_supports as code_supports,
    materialize_schedule as code_schedule,
)
from opd_code.data import atomic_write_parquet  # noqa: E402


OUT = ROOT / "data/opd/qwen3_4b_teacher_switch_grid_m48_b256_s25_v1"
MIXTURE_DATA = (
    ROOT
    / "data/opd/qwen3_30b_to_4b_deepmath_gsm8k_mixture_m48_b256_s15_v1"
)
MATH_SOURCE = Path(
    "/path/to/code/"
    "G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
)
MATH_BASELINE = ROOT / "results/opd/qwen3_4b_gopd_math/rollouts"
MATH_SUPPORT42 = (
    ROOT
    / "data/opd/qwen3_4b_math_diversity_b256_s15_v1"
    / "supports/seed_42/support_48.parquet"
)
MATH_SCHEDULE42 = (
    ROOT
    / "data/opd/qwen3_4b_math_m48_rollout_curve_b256_s60_v1"
    / "schedules/seed_42/support_48_s60_b256.parquet"
)
CODE_SOURCE = ROOT / "data/rl/eurus_code_grpo/train.parquet"
CODE_SUPPORT42 = (
    ROOT
    / "data/opd/qwen3_4b_code_diversity_b256_s100_v1"
    / "supports/seed_42/support_48.parquet"
)
CODE_SCHEDULE42 = (
    ROOT
    / "data/opd/qwen3_4b_code_diversity_b256_s100_v1"
    / "schedules/seed_42/support_48_s100_b256.parquet"
)
BLOCKS = ((43, 42), (44, 42), (42, 43), (42, 44))
TOTAL = 25 * 256
BATCH = 256


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def math_metadata() -> dict[int, dict[str, Any]]:
    metadata, _ = load_baseline_metadata(
        MATH_BASELINE,
        max_step=4,
        batch_size=1024,
    )
    order = sorted(
        metadata,
        key=lambda index: (
            int(metadata[index]["baseline_step"]),
            int(metadata[index]["baseline_position_in_batch"]),
        ),
    )[:3840]
    return {index: metadata[index] for index in order}


def math_support(seed: int) -> list[dict[str, Any]]:
    if seed == 42:
        path = MATH_SUPPORT42
    else:
        path = (
            MIXTURE_DATA
            / f"selection_seed_{seed}/rankings/deepmath_order_48.parquet"
        )
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 48:
        raise ValueError(path)
    return rows


def code_support(seed: int) -> tuple[list[dict[str, Any]], dict[int, tuple[str, int]]]:
    population = pq.read_table(CODE_SOURCE).to_pylist()
    supports, strata = code_supports(
        population,
        support_sizes=(48, len(population)),
        seed=seed,
    )
    return list(supports[48]), strata


def main() -> None:
    marker = OUT / "TEACHER_SWITCH_SCHEDULES_VALIDATED.json"
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen grid: {OUT}")
    OUT.mkdir(parents=True)
    metadata = math_metadata()
    entries: dict[str, Any] = {}

    entries["42:42"] = {
        "support_seed": 42,
        "training_seed": 42,
        "math_support": str(MATH_SUPPORT42),
        "math_support_sha256": sha256(MATH_SUPPORT42),
        "math_schedule": str(MATH_SCHEDULE42),
        "math_schedule_sha256": sha256(MATH_SCHEDULE42),
        "math_consumed_rows": TOTAL,
        "code_support": str(CODE_SUPPORT42),
        "code_support_sha256": sha256(CODE_SUPPORT42),
        "code_schedule": str(CODE_SCHEDULE42),
        "code_schedule_sha256": sha256(CODE_SCHEDULE42),
        "code_consumed_rows": TOTAL,
        "reused_original_four_grid": True,
    }

    code_cache: dict[int, tuple[list[dict[str, Any]], dict[int, tuple[str, int]]]] = {}
    for support_seed, training_seed in BLOCKS:
        destination = OUT / f"support_seed_{support_seed}/train_seed_{training_seed}"
        destination.mkdir(parents=True)

        math_rows = math_support(support_seed)
        math_support_path = destination / "deepmath48_support.parquet"
        atomic_write_parquet(math_rows, math_support_path)
        math_rows_schedule, _ = math_schedule(
            math_rows,
            metadata,
            total_rollouts=TOTAL,
            batch_size=BATCH,
            selection_seed=support_seed,
            training_seed=training_seed,
        )
        math_schedule_path = destination / "deepmath48_s25_b256.parquet"
        atomic_write_parquet(math_rows_schedule, math_schedule_path)

        if support_seed not in code_cache:
            code_cache[support_seed] = code_support(support_seed)
        code_rows, code_strata = code_cache[support_seed]
        code_support_path = destination / "code48_support.parquet"
        atomic_write_parquet(code_rows, code_support_path)
        code_rows_schedule, _ = code_schedule(
            code_rows,
            code_strata,
            total_rollouts=TOTAL,
            batch_size=BATCH,
            selection_seed=support_seed,
            training_seed=training_seed,
        )
        code_schedule_path = destination / "code48_s25_b256.parquet"
        atomic_write_parquet(code_rows_schedule, code_schedule_path)

        key = f"{support_seed}:{training_seed}"
        entries[key] = {
            "support_seed": support_seed,
            "training_seed": training_seed,
            "math_support": str(math_support_path),
            "math_support_sha256": sha256(math_support_path),
            "math_schedule": str(math_schedule_path),
            "math_schedule_sha256": sha256(math_schedule_path),
            "math_consumed_rows": TOTAL,
            "code_support": str(code_support_path),
            "code_support_sha256": sha256(code_support_path),
            "code_schedule": str(code_schedule_path),
            "code_schedule_sha256": sha256(code_schedule_path),
            "code_consumed_rows": TOTAL,
            "reused_original_four_grid": False,
        }
        atomic_json(destination / "manifest.json", entries[key])

    payload = {
        "schema_version": "qwen3_4b_teacher_switch_grid_v1",
        "status": "frozen_before_training",
        "support_size": 48,
        "steps": 25,
        "batch_size": 256,
        "presentations": TOTAL,
        "blocks": entries,
        "support_policy": {
            "deepmath": "nested_stratified_random on the frozen 3840-prompt population",
            "code": "nested_source_length_stratified_random on Eurus Code",
        },
        "seed_effects": (
            "support_seed changes support IDs; training_seed changes SHA256 "
            "schedule order and is also passed to the trainer/rollout workers"
        ),
    }
    atomic_json(OUT / "manifest.json", payload)
    atomic_json(
        marker,
        {
            "status": "validated",
            "manifest": str(OUT / "manifest.json"),
            "manifest_sha256": sha256(OUT / "manifest.json"),
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
