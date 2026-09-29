#!/usr/bin/env python3
"""Materialize nested Qwen3-4B math prompt-diversity schedules.

The source population is the exact 25,600-prompt support consumed by the
existing full-data OPD run through optimizer step 25.  Smaller supports are
nested, nuisance-stratified random prefixes of that population.  Every arm has
25,600 prompt presentations and 25 batches of 1,024; prompt counts differ by at
most one when the presentation budget is not divisible by the support size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

# Directory laid out like the original project (data/opd, data/rl, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[1] / "outputs" / "opd_project_root"))
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA_VERSION = "opd_qwen3_4b_math_diversity_v1"
DEFAULT_SOURCE = Path(
    "/path/to/code/"
    "G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet"
)
DEFAULT_ROLLOUT_DIR = ROOT / "results/opd/qwen3_4b_gopd_math/rollouts"
DEFAULT_OUTPUT_DIR = ROOT / "data/opd/qwen3_4b_math_diversity_v1"
DEFAULT_SUPPORT_SIZES = (1, 48, 384, 3072, 25600)
PROMPT_LENGTH_BINS = 4
BASELINE_STEP_BINS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def load_baseline_metadata(
    rollout_dir: Path,
    *,
    max_step: int,
    batch_size: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    metadata: dict[int, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for step in range(1, max_step + 1):
        path = rollout_dir / f"{step}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        rows = 0
        with path.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                extra = row.get("extra_info") or {}
                index = int(extra["index"])
                if index in metadata:
                    raise ValueError(
                        f"baseline prompt index {index} occurs more than once"
                    )
                observed_step = int(row["step"])
                if observed_step != step:
                    raise ValueError(
                        f"{path}: expected step {step}, found {observed_step}"
                    )
                metadata[index] = {
                    "baseline_step": step,
                    "baseline_batch_index": int(row["batch_index_in_epoch"]),
                    "baseline_position_in_batch": rows,
                    "score": float(row["score"]),
                    "prompt_token_count": int(row["prompt_token_count"]),
                    "response_token_count": int(row["response_token_count"]),
                    "response_truncated": bool(row["response_truncated"]),
                    "k1_mean": float(row["k1_mean"]),
                }
                rows += 1
        if rows != batch_size:
            raise ValueError(f"{path}: expected {batch_size} rows, found {rows}")
        provenance[path.name] = {
            "path": str(path),
            "rows": rows,
            "sha256": digest.hexdigest(),
        }
    expected = max_step * batch_size
    if len(metadata) != expected:
        raise AssertionError(
            f"expected {expected} unique baseline prompts, found {len(metadata)}"
        )
    return metadata, provenance


def rank_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    value: Callable[[Mapping[str, Any]], float],
    bins: int,
    seed: int,
    tag: str,
) -> dict[int, int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            value(row),
            stable_int(seed, "rank-bin", tag, int(row["extra_info"]["index"])),
        ),
    )
    return {
        int(row["extra_info"]["index"]): min(
            bins - 1, rank * bins // len(ordered)
        )
        for rank, row in enumerate(ordered)
    }


def construct_nested_supports(
    population: Sequence[Mapping[str, Any]],
    metadata: Mapping[int, Mapping[str, Any]],
    *,
    support_sizes: Sequence[int],
    seed: int,
) -> tuple[
    dict[int, list[Mapping[str, Any]]],
    dict[int, tuple[int, int, int]],
]:
    sizes = tuple(sorted(set(support_sizes)))
    if not sizes or sizes[-1] != len(population):
        raise ValueError("largest support must equal the baseline population")
    if sizes[0] < 1:
        raise ValueError("support sizes must be positive")

    length_bins = rank_bins(
        population,
        value=lambda row: float(
            metadata[int(row["extra_info"]["index"])]["prompt_token_count"]
        ),
        bins=PROMPT_LENGTH_BINS,
        seed=seed,
        tag="prompt-length",
    )
    max_baseline_step = max(
        int(metadata[int(row["extra_info"]["index"])]["baseline_step"])
        for row in population
    )
    strata: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    row_strata: dict[int, tuple[int, int, int]] = {}
    for row in population:
        index = int(row["extra_info"]["index"])
        baseline = metadata[index]
        key = (
            min(
                BASELINE_STEP_BINS - 1,
                (int(baseline["baseline_step"]) - 1)
                * BASELINE_STEP_BINS
                // max_baseline_step,
            ),
            length_bins[index],
            int(float(baseline["score"]) >= 1.0),
        )
        strata[key].append(row)
        row_strata[index] = key

    # Build one randomized low-discrepancy order across strata.  At every
    # prefix, choose the stratum with the largest proportional allocation
    # deficit, then consume its next deterministically shuffled member.
    # This keeps all requested supports nested while bounding stratum-count
    # discrepancies much more tightly than independently rounded quotas.
    for key, members in strata.items():
        members.sort(
            key=lambda row: stable_int(
                seed, "within-stratum", key, int(row["extra_info"]["index"])
            )
        )
    population_size = len(population)
    selected_counts: Counter[tuple[int, int, int]] = Counter()
    next_offsets: Counter[tuple[int, int, int]] = Counter()
    ordered_population: list[Mapping[str, Any]] = []
    for prefix_size in range(1, population_size + 1):
        available = [
            key
            for key, members in strata.items()
            if next_offsets[key] < len(members)
        ]
        key = max(
            available,
            key=lambda candidate: (
                prefix_size * len(strata[candidate]) / population_size
                - selected_counts[candidate],
                -stable_int(
                    seed, "stratum-deficit-tie", prefix_size, candidate
                ),
            ),
        )
        ordered_population.append(strata[key][next_offsets[key]])
        next_offsets[key] += 1
        selected_counts[key] += 1
    supports = {size: ordered_population[:size] for size in sizes}
    for smaller, larger in zip(sizes, sizes[1:]):
        small = {
            int(row["extra_info"]["index"]) for row in supports[smaller]
        }
        large = {
            int(row["extra_info"]["index"]) for row in supports[larger]
        }
        if not small <= large:
            raise AssertionError(f"support {smaller} is not nested in {larger}")
    return supports, row_strata


def numeric_balance(
    population: Sequence[Mapping[str, Any]],
    support: Sequence[Mapping[str, Any]],
    getter: Callable[[Mapping[str, Any]], float],
) -> dict[str, float]:
    population_values = [getter(row) for row in population]
    support_values = [getter(row) for row in support]
    population_sd = statistics.pstdev(population_values)
    delta = statistics.fmean(support_values) - statistics.fmean(
        population_values
    )
    return {
        "population_mean": statistics.fmean(population_values),
        "population_sd": population_sd,
        "support_mean": statistics.fmean(support_values),
        "standardized_mean_difference": (
            delta / population_sd if population_sd else 0.0
        ),
    }


def categorical_balance(
    population: Sequence[Mapping[str, Any]],
    support: Sequence[Mapping[str, Any]],
    getter: Callable[[Mapping[str, Any]], Hashable],
) -> dict[str, Any]:
    population_counts = Counter(getter(row) for row in population)
    support_counts = Counter(getter(row) for row in support)
    categories = sorted(set(population_counts) | set(support_counts))
    differences = {
        str(category): (
            support_counts[category] / len(support)
            - population_counts[category] / len(population)
        )
        for category in categories
    }
    return {
        "counts": {str(key): value for key, value in support_counts.items()},
        "proportion_differences": differences,
        "max_absolute_proportion_difference": max(
            abs(value) for value in differences.values()
        ),
    }


def build_balance_report(
    population: Sequence[Mapping[str, Any]],
    supports: Mapping[int, Sequence[Mapping[str, Any]]],
    metadata: Mapping[int, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    def observed(row: Mapping[str, Any], field: str) -> float:
        return float(metadata[int(row["extra_info"]["index"])][field])

    max_baseline_step = max(
        int(observed(row, "baseline_step")) for row in population
    )
    arms: dict[str, Any] = {}
    for size, support in supports.items():
        numeric = {
            field: numeric_balance(
                population,
                support,
                lambda row, name=field: observed(row, name),
            )
            for field in (
                "baseline_step",
                "score",
                "prompt_token_count",
                "response_token_count",
                "k1_mean",
            )
        }
        arms[str(size)] = {
            "support_size": size,
            "numeric": numeric,
            "max_absolute_numeric_smd": max(
                abs(value["standardized_mean_difference"])
                for value in numeric.values()
            ),
            "baseline_step_bin": categorical_balance(
                population,
                support,
                lambda row: min(
                    BASELINE_STEP_BINS - 1,
                    (int(observed(row, "baseline_step")) - 1)
                    * BASELINE_STEP_BINS
                    // max_baseline_step,
                ),
            ),
            "observed_pass": categorical_balance(
                population,
                support,
                lambda row: int(observed(row, "score") >= 1.0),
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_seed": seed,
        "population_size": len(population),
        "selection_design": (
            "nested randomized low-discrepancy prefixes stratified by "
            f"{BASELINE_STEP_BINS} baseline-step bins x "
            f"{PROMPT_LENGTH_BINS} prompt-length bins x observed pass"
        ),
        "arms": arms,
    }


def annotated_row(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    support_size: int,
    selection_seed: int,
    training_seed: int,
    total_rollouts: int,
    batch_size: int,
    presentation_index: int,
    schedule_index: int,
) -> dict[str, Any]:
    extra = dict(row["extra_info"])
    extra["math_prompt_diversity"] = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": "nested_stratified_random",
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "support_size": support_size,
        "total_rollouts": total_rollouts,
        "baseline_step": int(baseline["baseline_step"]),
        "baseline_batch_index": int(baseline["baseline_batch_index"]),
        "baseline_position_in_batch": int(
            baseline["baseline_position_in_batch"]
        ),
        "baseline_score": float(baseline["score"]),
        "baseline_prompt_token_count": int(baseline["prompt_token_count"]),
        "baseline_response_token_count": int(
            baseline["response_token_count"]
        ),
        "presentation_index": presentation_index,
        "schedule_index": schedule_index,
        "optimizer_step": (
            schedule_index // batch_size if schedule_index >= 0 else -1
        ),
        "position_in_step": (
            schedule_index % batch_size if schedule_index >= 0 else -1
        ),
    }
    return {
        "data_source": row["data_source"],
        "prompt": row["prompt"],
        "ability": row["ability"],
        "reward_model": row["reward_model"],
        "extra_info": extra,
    }


def materialize_schedule(
    support: Sequence[Mapping[str, Any]],
    metadata: Mapping[int, Mapping[str, Any]],
    *,
    total_rollouts: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
) -> tuple[list[dict[str, Any]], Counter[int]]:
    support_size = len(support)
    reproduce_baseline_order = support_size == total_rollouts
    if reproduce_baseline_order:
        support = sorted(
            support,
            key=lambda row: (
                int(
                    metadata[int(row["extra_info"]["index"])]["baseline_step"]
                ),
                int(
                    metadata[int(row["extra_info"]["index"])][
                        "baseline_position_in_batch"
                    ]
                ),
            ),
        )
    base_presentations, remainder = divmod(total_rollouts, support_size)
    extra_indices = {
        int(row["extra_info"]["index"])
        for row in sorted(
            support,
            key=lambda row: stable_int(
                training_seed,
                "presentation-remainder",
                support_size,
                int(row["extra_info"]["index"]),
            ),
        )[:remainder]
    }
    target_counts = {
        int(row["extra_info"]["index"]): base_presentations
        + (int(row["extra_info"]["index"]) in extra_indices)
        for row in support
    }
    max_presentations = max(target_counts.values())
    presentation_counts: Counter[int] = Counter()
    schedule: list[dict[str, Any]] = []
    for presentation in range(max_presentations):
        eligible = [
            row
            for row in support
            if target_counts[int(row["extra_info"]["index"])] > presentation
        ]
        if not reproduce_baseline_order:
            eligible.sort(
                key=lambda row: stable_int(
                    training_seed,
                    "schedule-round",
                    support_size,
                    presentation,
                    int(row["extra_info"]["index"]),
                )
            )
        for row in eligible:
            index = int(row["extra_info"]["index"])
            schedule.append(
                annotated_row(
                    row,
                    metadata[index],
                    support_size=support_size,
                    selection_seed=selection_seed,
                    training_seed=training_seed,
                    total_rollouts=total_rollouts,
                    batch_size=batch_size,
                    presentation_index=presentation_counts[index],
                    schedule_index=len(schedule),
                )
            )
            presentation_counts[index] += 1
    if len(schedule) != total_rollouts:
        raise AssertionError("schedule has the wrong number of rows")
    if max(presentation_counts.values()) - min(
        presentation_counts.values()
    ) > 1:
        raise AssertionError("prompt presentation counts differ by more than 1")
    return schedule, presentation_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--baseline-rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--support-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SUPPORT_SIZES,
    )
    parser.add_argument("--total-rollouts", type=int, default=25600)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--baseline-steps", type=int, default=25)
    parser.add_argument("--baseline-batch-size", type=int, default=1024)
    parser.add_argument(
        "--baseline-pool-size",
        type=int,
        default=None,
        help="Use this many prompts from the baseline rollout prefix.",
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    rollout_dir = args.baseline_rollout_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sizes = tuple(sorted(set(args.support_sizes)))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if args.total_rollouts % args.batch_size:
        raise ValueError("total rollouts must be divisible by training batch size")
    if sizes[-1] != args.total_rollouts:
        raise ValueError("largest support must equal total rollouts")
    baseline_pool_size = args.baseline_pool_size or args.total_rollouts
    if baseline_pool_size != args.total_rollouts:
        raise ValueError(
            "baseline pool size must equal total rollouts for the full-support arm"
        )
    if baseline_pool_size > args.baseline_steps * args.baseline_batch_size:
        raise ValueError("baseline rollout prefix is smaller than requested pool")

    manifest_path = output_dir / "manifest.json"
    marker_path = output_dir / "DIVERSITY_SCHEDULES_VALIDATED.json"
    if (manifest_path.exists() or marker_path.exists()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite frozen diversity schedules")
    output_dir.mkdir(parents=True, exist_ok=True)
    support_dir = output_dir / "supports" / f"seed_{args.selection_seed}"
    schedule_dir = output_dir / "schedules" / f"seed_{args.training_seed}"
    support_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    from opd_code.data import atomic_write_parquet

    metadata, rollout_provenance = load_baseline_metadata(
        rollout_dir,
        max_step=args.baseline_steps,
        batch_size=args.baseline_batch_size,
    )
    baseline_order = sorted(
        metadata,
        key=lambda index: (
            int(metadata[index]["baseline_step"]),
            int(metadata[index]["baseline_position_in_batch"]),
        ),
    )
    selected_indices = baseline_order[:baseline_pool_size]
    metadata = {index: metadata[index] for index in selected_indices}
    source_rows = pq.read_table(source_path).to_pylist()
    by_index = {
        int(row["extra_info"]["index"]): row for row in source_rows
    }
    if len(by_index) != len(source_rows):
        raise ValueError("source contains duplicate extra_info.index values")
    missing = sorted(set(metadata) - set(by_index))
    if missing:
        raise ValueError(f"baseline indexes missing from source: {missing[:10]}")
    population = [by_index[index] for index in sorted(metadata)]
    supports, strata = construct_nested_supports(
        population,
        metadata,
        support_sizes=sizes,
        seed=args.selection_seed,
    )
    balance_report = build_balance_report(
        population,
        supports,
        metadata,
        seed=args.selection_seed,
    )
    balance_path = output_dir / "balance_report.json"
    atomic_json(balance_path, balance_report)

    previous_ids: set[int] = set()
    membership_rows: list[dict[str, Any]] = []
    support_outputs: dict[str, Any] = {}
    schedule_outputs: dict[str, Any] = {}
    steps = args.total_rollouts // args.batch_size
    for size in sizes:
        support = supports[size]
        support_ids = {
            int(row["extra_info"]["index"]) for row in support
        }
        for row in support:
            index = int(row["extra_info"]["index"])
            if index not in previous_ids:
                membership_rows.append(
                    {
                        "index": index,
                        "minimum_support_size": size,
                        "stratum": list(strata[index]),
                        "baseline": metadata[index],
                        "selection_seed": args.selection_seed,
                    }
                )
        support_rows = [
            annotated_row(
                row,
                metadata[int(row["extra_info"]["index"])],
                support_size=size,
                selection_seed=args.selection_seed,
                training_seed=args.training_seed,
                total_rollouts=args.total_rollouts,
                batch_size=args.batch_size,
                presentation_index=-1,
                schedule_index=-1,
            )
            for row in support
        ]
        support_path = support_dir / f"support_{size}.parquet"
        atomic_write_parquet(support_rows, support_path)
        support_outputs[str(size)] = {
            "path": str(support_path),
            "rows": pq.read_metadata(support_path).num_rows,
            "sha256": sha256(support_path),
        }

        schedule, counts = materialize_schedule(
            support,
            metadata,
            total_rollouts=args.total_rollouts,
            batch_size=args.batch_size,
            selection_seed=args.selection_seed,
            training_seed=args.training_seed,
        )
        schedule_path = (
            schedule_dir / f"support_{size}_s{steps}_b{args.batch_size}.parquet"
        )
        atomic_write_parquet(schedule, schedule_path)
        schedule_metadata = pq.read_metadata(schedule_path)
        if schedule_metadata.num_rows != args.total_rollouts:
            raise AssertionError("written schedule row count mismatch")
        schedule_outputs[str(size)] = {
            "path": str(schedule_path),
            "rows": schedule_metadata.num_rows,
            "unique_prompts": len(counts),
            "presentations_min": min(counts.values()),
            "presentations_max": max(counts.values()),
            "sha256": sha256(schedule_path),
        }
        previous_ids = support_ids
        del schedule

    membership_rows.sort(
        key=lambda row: (
            int(row["minimum_support_size"]),
            stable_int(args.selection_seed, "membership", int(row["index"])),
        )
    )
    membership_path = output_dir / "support_membership.jsonl"
    atomic_jsonl(membership_path, membership_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "selection_method": "nested_stratified_random",
        "source_population": {
            "path": str(source_path),
            "rows": len(source_rows),
            "sha256": sha256(source_path),
        },
        "baseline_support": {
            "run": str(ROOT / "results/opd/qwen3_4b_gopd_math"),
            "steps": args.baseline_steps,
            "rollout_batch_size": args.baseline_batch_size,
            "rows": len(population),
            "rollout_files": rollout_provenance,
        },
        "selection_seed": args.selection_seed,
        "training_seed": args.training_seed,
        "support_sizes": list(sizes),
        "nested": True,
        "total_rollouts_per_arm": args.total_rollouts,
        "batch_size": args.batch_size,
        "optimizer_steps": steps,
        "balance_report": {
            "path": str(balance_path),
            "sha256": sha256(balance_path),
        },
        "support_membership": {
            "path": str(membership_path),
            "sha256": sha256(membership_path),
        },
        "supports": support_outputs,
        "schedules": schedule_outputs,
    }
    atomic_json(manifest_path, manifest)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }
    atomic_json(marker_path, marker)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
