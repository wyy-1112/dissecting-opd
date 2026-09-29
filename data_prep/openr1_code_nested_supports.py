#!/usr/bin/env python3
"""Materialize nested, covariate-stratified random OPD diversity schedules."""
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

SCHEMA_VERSION = "opd_prompt_diversity_v1"
DEFAULT_SUPPORT_SIZES = (48, 384, 3072)
STRATIFICATION_BINS = 6


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


def problem_id(row: Mapping[str, Any]) -> str:
    return str(row["extra_info"]["problem_id"])


def source_subset(row: Mapping[str, Any]) -> str:
    return str(row["extra_info"]["source_subset"])


def prompt_length(row: Mapping[str, Any]) -> float:
    extra = row["extra_info"]
    value = extra.get("prompt_token_count", extra.get("prompt_tokens"))
    if value is None:
        raise KeyError("extra_info must contain prompt_token_count or prompt_tokens")
    return float(value)


def reward_stat(row: Mapping[str, Any], name: str) -> float:
    return float(row["extra_info"]["offline_screen"][name])


def apportion(
    counts: Mapping[Hashable, int],
    target: int,
    *,
    seed: int,
    tag: str,
) -> dict[Hashable, int]:
    """Use randomized-tie largest remainder apportionment."""
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot apportion target={target} from total={total}")
    if not counts:
        if target:
            raise ValueError("cannot apportion a non-zero target from no groups")
        return {}

    raw = {key: target * count / total for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = target - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            -(raw[key] - quotas[key]),
            stable_int(seed, "quota", tag, repr(key)),
        ),
    )
    for key in ranked[:remainder]:
        quotas[key] += 1
    if sum(quotas.values()) != target:
        raise AssertionError("apportionment does not sum to target")
    if any(quotas[key] > counts[key] for key in counts):
        raise AssertionError("apportionment exceeds a group population")
    return quotas


def rank_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    value: Callable[[Mapping[str, Any]], float],
    bins: int,
    seed: int,
    tag: str,
) -> dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            value(row),
            stable_int(seed, "rank", tag, problem_id(row)),
        ),
    )
    return {
        problem_id(row): min(bins - 1, rank * bins // len(ordered))
        for rank, row in enumerate(ordered)
    }


def construct_nested_supports(
    rows: Sequence[Mapping[str, Any]],
    *,
    support_sizes: Sequence[int],
    seed: int,
    bins: int = STRATIFICATION_BINS,
) -> tuple[
    dict[int, list[Mapping[str, Any]]],
    dict[str, tuple[str, int, int]],
]:
    if not rows:
        raise ValueError("source pool is empty")
    sizes = tuple(sorted(set(support_sizes)))
    if not sizes or sizes[-1] != len(rows):
        raise ValueError("largest support must equal the source-pool size")
    if sizes[0] < 1:
        raise ValueError("support sizes must be positive")

    identifiers = [problem_id(row) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("source pool contains duplicate problem_id values")

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[source_subset(row)].append(row)

    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for source, source_rows in by_source.items():
        for label, getter in (
            ("length", prompt_length),
            ("reward", lambda row: reward_stat(row, "reward_mean")),
        ):
            assigned = rank_bins(
                source_rows,
                value=getter,
                bins=bins,
                seed=seed,
                tag=f"{source}:{label}",
            )
            for identifier, bin_index in assigned.items():
                labels[identifier][label] = bin_index

    strata: dict[
        tuple[str, int, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    row_strata: dict[str, tuple[str, int, int]] = {}
    for row in rows:
        identifier = problem_id(row)
        key = (
            source_subset(row),
            labels[identifier]["length"],
            labels[identifier]["reward"],
        )
        strata[key].append(row)
        row_strata[identifier] = key

    for key, members in strata.items():
        members.sort(
            key=lambda row: stable_int(
                seed, "within-stratum", key, problem_id(row)
            )
        )

    source_counts = {
        source: len(source_rows) for source, source_rows in by_source.items()
    }
    quotas_by_size: dict[int, dict[tuple[str, int, int], int]] = {}
    for size in sizes:
        source_quotas = apportion(
            source_counts,
            size,
            seed=seed,
            tag=f"source:{size}",
        )
        quotas: dict[tuple[str, int, int], int] = {}
        for source, source_quota in source_quotas.items():
            source_strata = {
                key: len(members)
                for key, members in strata.items()
                if key[0] == source
            }
            quotas.update(
                apportion(
                    source_strata,
                    source_quota,
                    seed=seed,
                    tag=f"stratum:{source}:{size}",
                )
            )
        quotas_by_size[size] = quotas

    for smaller, larger in zip(sizes, sizes[1:]):
        violations = [
            key
            for key in strata
            if quotas_by_size[smaller][key] > quotas_by_size[larger][key]
        ]
        if violations:
            raise ValueError(
                "stratified quotas are not nested for "
                f"{smaller}->{larger}: {violations[:5]}"
            )

    supports: dict[int, list[Mapping[str, Any]]] = {}
    for size in sizes:
        selected = [
            row
            for key, members in strata.items()
            for row in members[: quotas_by_size[size][key]]
        ]
        selected.sort(
            key=lambda row: stable_int(
                seed, "support-order", size, problem_id(row)
            )
        )
        if len(selected) != size:
            raise AssertionError(f"support {size} has {len(selected)} rows")
        supports[size] = selected

    for smaller, larger in zip(sizes, sizes[1:]):
        small_ids = {problem_id(row) for row in supports[smaller]}
        large_ids = {problem_id(row) for row in supports[larger]}
        if not small_ids <= large_ids:
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
    delta = statistics.fmean(support_values) - statistics.fmean(population_values)
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
    getter: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    population_counts = Counter(getter(row) for row in population)
    support_counts = Counter(getter(row) for row in support)
    categories = sorted(set(population_counts) | set(support_counts))
    population_size = len(population)
    support_size = len(support)
    proportions = {
        category: {
            "population": population_counts[category] / population_size,
            "support": support_counts[category] / support_size,
        }
        for category in categories
    }
    return {
        "counts": dict(sorted(support_counts.items())),
        "max_absolute_proportion_difference": max(
            abs(values["support"] - values["population"])
            for values in proportions.values()
        ),
        "proportions": proportions,
    }


def build_balance_report(
    population: Sequence[Mapping[str, Any]],
    supports: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    bins: int,
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for size, support in supports.items():
        numeric = {
            "prompt_token_count": numeric_balance(
                population, support, prompt_length
            ),
            "offline_reward_mean": numeric_balance(
                population,
                support,
                lambda row: reward_stat(row, "reward_mean"),
            ),
            "offline_reward_std": numeric_balance(
                population,
                support,
                lambda row: reward_stat(row, "reward_std"),
            ),
            "offline_full_pass_rate": numeric_balance(
                population,
                support,
                lambda row: reward_stat(row, "full_pass_rollouts")
                / reward_stat(row, "num_rollouts"),
            ),
        }
        arms[str(size)] = {
            "support_size": size,
            "numeric": numeric,
            "source_subset": categorical_balance(
                population, support, source_subset
            ),
            "difficulty": categorical_balance(
                population,
                support,
                lambda row: str(row["extra_info"].get("difficulty", "missing")),
            ),
            "max_absolute_numeric_smd": max(
                abs(values["standardized_mean_difference"])
                for values in numeric.values()
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_design": (
            "hierarchical proportional allocation by source_subset, then "
            f"source-conditional prompt-length {bins}-quantile x "
            f"offline-reward-mean {bins}-quantile; random within strata"
        ),
        "stratification_bins": bins,
        "selection_seed": seed,
        "population_size": len(population),
        "arms": arms,
    }


def training_row(
    row: Mapping[str, Any],
    *,
    support_size: int,
    total_rollouts: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
    presentation_index: int,
    schedule_index: int,
) -> dict[str, Any]:
    extra = dict(row["extra_info"])
    extra["prompt_diversity"] = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": "nested_stratified_random",
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "support_size": support_size,
        "total_rollouts": total_rollouts,
        "presentation_index": presentation_index,
        "schedule_index": schedule_index,
        "optimizer_step": schedule_index // batch_size,
        "position_in_step": schedule_index % batch_size,
    }
    return {
        "schema_version": row["schema_version"],
        "sample_id": row["sample_id"],
        "data_source": row["data_source"],
        "prompt": row["prompt"],
        "ability": row["ability"],
        "reward_model": row["reward_model"],
        "extra_info": extra,
    }


def materialize_schedule(
    support: Sequence[Mapping[str, Any]],
    *,
    total_rollouts: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
) -> list[dict[str, Any]]:
    support_size = len(support)
    base_presentations, remainder = divmod(total_rollouts, support_size)
    extra_ids = {
        problem_id(row)
        for row in sorted(
            support,
            key=lambda row: stable_int(
                training_seed,
                "presentation-remainder",
                support_size,
                problem_id(row),
            ),
        )[:remainder]
    }
    target_counts = {
        problem_id(row): base_presentations + (problem_id(row) in extra_ids)
        for row in support
    }
    schedule: list[dict[str, Any]] = []
    for presentation in range(max(target_counts.values())):
        ordered = sorted(
            (
                row
                for row in support
                if target_counts[problem_id(row)] > presentation
            ),
            key=lambda row: stable_int(
                training_seed,
                "schedule",
                support_size,
                presentation,
                problem_id(row),
            ),
        )
        for row in ordered:
            schedule.append(
                training_row(
                    row,
                    support_size=support_size,
                    total_rollouts=total_rollouts,
                    batch_size=batch_size,
                    selection_seed=selection_seed,
                    training_seed=training_seed,
                    presentation_index=presentation,
                    schedule_index=len(schedule),
                )
            )
    if len(schedule) != total_rollouts:
        raise AssertionError("schedule has the wrong number of rows")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/opd/rl_exposure_3072/seen_3072.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/opd/prompt_diversity_v1",
    )
    parser.add_argument(
        "--support-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SUPPORT_SIZES,
    )
    parser.add_argument("--total-rollouts", type=int, default=30720)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--selection-seed", type=int, default=20260818)
    parser.add_argument("--training-seed", type=int, default=20260818)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    sizes = tuple(sorted(set(args.support_sizes)))
    if args.total_rollouts < 1 or args.batch_size < 1:
        raise ValueError("total rollouts and batch size must be positive")
    if args.total_rollouts % args.batch_size:
        raise ValueError("total rollouts must be divisible by batch size")
    if sizes[-1] > args.total_rollouts:
        raise ValueError("largest support cannot exceed total rollouts")

    manifest_path = output_dir / "manifest.json"
    marker_path = output_dir / "DIVERSITY_SCHEDULES_VALIDATED.json"
    if (manifest_path.exists() or marker_path.exists()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite frozen diversity schedules")
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir = output_dir / "schedules" / f"seed_{args.training_seed}"
    support_dir = output_dir / "supports" / f"seed_{args.selection_seed}"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    from opd_code.data import atomic_write_parquet

    population = pq.read_table(source_path).to_pylist()
    supports, strata = construct_nested_supports(
        population,
        support_sizes=sizes,
        seed=args.selection_seed,
    )
    balance_report = build_balance_report(
        population,
        supports,
        seed=args.selection_seed,
        bins=STRATIFICATION_BINS,
    )
    balance_path = output_dir / "balance_report.json"
    atomic_json(balance_path, balance_report)

    membership_rows = []
    previous_ids: set[str] = set()
    support_outputs: dict[str, Any] = {}
    schedule_outputs: dict[str, Any] = {}
    for size in sizes:
        support = supports[size]
        support_ids = {problem_id(row) for row in support}
        newly_added = support_ids - previous_ids
        for row in support:
            identifier = problem_id(row)
            if identifier in newly_added:
                key = strata[identifier]
                membership_rows.append(
                    {
                        "problem_id": identifier,
                        "sample_id": str(row["sample_id"]),
                        "minimum_support_size": size,
                        "source_subset": key[0],
                        "prompt_length_bin": key[1],
                        "offline_reward_mean_bin": key[2],
                        "selection_seed": args.selection_seed,
                    }
                )
        support_path = support_dir / f"support_{size}.parquet"
        atomic_write_parquet(list(support), support_path)
        support_outputs[str(size)] = {
            "path": str(support_path),
            "rows": pq.read_metadata(support_path).num_rows,
            "sha256": sha256(support_path),
        }

        schedule = materialize_schedule(
            support,
            total_rollouts=args.total_rollouts,
            batch_size=args.batch_size,
            selection_seed=args.selection_seed,
            training_seed=args.training_seed,
        )
        steps = args.total_rollouts // args.batch_size
        schedule_path = (
            schedule_dir / f"support_{size}_s{steps}_b{args.batch_size}.parquet"
        )
        atomic_write_parquet(schedule, schedule_path)
        metadata = pq.read_metadata(schedule_path)
        if metadata.num_rows != args.total_rollouts:
            raise AssertionError(
                f"{schedule_path} has {metadata.num_rows} rows after write"
            )
        schedule_outputs[str(size)] = {
            "path": str(schedule_path),
            "rows": metadata.num_rows,
            "unique_prompts": size,
            "presentations_min": args.total_rollouts // size,
            "presentations_max": (
                args.total_rollouts + size - 1
            )
            // size,
            "sha256": sha256(schedule_path),
        }
        previous_ids = support_ids
        del schedule

    membership_rows.sort(
        key=lambda row: (
            int(row["minimum_support_size"]),
            stable_int(
                args.selection_seed, "membership", str(row["problem_id"])
            ),
        )
    )
    membership_path = output_dir / "support_membership.jsonl"
    atomic_jsonl(membership_path, membership_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "selection_method": "nested_stratified_random",
        "stratification_bins": STRATIFICATION_BINS,
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "source_rows": len(population),
        "selection_seed": args.selection_seed,
        "training_seed": args.training_seed,
        "support_sizes": list(sizes),
        "nested": True,
        "total_rollouts_per_arm": args.total_rollouts,
        "batch_size": args.batch_size,
        "optimizer_steps": args.total_rollouts // args.batch_size,
        "balance_report": str(balance_path),
        "balance_report_sha256": sha256(balance_path),
        "support_membership": str(membership_path),
        "support_membership_sha256": sha256(membership_path),
        "supports": support_outputs,
        "schedules": schedule_outputs,
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
