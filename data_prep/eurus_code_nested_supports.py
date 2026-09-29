#!/usr/bin/env python3
"""Materialize nested M=1/48/3840/full schedules from the Code RL data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Callable, Hashable, Mapping, Sequence

# Directory laid out like the original project (data/opd, data/rl, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[1] / "outputs" / "opd_project_root"))
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_SOURCE = ROOT / "data/rl/eurus_code_grpo/train.parquet"
DEFAULT_OUTPUT = ROOT / "data/opd/qwen3_4b_code_diversity_b256_s100_v1"
SCHEMA_VERSION = "qwen3_4b_code_diversity_v1"
DEFAULT_SUPPORT_SIZES = (1, 48, 3840, 22618)
LENGTH_BINS = 6


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


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prompt_index(row: Mapping[str, Any]) -> int:
    return int(row["extra_info"]["index"])


def prompt_characters(row: Mapping[str, Any]) -> int:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"row {prompt_index(row)} has no chat prompt")
    return sum(
        len(str(message.get("content", "")))
        for message in prompt
        if isinstance(message, Mapping)
    )


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
            stable_int(seed, "rank-bin", tag, prompt_index(row)),
        ),
    )
    return {
        prompt_index(row): min(bins - 1, rank * bins // len(ordered))
        for rank, row in enumerate(ordered)
    }


def construct_nested_supports(
    rows: Sequence[Mapping[str, Any]],
    *,
    support_sizes: Sequence[int],
    seed: int,
) -> tuple[dict[int, list[Mapping[str, Any]]], dict[int, tuple[str, int]]]:
    sizes = tuple(sorted(set(support_sizes)))
    if not sizes or sizes[-1] != len(rows):
        raise ValueError("largest support must equal the source population")
    indices = [prompt_index(row) for row in rows]
    if len(set(indices)) != len(rows):
        raise ValueError("source contains duplicate extra_info.index values")

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["data_source"])].append(row)
    length_bins: dict[int, int] = {}
    for source, source_rows in by_source.items():
        length_bins.update(
            rank_bins(
                source_rows,
                value=lambda row: float(prompt_characters(row)),
                bins=LENGTH_BINS,
                seed=seed,
                tag=source,
            )
        )

    strata: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    row_strata: dict[int, tuple[str, int]] = {}
    for row in rows:
        index = prompt_index(row)
        key = (str(row["data_source"]), length_bins[index])
        strata[key].append(row)
        row_strata[index] = key
    for key, members in strata.items():
        members.sort(
            key=lambda row: stable_int(
                seed,
                "within-stratum",
                key,
                prompt_index(row),
            )
        )

    selected_counts: Counter[tuple[str, int]] = Counter()
    offsets: Counter[tuple[str, int]] = Counter()
    ordered_population: list[Mapping[str, Any]] = []
    population_size = len(rows)
    for prefix_size in range(1, population_size + 1):
        available = [
            key for key, members in strata.items() if offsets[key] < len(members)
        ]
        selected_key = max(
            available,
            key=lambda key: (
                prefix_size * len(strata[key]) / population_size
                - selected_counts[key],
                -stable_int(seed, "stratum-deficit-tie", prefix_size, key),
            ),
        )
        ordered_population.append(strata[selected_key][offsets[selected_key]])
        offsets[selected_key] += 1
        selected_counts[selected_key] += 1

    supports = {size: ordered_population[:size] for size in sizes}
    for smaller, larger in zip(sizes, sizes[1:]):
        small = {prompt_index(row) for row in supports[smaller]}
        large = {prompt_index(row) for row in supports[larger]}
        if not small <= large:
            raise AssertionError(f"M={smaller} is not nested in M={larger}")
    return supports, row_strata


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
    stratum: tuple[str, int],
) -> dict[str, Any]:
    result = dict(row)
    extra = dict(row["extra_info"])
    extra["code_prompt_diversity"] = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": "nested_source_length_stratified_random",
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "support_size": support_size,
        "total_rollouts": total_rollouts,
        "presentation_index": presentation_index,
        "schedule_index": schedule_index,
        "optimizer_step": schedule_index // batch_size,
        "position_in_step": schedule_index % batch_size,
        "source": stratum[0],
        "prompt_character_length_bin": stratum[1],
    }
    result["extra_info"] = extra
    return result


def materialize_schedule(
    support: Sequence[Mapping[str, Any]],
    row_strata: Mapping[int, tuple[str, int]],
    *,
    total_rollouts: int,
    batch_size: int,
    selection_seed: int,
    training_seed: int,
) -> tuple[list[dict[str, Any]], Counter[int]]:
    support_size = len(support)
    base_presentations, remainder = divmod(total_rollouts, support_size)
    extras = {
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
        prompt_index(row): base_presentations + (prompt_index(row) in extras)
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
                    stratum=row_strata[index],
                )
            )
            counts[index] += 1
    if len(schedule) != total_rollouts:
        raise AssertionError("schedule has the wrong row count")
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("prompt presentation counts differ by more than one")
    return schedule, counts


def categorical_balance(
    population: Sequence[Mapping[str, Any]],
    support: Sequence[Mapping[str, Any]],
    getter: Callable[[Mapping[str, Any]], Hashable],
) -> dict[str, object]:
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
        "max_absolute_proportion_difference": max(
            abs(value) for value in differences.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--support-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SUPPORT_SIZES,
    )
    parser.add_argument("--total-rollouts", type=int, default=25_600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sizes = tuple(sorted(set(args.support_sizes)))
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.total_rollouts % args.batch_size:
        raise ValueError("total rollouts must be divisible by batch size")

    manifest_path = output_dir / "manifest.json"
    marker_path = output_dir / "DIVERSITY_SCHEDULES_VALIDATED.json"
    if (manifest_path.exists() or marker_path.exists()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite frozen schedules")
    support_dir = output_dir / "supports" / f"seed_{args.selection_seed}"
    schedule_dir = output_dir / "schedules" / f"seed_{args.training_seed}"
    support_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    from opd_code.data import atomic_write_parquet

    population = pq.read_table(source).to_pylist()
    if sizes[-1] != len(population):
        raise ValueError(
            f"largest support {sizes[-1]} must equal source rows {len(population)}"
        )
    supports, row_strata = construct_nested_supports(
        population,
        support_sizes=sizes,
        seed=args.selection_seed,
    )

    support_outputs: dict[str, object] = {}
    schedule_outputs: dict[str, object] = {}
    membership: list[dict[str, object]] = []
    previous: set[int] = set()
    balance: dict[str, object] = {}
    for size in sizes:
        support = supports[size]
        indices = {prompt_index(row) for row in support}
        for row in support:
            index = prompt_index(row)
            if index not in previous:
                source_name, length_bin = row_strata[index]
                membership.append(
                    {
                        "index": index,
                        "minimum_support_size": size,
                        "data_source": source_name,
                        "prompt_character_length_bin": length_bin,
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
        schedule, counts = materialize_schedule(
            support,
            row_strata,
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
        schedule_outputs[str(size)] = {
            "path": str(schedule_path),
            "rows": pq.read_metadata(schedule_path).num_rows,
            "unique_prompts": size,
            "presentations_min": min(counts.values()),
            "presentations_max": max(counts.values()),
            "sha256": sha256(schedule_path),
        }
        balance[str(size)] = {
            "data_source": categorical_balance(
                population,
                support,
                lambda row: str(row["data_source"]),
            )
        }
        previous = indices

    membership.sort(
        key=lambda row: (
            int(row["minimum_support_size"]),
            stable_int(args.selection_seed, "membership", int(row["index"])),
        )
    )
    membership_path = output_dir / "support_membership.jsonl"
    atomic_jsonl(membership_path, membership)
    balance_path = output_dir / "balance_report.json"
    atomic_json(
        balance_path,
        {
            "schema_version": SCHEMA_VERSION,
            "selection_seed": args.selection_seed,
            "length_bins": LENGTH_BINS,
            "arms": balance,
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "selection_method": "nested_source_length_stratified_random",
        "selection_seed": args.selection_seed,
        "training_seed": args.training_seed,
        "source": str(source),
        "source_rows": len(population),
        "source_sha256": sha256(source),
        "support_sizes": list(sizes),
        "nested": True,
        "batch_size": args.batch_size,
        "optimizer_steps": args.total_rollouts // args.batch_size,
        "total_rollouts_per_arm": args.total_rollouts,
        "supports": support_outputs,
        "schedules": schedule_outputs,
        "support_membership": str(membership_path),
        "support_membership_sha256": sha256(membership_path),
        "balance_report": str(balance_path),
        "balance_report_sha256": sha256(balance_path),
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
