#!/usr/bin/env python3
"""Build a nested source/length/difficulty-stratified random ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "opd_stratified_selection_ranking_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_hash(seed: int, *parts: object) -> str:
    text = json.dumps((seed, *parts), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_id(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping) or "index" not in extra:
        raise ValueError("every source row must have extra_info.index")
    return str(extra["index"])


def prompt_characters(row: Mapping[str, Any]) -> int:
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError(f"prompt is not a message list: {prompt_id(row)}")
    return sum(
        len(str(message.get("content", "")))
        for message in prompt
        if isinstance(message, Mapping)
    )


def difficulty_bin(pass_rate: float) -> str:
    if pass_rate == 0:
        return "zero"
    if pass_rate < 0.5:
        return "low"
    if pass_rate < 1:
        return "medium"
    return "perfect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cohort-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--length-strata", type=int, default=4)
    args = parser.parse_args()

    source = args.source.resolve()
    cohort = args.cohort_jsonl.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen ranking: {output}")
    if args.rollouts_per_prompt < 1 or args.length_strata < 1:
        raise ValueError("rollout and stratum counts must be positive")

    import pyarrow.parquet as pq

    rows = pq.read_table(source).to_pylist()
    by_id = {prompt_id(row): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("source contains duplicate prompt IDs")

    pass_rates: dict[str, float] = {}
    with cohort.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identifier = str(row["problem_id"])
            samples = row["samples"][: args.rollouts_per_prompt]
            if len(samples) != args.rollouts_per_prompt:
                raise ValueError(f"insufficient frozen rollouts for {identifier}")
            pass_rates[identifier] = sum(
                float(sample["reward"]) for sample in samples
            ) / len(samples)
    if set(pass_rates) != set(by_id):
        raise ValueError("source and frozen-rollout candidate sets differ")

    length_order = sorted(
        by_id,
        key=lambda identifier: (
            prompt_characters(by_id[identifier]),
            seeded_hash(args.selection_seed, "length-tie", identifier),
        ),
    )
    length_bin = {
        identifier: min(
            args.length_strata - 1,
            rank * args.length_strata // len(length_order),
        )
        for rank, identifier in enumerate(length_order)
    }

    strata: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for identifier, row in by_id.items():
        key = (
            str(row.get("data_source", "missing")),
            length_bin[identifier],
            difficulty_bin(pass_rates[identifier]),
        )
        strata[key].append(identifier)
        metadata[identifier] = {
            "prompt_id": identifier,
            "source_stratum": key[0],
            "length_stratum": key[1],
            "difficulty_stratum": key[2],
            "prompt_characters": prompt_characters(row),
            "student_pass_rate_k": pass_rates[identifier],
        }
    for key, identifiers in strata.items():
        identifiers.sort(
            key=lambda identifier: seeded_hash(
                args.selection_seed, "within-stratum", key, identifier
            )
        )

    total = len(rows)
    selected_counts: Counter[tuple[str, int, str]] = Counter()
    positions: Counter[tuple[str, int, str]] = Counter()
    ordered: list[str] = []
    stratum_keys = sorted(strata)
    while len(ordered) < total:
        position = len(ordered) + 1
        eligible = [
            key
            for key in stratum_keys
            if positions[key] < len(strata[key])
        ]
        key = max(
            eligible,
            key=lambda candidate: (
                position * len(strata[candidate]) / total
                - selected_counts[candidate],
                seeded_hash(args.selection_seed, "stratum-tie", position, candidate),
            ),
        )
        identifier = strata[key][positions[key]]
        positions[key] += 1
        selected_counts[key] += 1
        ordered.append(identifier)

    payload = {
        "schema_version": SCHEMA,
        "status": "frozen_before_training",
        "selection_method": "proportional_source_length_difficulty_stratified_random",
        "selection_seed": args.selection_seed,
        "ordered_prompt_ids": ordered,
        "ranking_rows": [metadata[identifier] for identifier in ordered],
        "stratification": {
            "length_strata": args.length_strata,
            "difficulty_bins": ["zero", "low", "medium", "perfect"],
            "difficulty_rollouts_per_prompt": args.rollouts_per_prompt,
            "allocation": "weighted deficit interleaving proportional to joint-stratum population",
            "within_stratum_order": "seeded SHA256 permutation",
            "joint_stratum_counts": {
                json.dumps(key): len(strata[key]) for key in stratum_keys
            },
        },
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "rows": total,
        },
        "frozen_rollouts": {
            "path": str(cohort),
            "sha256": sha256(cohort),
            "rows": len(pass_rates),
        },
        "selection_cost": {
            "marginal_gpu_hours": 0,
            "note": "reuses frozen initial-student rollouts; report upstream K=8 generation cost separately",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "selection_method": payload["selection_method"],
                "candidates": total,
                "joint_strata": len(strata),
                "first_8": ordered[:8],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
