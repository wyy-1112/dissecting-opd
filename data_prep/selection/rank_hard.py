#!/usr/bin/env python3
"""Rank a frozen OPD candidate pool by initial-student rollout difficulty."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "opd_hard_selection_ranking_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_id(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping) or "index" not in extra:
        raise ValueError("every source row must have extra_info.index")
    return str(extra["index"])


def tie_break(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:hard:{identifier}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cohort-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    cohort = args.cohort_jsonl.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen ranking: {output}")
    if args.rollouts_per_prompt < 1:
        raise ValueError("rollouts-per-prompt must be positive")

    import pyarrow.parquet as pq

    source_rows = pq.read_table(source).to_pylist()
    source_ids = [prompt_id(row) for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source contains duplicate prompt IDs")

    scores: dict[str, dict[str, Any]] = {}
    with cohort.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            identifier = str(row["problem_id"])
            if identifier in scores:
                raise ValueError(f"duplicate cohort problem_id {identifier}")
            samples = row["samples"]
            if len(samples) < args.rollouts_per_prompt:
                raise ValueError(
                    f"{identifier} has {len(samples)} samples, "
                    f"needs {args.rollouts_per_prompt}"
                )
            selected = samples[: args.rollouts_per_prompt]
            rewards = [float(sample["reward"]) for sample in selected]
            if not all(math.isfinite(value) for value in rewards):
                raise ValueError(f"non-finite reward for {identifier}")
            scores[identifier] = {
                "prompt_id": identifier,
                "correct_rollouts": sum(value > 0.5 for value in rewards),
                "student_pass_rate": sum(rewards) / len(rewards),
                "rollouts": len(rewards),
                "cohort_line": line_number,
            }

    if set(scores) != set(source_ids):
        raise ValueError(
            "source/cohort ID mismatch: "
            f"missing={sorted(set(source_ids) - set(scores))[:10]} "
            f"extra={sorted(set(scores) - set(source_ids))[:10]}"
        )
    ranked = sorted(
        scores.values(),
        key=lambda row: (
            row["student_pass_rate"],
            tie_break(args.selection_seed, row["prompt_id"]),
        ),
    )
    payload = {
        "schema_version": SCHEMA,
        "status": "frozen_before_training",
        "selection_method": "hard_lowest_initial_student_pass_rate",
        "selection_seed": args.selection_seed,
        "ordered_prompt_ids": [row["prompt_id"] for row in ranked],
        "ranking_rows": ranked,
        "difficulty_definition": {
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "samples": "first K samples in the frozen cohort row",
            "score": "mean official training reward; ascending is harder",
            "tie_break": "SHA256(selection_seed, 'hard', prompt_id)",
        },
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "rows": len(source_ids),
        },
        "frozen_rollouts": {
            "path": str(cohort),
            "sha256": sha256(cohort),
            "rows": len(scores),
        },
        "selection_cost": {
            "marginal_gpu_hours": 0,
            "note": "reuses frozen initial-student rollouts; report their upstream K=8 generation cost separately",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({key: payload[key] for key in ("schema_version", "selection_method", "selection_seed")}, indent=2))


if __name__ == "__main__":
    main()
