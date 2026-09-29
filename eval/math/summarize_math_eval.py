#!/usr/bin/env python3
"""Summarize G-OPD math JSONL generations with the official scored fields."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


BENCHMARKS = ("aime24", "aime25", "hmmt25_feb", "hmmt25_nov")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    per_benchmark: dict[str, dict[str, object]] = {}
    for name in args.benchmarks:
        path = eval_dir / f"{name}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"empty benchmark output: {path}")
        sample_counts = {len(row["acc_list"]) for row in rows}
        if len(sample_counts) != 1 or next(iter(sample_counts)) < 1:
            raise ValueError(f"inconsistent sample count: {path}")

        accuracies = [
            bool(value)
            for row in rows
            for value in row["acc_list"]
        ]
        predictions = [
            value
            for row in rows
            for value in row["pred_answers"]
        ]
        responses = [
            str(value)
            for row in rows
            for value in row["responses"]
        ]
        if not (len(accuracies) == len(predictions) == len(responses)):
            raise ValueError(f"misaligned scored fields: {path}")
        think_closed_rate = statistics.fmean(
            "</think>" in response for response in responses
        )
        per_benchmark[name] = {
            "mean_at_k": statistics.fmean(accuracies),
            "pass_at_k": statistics.fmean(
                any(bool(value) for value in row["acc_list"])
                for row in rows
            ),
            "problems": len(rows),
            "samples": len(accuracies),
            "missing_boxed_rate": statistics.fmean(
                prediction is None for prediction in predictions
            ),
            "think_closed_rate": think_closed_rate,
            "unfinished_think_rate": 1.0 - think_closed_rate,
        }

    payload = {
        "average_mean_at_k": statistics.fmean(
            float(row["mean_at_k"]) for row in per_benchmark.values()
        ),
        "benchmarks": per_benchmark,
        "eval_dir": str(eval_dir),
        "label": args.label,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
