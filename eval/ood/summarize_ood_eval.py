"""Collect the per-benchmark OOD summaries into one table.

Usage:
  python3 scripts/eval/summarize_ood_eval.py --result-root results/ood_eval \
      --out results/ood_eval/summary.json --csv results/ood_eval/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ood_common import BENCHMARKS

# Reading order: untrained student, the RL teacher, then the distilled students.
LABEL_ORDER = ["qwen3_4b_base", "grpo_step500", "opd_step25", "opd_step50"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--csv", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.result_root)

    collected: dict[str, dict[str, dict]] = {}
    for summary_path in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        collected.setdefault(summary["label"], {})[summary["benchmark"]] = summary

    labels = [lb for lb in LABEL_ORDER if lb in collected]
    labels += sorted(lb for lb in collected if lb not in LABEL_ORDER)

    out = {"labels": labels, "benchmarks": list(BENCHMARKS), "results": collected}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    rows = []
    for label in labels:
        row = {"label": label}
        for bench in BENCHMARKS:
            summary = collected[label].get(bench)
            if summary is None:
                continue
            n = summary["n"]
            row[f"{bench}_mean@{n}"] = summary[f"mean@{n}"]
            row[f"{bench}_hit@{n}"] = summary[f"hit@{n}"]
            row[f"{bench}_avg_tokens"] = summary["avg_output_tokens"]
        rows.append(row)

    if args.csv and rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with Path(args.csv).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    width = max((len(lb) for lb in labels), default=5)
    for row in rows:
        parts = [f"{row['label']:<{width}}"]
        for key, value in row.items():
            if "mean@" not in key:
                continue
            parts.append(f"{key}={value:.4f}")
        print("  ".join(parts))
    print(f"[summary] {args.out}")


if __name__ == "__main__":
    main()
