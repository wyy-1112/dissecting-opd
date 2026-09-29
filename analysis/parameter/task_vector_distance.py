#!/usr/bin/env python3
"""Distances between task vectors from the output of the two parameter tools.

Both hf_task_vector_cosine.py (keys ``norms``/``cosines``) and
fsdp_task_vector_cosine.py (keys ``exact_norms``/``exact_cosines``)
report, for task vectors a = theta_A - theta_0 and b = theta_B - theta_0,
||a||, ||b|| and cos(a, b). This script adds

    distance(a, b)          = ||a - b|| = sqrt(||a||^2 + ||b||^2 - 2 ||a|| ||b|| cos(a, b))
    relative_distance(a, b) = ||a - b|| / ||b||      (b = --reference, if given)
    projection(a on b)      = <a, b> / ||b||^2       (scalar fit a ~ alpha * b)

Pair keys have the form ``LEFT_vs_RIGHT``.
"""
from __future__ import annotations

import argparse
import json
import math


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("geometry_json")
    parser.add_argument("--reference", default=None, help="label used as denominator for relative distance")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = json.load(open(args.geometry_json))
    norms = data.get("exact_norms") or data["norms"]
    cosines = data.get("exact_cosines") or data["cosines"]
    rows = []
    for key, cosine in cosines.items():
        left, right = key.split("_vs_")
        a, b = norms[left], norms[right]
        dot = cosine * a * b
        distance = math.sqrt(max(a * a + b * b - 2.0 * dot, 0.0))
        row = {"pair": key, "cosine": cosine, "norm_left": a, "norm_right": b, "distance": distance}
        if args.reference in (left, right):
            ref, other = (b, a) if args.reference == right else (a, b)
            row["relative_distance"] = distance / ref
            row["projection_on_reference"] = dot / (ref * ref)
        rows.append(row)
    text = json.dumps({"source": args.geometry_json, "reference": args.reference, "pairs": rows}, indent=2)
    if args.output:
        open(args.output, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
