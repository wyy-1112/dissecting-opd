#!/usr/bin/env python3
"""Relate support-set mean-gradient representativeness to endpoint quality."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "opd_update_representativeness_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_id(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, str):
        extra = json.loads(extra)
    if isinstance(extra, Mapping) and "index" in extra:
        return str(extra["index"])
    for key in ("problem_id", "prompt_id", "index"):
        if key in row:
            return str(row[key])
    raise ValueError("row has no stable prompt identifier")


def support_ids(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        identifier = prompt_id(row)
        if identifier not in seen:
            ordered.append(identifier)
            seen.add(identifier)
    return ordered


def pooled_error(
    gradients: Any,
    lengths: Any,
    indices: Sequence[int],
) -> dict[str, Any]:
    """Compute E(S) in each CountSketch and in their concatenated geometry."""
    import torch

    gradients = gradients.double()
    lengths = lengths.double()
    chosen = torch.as_tensor(indices, dtype=torch.long)
    pool = (gradients * lengths[None, :, None]).sum(dim=1) / lengths.sum()
    support_lengths = lengths[chosen]
    support = (
        gradients[:, chosen, :] * support_lengths[None, :, None]
    ).sum(dim=1) / support_lengths.sum()
    residual = support - pool
    numerator_sq = residual.square().sum(dim=1)
    denominator_sq = pool.square().sum(dim=1)
    support_sq = support.square().sum(dim=1)
    support_pool_dot = (support * pool).sum(dim=1)
    if bool((denominator_sq <= 0).any()):
        raise ValueError("candidate-pool mean gradient has zero norm")
    per_projection = (numerator_sq / denominator_sq).sqrt()
    concatenated = (numerator_sq.sum() / denominator_sq.sum()).sqrt()
    per_projection_cosine = support_pool_dot / (
        support_sq * denominator_sq
    ).sqrt().clamp(min=1e-24)
    concatenated_cosine = support_pool_dot.sum() / (
        support_sq.sum() * denominator_sq.sum()
    ).sqrt().clamp(min=1e-24)
    return {
        "concatenated": float(concatenated),
        "per_projection": [float(value) for value in per_projection],
        "cosine_to_pool": float(concatenated_cosine),
        "cosine_to_pool_per_projection": [
            float(value) for value in per_projection_cosine
        ],
        "support_effective_tokens": int(support_lengths.sum()),
        "pool_effective_tokens": int(lengths.sum()),
        "support_token_share": float(support_lengths.sum() / lengths.sum()),
    }


def load_scores(method: Mapping[str, Any]) -> dict[str, Any] | None:
    math_path = method.get("math_summary")
    ood_dir = method.get("ood_dir")
    if not math_path or not ood_dir:
        return None
    math_path = Path(math_path)
    ood_dir = Path(ood_dir)
    paths = {
        "math": math_path,
        "gpqa_diamond": ood_dir / "gpqa_diamond" / "summary.json",
        "humaneval_plus": ood_dir / "humaneval_plus" / "summary.json",
        "livecodebench_v6": ood_dir / "livecodebench_v6" / "summary.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    math_row = json.loads(paths["math"].read_text())
    rows = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
        if name != "math"
    }
    values = {
        "math": float(math_row["average_mean_at_k"]),
        "gpqa_diamond": float(rows["gpqa_diamond"]["mean@8"]),
        "humaneval_plus": float(rows["humaneval_plus"]["mean@8"]),
        "livecodebench_v6": float(rows["livecodebench_v6"]["mean@8"]),
    }
    return {
        **values,
        "avg4": float(np.mean(list(values.values()))),
        "files": {name: str(path.resolve()) for name, path in paths.items()},
        "file_sha256": {name: sha256(path) for name, path in paths.items()},
    }


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if len(x_array) < 2 or x_array.std() == 0 or y_array.std() == 0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def exact_permutation_p(
    x: Sequence[float],
    y: Sequence[float],
    *,
    rank: bool,
) -> float | None:
    if len(x) > 9 or len(x) < 3:
        return None
    x_values = average_ranks(x) if rank else np.asarray(x, dtype=np.float64)
    y_values = average_ranks(y) if rank else np.asarray(y, dtype=np.float64)
    observed = abs(correlation(x_values, y_values))
    exceed = 0
    total = 0
    for permutation in itertools.permutations(y_values.tolist()):
        value = abs(correlation(x_values, permutation))
        exceed += int(value >= observed - 1e-12)
        total += 1
    return exceed / total


def association(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    complete = [row for row in rows if row["scores"] is not None]
    errors = [row["representativeness"][split]["concatenated"] for row in complete]
    scores = [row["scores"]["avg4"] for row in complete]
    pearson = correlation(errors, scores)
    spearman = correlation(average_ranks(errors), average_ranks(scores))
    return {
        "n": len(complete),
        "methods": [row["name"] for row in complete],
        "pearson_r": pearson,
        "pearson_exact_permutation_p_two_sided": exact_permutation_p(
            errors, scores, rank=False
        ),
        "spearman_rho": spearman,
        "spearman_exact_permutation_p_two_sided": exact_permutation_p(
            errors, scores, rank=True
        ),
    }


def draw_plot(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [row for row in rows if row["scores"] is not None]
    colors = {
        "uniform": "#5B8FF9",
        "semantic": "#61DDAA",
        "stratified": "#65789B",
        "hard": "#F6BD16",
        "shortest": "#E8684A",
        "cost_d_opt": "#6DC8EC",
    }
    markers = {
        "uniform": "o",
        "semantic": "s",
        "stratified": "D",
        "hard": "^",
        "shortest": "X",
        "cost_d_opt": "P",
    }
    figure, axis = plt.subplots(figsize=(6.6, 4.6), dpi=180)
    for row in complete:
        method = row["method"]
        x = 100 * row["representativeness"]["discovery"]["concatenated"]
        y = 100 * row["scores"]["avg4"]
        axis.scatter(
            x,
            y,
            s=64 if method != "uniform" else 52,
            marker=markers.get(method, "o"),
            color=colors.get(method, "#777777"),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            row["plot_label"],
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Update representativeness error E(S) (%) ↓")
    axis.set_ylabel("Actual Avg₄ (%) ↑")
    axis.set_title("JustRL–DeepMath: initial update representativeness vs performance")
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    import torch

    artifact_path = args.artifact.resolve()
    spec_path = args.spec.resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    spec = json.loads(spec_path.read_text())
    problem_ids = [str(value) for value in artifact["problem_ids"]]
    id_to_index = {identifier: index for index, identifier in enumerate(problem_ids)}
    if len(id_to_index) != len(problem_ids):
        raise ValueError("CountSketch artifact contains duplicate problem IDs")

    rows = []
    for method in spec["methods"]:
        path = Path(method["support_path"]).resolve()
        identifiers = support_ids(path)
        if len(identifiers) != int(method["support_size"]):
            raise ValueError(
                f"{method['name']} support size {len(identifiers)} "
                f"!= {method['support_size']}"
            )
        missing = sorted(set(identifiers) - set(id_to_index))
        if missing:
            raise ValueError(f"{method['name']} IDs outside CountSketch pool: {missing}")
        indices = [id_to_index[identifier] for identifier in identifiers]
        representativeness = {}
        for split in ("discovery", "confirmation"):
            representativeness[split] = pooled_error(
                artifact[split]["g_i"],
                artifact[split]["L_i"],
                indices,
            )
            representativeness[f"{split}_raw"] = pooled_error(
                artifact[split]["raw_g_i"],
                artifact[split]["L_i"],
                indices,
            )
        rows.append(
            {
                "name": method["name"],
                "method": method["method"],
                "plot_label": method["plot_label"],
                "selection_seed": method.get("selection_seed"),
                "support_path": str(path),
                "support_sha256": sha256(path),
                "support_ids": identifiers,
                "representativeness": representativeness,
                "scores": load_scores(method),
                "endpoint_status": method.get("endpoint_status", "complete"),
            }
        )

    payload = {
        "schema_version": SCHEMA,
        "definition": {
            "gradient": "pre-SVD, Adam-metric CountSketch g_i",
            "prompt_gradient": "g_i = sum_r T_ir g_ir / L_i",
            "support_mean": "sum_{i in S} L_i g_i / sum_{i in S} L_i",
            "pool_mean": "sum_{i in C} L_i g_i / sum_{i in C} L_i",
            "error": "||gbar_S-gbar_C||_2 / ||gbar_C||_2",
            "projection_aggregation": (
                "concatenate three independent CountSketch projections; "
                "equivalently sum squared norms before taking the ratio"
            ),
            "primary_split": "discovery",
            "confirmation_role": "stability sensitivity only",
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": sha256(artifact_path),
            "schema_version": artifact["schema_version"],
            "candidate_prompts": len(problem_ids),
            "projection_seeds": artifact["projection_seeds"],
            "sketch_dimension_per_projection": int(
                artifact["discovery"]["g_i"].shape[-1]
            ),
        },
        "spec": {"path": str(spec_path), "sha256": sha256(spec_path)},
        "rows": rows,
        "association": {
            "discovery": association(rows, "discovery"),
            "confirmation": association(rows, "confirmation"),
        },
        "caveat": (
            "E(S) is a pre-training mean-gradient diagnostic. It is not the "
            "full Adam update, optimizer state, or the nonlinear OPD trajectory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    draw_plot(rows, args.plot)
    print(json.dumps(payload["association"], indent=2))
    print(f"[report] {args.output.resolve()}")
    print(f"[plot] {args.plot.resolve()}")


if __name__ == "__main__":
    main()
