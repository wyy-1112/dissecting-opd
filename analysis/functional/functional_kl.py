#!/usr/bin/env python3
"""Turn the frozen-top-K probe scores into a realized cross-prompt influence matrix.

The Fisher geometry report answers a pooled question: over one big concatenation
of probe positions, how aligned is an OPD endpoint's functional change with the
RL endpoint's.  A high cosine there is compatible with two very different
stories, because the pooled vector does not say *where* the agreement lives.  It
could be that training on one prompt moved the model toward the teacher on many
unrelated prompts, or that it moved a great deal on a handful of positions that
dominate the norm and did nothing on the rest.  Only a per-prompt decomposition
separates those, and the paper's claim is the first one.

So this reads the same artifacts one column at a time and reports, for every
probe independently, how much of the base-to-teacher gap the endpoint actually
closed.  With ``L_j(theta)`` the teacher-student divergence on probe ``j``'s
frozen contexts,

    I_ij = L_j(base) - L_j(theta_i),

which is the finite-update counterpart of the local ``H_ij = -g_j^T u_i`` the
cross-influence analysis reported: same matrix shape, same reading, but measured
after the update instead of predicted from a gradient at the base.  ``R_ij =
I_ij / L_j(base)`` is the same quantity as a fraction of the available gap,
which is what makes columns comparable when their absolute gaps differ by an
order of magnitude.

Two caveats are structural rather than incidental, and both are recorded in the
output.  The frozen support is the *base* model's top-K, so every divergence
here is restricted to the tokens the base considered plausible; the retained
probability mass of every model is reported per domain so the restriction can be
judged.  And the histories were sampled from the base model, so these are the
states the base visits, not the ones the trained endpoints visit -- a model that
improves by moving to friendlier states rather than by fitting the teacher where
it used to be would not show up here as an improvement.

Breadth and depth are reported separately because the DeepSeek prediction turns
on exactly that difference.  An arm with better teacher exposure should close
the gap on *more* columns; an arm that is merely better optimized closes more of
the gap on the same columns.  Reporting only a mean cannot tell those apart, so
the column-level hit counts and the concentration of the total improvement are
reported alongside it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from probe_scores import (  # noqa: E402
    canonical_sha256,
    load_fisher_reference,
    load_fisher_score,
    parse_labeled_path,
    sha256_file,
    write_json_atomic,
)

REPORT_SCHEMA = "opd_confirmation_influence_matrix_v1"
HIT_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)


def restricted_log_distribution(logprobs: np.ndarray) -> np.ndarray:
    """Renormalize full-vocabulary log-probabilities onto the frozen support.

    Each row of ``logprobs`` is already normalized over the whole vocabulary, so
    subtracting the row's log-sum-exp over the frozen columns is exactly the
    conditional distribution given that the token is in the support.  Comparing
    endpoints in that conditional space is the only comparison the artifacts can
    support without re-running the models, and it removes the part of the
    difference that is just mass leaving the support.
    """
    peak = logprobs.max(axis=1, keepdims=True)
    log_support_mass = peak + np.log(np.exp(logprobs - peak).sum(axis=1, keepdims=True))
    return logprobs - log_support_mass


def support_mass(logprobs: np.ndarray) -> np.ndarray:
    """Probability each model places inside the frozen support, per position."""
    return np.exp(logprobs.astype(np.float64)).sum(axis=1)


def forward_kl(teacher_log_q: np.ndarray, model_log_q: np.ndarray) -> np.ndarray:
    """KL(teacher || model) per position, both restricted to the frozen support.

    The teacher-first direction is the one distillation actually minimizes, so a
    drop in it is the same quantity the training objective was pushing on.
    """
    teacher_q = np.exp(teacher_log_q)
    return (teacher_q * (teacher_log_q - model_log_q)).sum(axis=1)


def centered_fisher_vector(
    base_logprobs: np.ndarray,
    endpoint_logprobs: np.ndarray,
) -> np.ndarray:
    """The report's own Fisher representation, recomputed for one probe block.

    Duplicated from ``fisher_centered_vector`` rather than imported so that the
    per-probe slicing stays local; the arithmetic is identical, including the
    per-context centering that makes the vector invariant to a constant logit
    shift.
    """
    probabilities = np.exp(base_logprobs)
    retained = probabilities.sum(axis=1)
    delta = endpoint_logprobs - base_logprobs
    weights = probabilities / retained[:, None]
    center = (weights * delta).sum(axis=1)
    return np.sqrt(probabilities) * (delta - center[:, None])


def scalar_fit(vector: np.ndarray, teacher_vector: np.ndarray) -> dict[str, float | None]:
    """Direction, best scalar projection, and what the projection leaves over.

    ``alpha`` is how much of the teacher's functional change the endpoint
    realizes along the teacher's own direction, and ``residual_over_teacher`` is
    the part of the endpoint's change that no rescaling of the teacher can
    explain, in units of the teacher's norm.  Reporting both stops a large
    cosine from being read as a large realized change, and stops a large change
    from being read as the teacher's change.
    """
    left = vector.reshape(-1)
    right = teacher_vector.reshape(-1)
    left_norm_squared = float(np.dot(left, left))
    right_norm_squared = float(np.dot(right, right))
    dot = float(np.dot(left, right))
    if left_norm_squared <= 0 or right_norm_squared <= 0:
        return {
            "cosine": None,
            "alpha": None,
            "residual_over_teacher": None,
            "norm_over_teacher": None,
        }
    alpha = dot / right_norm_squared
    residual_squared = max(left_norm_squared - alpha * dot, 0.0)
    return {
        "cosine": dot / math.sqrt(left_norm_squared * right_norm_squared),
        "alpha": alpha,
        "residual_over_teacher": math.sqrt(residual_squared / right_norm_squared),
        "norm_over_teacher": math.sqrt(left_norm_squared / right_norm_squared),
    }


def concentration(values: np.ndarray) -> dict[str, float | None]:
    """How unevenly a row's total improvement is spread over its columns.

    ``top1_share`` and ``top_quartile_share`` are the fractions of the total
    carried by the single largest and the largest quarter of columns; a broad
    effect keeps both near their uniform values, a concentrated one does not.
    Only positive entries contribute, because a row whose gains and losses
    cancel has no total to apportion.
    """
    positive = np.clip(values, 0.0, None)
    total = float(positive.sum())
    if total <= 0:
        return {"top1_share": None, "top_quartile_share": None}
    ordered = np.sort(positive)[::-1]
    quartile = max(1, len(ordered) // 4)
    return {
        "top1_share": float(ordered[0] / total),
        "top_quartile_share": float(ordered[:quartile].sum() / total),
    }


def summarize_row(
    gap: np.ndarray,
    influence: np.ndarray,
    tokens: np.ndarray,
) -> dict[str, Any]:
    """Breadth and depth of one endpoint's realized improvement over columns."""
    fraction = influence / gap
    return {
        "columns": int(len(influence)),
        "token_weighted_gap_closed": float(influence @ tokens / (gap @ tokens)),
        "mean_fraction_closed": float(fraction.mean()),
        "median_fraction_closed": float(np.median(fraction)),
        "columns_improved": int(np.sum(influence > 0)),
        "columns_worsened": int(np.sum(influence < 0)),
        "breadth_at_threshold": {
            f"{threshold:.2f}": int(np.sum(fraction > threshold))
            for threshold in HIT_THRESHOLDS
        },
        "worst_column_fraction_closed": float(fraction.min()),
        "best_column_fraction_closed": float(fraction.max()),
        "concentration": concentration(influence * tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="LABEL=NPZ",
        help="endpoint scores produced by probe_scores fisher-score",
    )
    parser.add_argument(
        "--teacher-label",
        required=True,
        help="which endpoint is this pair's teacher; it defines L_j and must be scored",
    )
    parser.add_argument(
        "--anchor-base-score",
        metavar="LABEL=NPZ",
        help="measure every change against this scoring of the base instead of the "
        "reference's own base log-probabilities; needed when the arms were scored at a "
        "higher precision than the pass that froze the token IDs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--matrix-output",
        type=Path,
        help="per-column arrays; defaults to the report path with an .npz suffix",
    )
    args = parser.parse_args()

    reference_manifest, reference = load_fisher_reference(args.reference)
    entries = [parse_labeled_path(value) for value in args.score]
    if len({label for label, _ in entries}) != len(entries):
        raise ValueError("score labels must be unique")
    if args.teacher_label not in {label for label, _ in entries}:
        raise ValueError(f"teacher label {args.teacher_label!r} was not scored")

    manifests: dict[str, dict[str, Any]] = {}
    logprobs: dict[str, np.ndarray] = {}
    for label, path in entries:
        manifests[label], logprobs[label] = load_fisher_score(
            label,
            path,
            reference_path=args.reference,
            reference=reference,
        )

    base_logprobs = reference["base_logprobs"]
    anchor: dict[str, Any] | None = None
    if args.anchor_base_score is not None:
        anchor_label, anchor_path = parse_labeled_path(args.anchor_base_score)
        anchor_manifest, base_logprobs = load_fisher_score(
            anchor_label,
            anchor_path,
            reference_path=args.reference,
            reference=reference,
        )
        anchor = {
            "label": anchor_label,
            "path": str(anchor_path),
            "sha256": sha256_file(anchor_path),
            "checkpoint": anchor_manifest.get("checkpoint"),
        }
    offsets = reference["position_offsets"]
    probe_ids = reference["probe_ids"]
    domains = reference["domains"]
    endpoints = [label for label, _ in entries if label != args.teacher_label]
    if not endpoints:
        raise ValueError("at least one non-teacher endpoint is required")

    columns = len(probe_ids)
    tokens = np.diff(offsets).astype(np.float64)
    base_gap = np.zeros(columns, dtype=np.float64)
    divergence = {label: np.zeros(columns, dtype=np.float64) for label in endpoints}
    fit: dict[str, list[dict[str, float | None]]] = {label: [] for label in endpoints}
    mass = {label: np.zeros(columns, dtype=np.float64) for label in ("base", *logprobs)}

    for index in range(columns):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        base_block = base_logprobs[start:stop]
        teacher_block = logprobs[args.teacher_label][start:stop]
        base_log_q = restricted_log_distribution(base_block)
        teacher_log_q = restricted_log_distribution(teacher_block)
        base_gap[index] = forward_kl(teacher_log_q, base_log_q).mean()
        mass["base"][index] = support_mass(base_block).mean()
        teacher_vector = centered_fisher_vector(base_block, teacher_block)
        for label, values in logprobs.items():
            block = values[start:stop]
            mass[label][index] = support_mass(block).mean()
            if label == args.teacher_label:
                continue
            divergence[label][index] = forward_kl(
                teacher_log_q,
                restricted_log_distribution(block),
            ).mean()
            fit[label].append(
                scalar_fit(centered_fisher_vector(base_block, block), teacher_vector)
            )
        print(f"[column] {index + 1}/{columns} {probe_ids[index]}", flush=True)

    if not np.all(base_gap > 0):
        raise ValueError(
            "a column has no base-to-teacher gap, so its fraction closed is "
            "undefined; the teacher label is probably wrong"
        )
    influence = {label: base_gap - divergence[label] for label in endpoints}

    domain_columns = {
        "all": np.arange(columns),
        **{
            str(domain): np.flatnonzero(domains == domain)
            for domain in dict.fromkeys(domains.tolist())
        },
    }
    rows: dict[str, dict[str, Any]] = {}
    for label in endpoints:
        pooled = {}
        for domain, selected in domain_columns.items():
            pooled[domain] = summarize_row(
                base_gap[selected],
                influence[label][selected],
                tokens[selected],
            )
            pooled[domain]["scalar_fit_column_median"] = {
                key: (
                    None
                    if all(fit[label][index][key] is None for index in selected)
                    else float(
                        np.median(
                            [
                                fit[label][index][key]
                                for index in selected
                                if fit[label][index][key] is not None
                            ]
                        )
                    )
                )
                for key in ("cosine", "alpha", "residual_over_teacher", "norm_over_teacher")
            }
        rows[label] = pooled

    matrix_output = args.matrix_output or args.output.with_suffix(".npz")
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = matrix_output.with_suffix(matrix_output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probe_ids=probe_ids,
            domains=domains,
            tokens=tokens,
            base_gap=base_gap,
            **{f"influence__{label}": influence[label] for label in endpoints},
            **{f"divergence__{label}": divergence[label] for label in endpoints},
            **{f"support_mass__{label}": values for label, values in mass.items()},
            **{
                f"{key}__{label}": np.array(
                    [
                        np.nan if entry[key] is None else entry[key]
                        for entry in fit[label]
                    ],
                    dtype=np.float64,
                )
                for label in endpoints
                for key in ("cosine", "alpha", "residual_over_teacher", "norm_over_teacher")
            },
        )
    temporary.replace(matrix_output)

    report = {
        "schema_version": REPORT_SCHEMA,
        "definition": {
            "column": "one frozen probe trajectory, scored identically under every model",
            "divergence": (
                "L_j(theta) = mean over probe j's positions of "
                "KL(teacher || theta), both renormalized onto the base model's "
                "frozen top-K support"
            ),
            "influence": "I_ij = L_j(base) - L_j(theta_i); positive closes the gap",
            "fraction_closed": "R_ij = I_ij / L_j(base)",
            "scalar_fit": (
                "cosine, alpha = <v_i,v_T>/<v_T,v_T>, and the residual after "
                "removing alpha*v_T, on the report's centered Fisher vectors"
            ),
        },
        "limitations": {
            "frozen_support": (
                "Every divergence is conditional on the base model's top-K; "
                "per-domain retained mass is reported and must be inspected."
            ),
            "history_origin": (
                "Histories were sampled by the base model, so improvement "
                "here means fitting the teacher on states the base visits, not "
                "on states the endpoint visits."
            ),
        },
        "teacher_label": args.teacher_label,
        "anchor_base_score": anchor,
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
            "base_checkpoint": reference_manifest["base_checkpoint"],
            "cohort": reference_manifest["cohort"],
            "top_k": reference_manifest["scoring"]["top_k"],
        },
        "models": {label: manifest["checkpoint"] for label, manifest in manifests.items()},
        "columns": {
            "total": columns,
            "by_domain": {
                domain: int(len(selected)) for domain, selected in domain_columns.items()
            },
            "tokens_by_domain": {
                domain: float(tokens[selected].sum())
                for domain, selected in domain_columns.items()
            },
        },
        "base_gap_by_domain": {
            domain: {
                "token_weighted_mean": float(
                    base_gap[selected] @ tokens[selected] / tokens[selected].sum()
                ),
                "median": float(np.median(base_gap[selected])),
            }
            for domain, selected in domain_columns.items()
        },
        "retained_support_mass_by_domain": {
            domain: {
                label: float(values[selected].mean())
                for label, values in mass.items()
            }
            for domain, selected in domain_columns.items()
        },
        "rows": rows,
        "matrix_artifact": {
            "path": str(matrix_output.resolve()),
            "sha256": sha256_file(matrix_output),
        },
    }
    report["report_payload_sha256"] = canonical_sha256(report)
    write_json_atomic(args.output, report)
    print(json.dumps({"output": str(args.output), "matrix": str(matrix_output)}, sort_keys=True))


if __name__ == "__main__":
    main()
