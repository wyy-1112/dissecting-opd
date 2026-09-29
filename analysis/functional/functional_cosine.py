#!/usr/bin/env python3
"""Test whether OPD endpoints are collinear in the full function-space representation.

The confirmation-influence report reduces every probe to one scalar KL improvement,
so a rank-1 fit there only says the endpoints improve the same probes in the same
proportions.  It cannot rule out that they change *different tokens* inside each
probe.  This script does the stronger test directly: it builds the whole centered
Fisher vector for each endpoint, forms the exact Gram matrix, and asks how much of
the endpoint family a single direction explains.

Everything is derived from the Gram matrix, so the cosines, the scalar fits, and
the spectrum are mutually consistent by construction rather than by convention.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_scores import (  # noqa: E402
    canonical_sha256,
    fisher_centered_vector,
    load_fisher_reference,
    load_fisher_score,
    parse_labeled_path,
    sha256_file,
    write_json_atomic,
)

REPORT_SCHEMA = "opd_endpoint_collinearity_v1"


def position_domains(
    domains: np.ndarray,
    position_offsets: np.ndarray,
) -> np.ndarray:
    """Expand per-probe domain labels onto the positions they own.

    The Gram matrix has to be restricted by domain at position granularity,
    because a domain's share of the total is set by how many scored positions it
    contributes, not by how many probes carry its label.
    """
    counts = np.diff(position_offsets).astype(np.int64)
    return np.repeat(domains, counts)


def gram_matrix(vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Exact inner products between every pair of flattened endpoint vectors."""
    stacked = np.stack([vectors[label].reshape(-1) for label in vectors])
    return stacked @ stacked.T


def cosine_table(gram: np.ndarray, labels: list[str]) -> dict[str, dict[str, float | None]]:
    """Pairwise cosines read straight off the Gram matrix."""
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    table: dict[str, dict[str, float | None]] = {}
    for row, left in enumerate(labels):
        table[left] = {}
        for column, right in enumerate(labels):
            scale = norms[row] * norms[column]
            table[left][right] = (
                None if scale <= 0 else float(gram[row, column] / scale)
            )
    return table


def scalar_fits(
    gram: np.ndarray,
    labels: list[str],
    reference_label: str,
) -> dict[str, dict[str, float | None]]:
    """Fit ``v_i = c_i * v_ref + e_i`` for every endpoint against one reference.

    ``residual_over_self`` is the fraction of the endpoint's own change that no
    rescaling of the reference can reproduce; it is the quantity that decides
    whether "the endpoints differ only in magnitude" is true.
    """
    index = {label: position for position, label in enumerate(labels)}
    pivot = index[reference_label]
    reference_norm_squared = float(gram[pivot, pivot])
    fits: dict[str, dict[str, float | None]] = {}
    for label in labels:
        row = index[label]
        self_norm_squared = float(gram[row, row])
        dot = float(gram[row, pivot])
        if reference_norm_squared <= 0 or self_norm_squared <= 0:
            fits[label] = {
                "scale": None,
                "cosine": None,
                "residual_over_self": None,
                "residual_over_reference": None,
                "norm_over_reference": None,
            }
            continue
        scale = dot / reference_norm_squared
        residual_squared = max(self_norm_squared - scale * dot, 0.0)
        fits[label] = {
            "scale": scale,
            "cosine": dot / math.sqrt(self_norm_squared * reference_norm_squared),
            "residual_over_self": math.sqrt(residual_squared / self_norm_squared),
            "residual_over_reference": math.sqrt(
                residual_squared / reference_norm_squared
            ),
            "norm_over_reference": math.sqrt(
                self_norm_squared / reference_norm_squared
            ),
        }
    return fits


def teacher_orthogonal_block(
    gram: np.ndarray,
    labels: list[str],
    teacher_label: str,
) -> dict[str, Any]:
    """Geometry of what is left after projecting the teacher's direction out.

    Writing ``v_i = alpha_i v_T + e_i``, the Gram matrix of the residuals is the
    Schur complement of the teacher's diagonal entry, so no new inner products
    are needed.  The question this answers is the one that decides whether the
    endpoints' shortfall is per-arm noise or a systematic second direction: if
    the residuals were noise their pairwise cosines would sit near zero, and if
    they share a direction the residual spectrum collapses to one component.
    """
    index = {label: position for position, label in enumerate(labels)}
    pivot = index[teacher_label]
    teacher_norm_squared = float(gram[pivot, pivot])
    if teacher_norm_squared <= 0:
        raise ValueError(f"teacher {teacher_label!r} has no functional change")
    residual_gram = gram - np.outer(gram[:, pivot], gram[pivot, :]) / teacher_norm_squared
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    residual_norms = np.sqrt(np.clip(np.diag(residual_gram), 0.0, None))
    others = [label for label in labels if label != teacher_label]
    positions = [index[label] for label in others]
    block = residual_gram[np.ix_(positions, positions)]
    return {
        "teacher_label": teacher_label,
        "residual_norm": {
            label: float(residual_norms[index[label]]) for label in others
        },
        "residual_over_self": {
            label: (
                None
                if norms[index[label]] <= 0
                else float(residual_norms[index[label]] / norms[index[label]])
            )
            for label in others
        },
        "residual_cosine": cosine_table(block, others),
        "residual_spectrum": spectrum(block, others),
    }


def spectrum(gram: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Singular spectrum of the stacked endpoint matrix, via the Gram eigenvalues.

    A family that lies on one ray puts all of its squared energy in the first
    component.  The second component is reported in units of the first because
    that ratio, not the energy fraction, is what a reader needs to judge whether
    a second direction is real or numerical noise.
    """
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    singular = np.sqrt(eigenvalues)
    total = float(eigenvalues.sum())
    return {
        "labels": labels,
        "singular_values": [float(value) for value in singular],
        "first_energy_fraction": None if total <= 0 else float(eigenvalues[0] / total),
        "second_over_first_singular": (
            None
            if singular[0] <= 0 or len(singular) < 2
            else float(singular[1] / singular[0])
        ),
        "participation_ratio": (
            None
            if total <= 0
            else float(total**2 / float((eigenvalues**2).sum()))
        ),
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
        "--reference-label",
        required=True,
        help="endpoint the scalar fit is taken against, e.g. the smallest support",
    )
    parser.add_argument(
        "--subspace-label",
        action="append",
        default=None,
        help="restrict the spectrum to these labels; defaults to every endpoint",
    )
    parser.add_argument(
        "--teacher-label",
        help="if given, also report the geometry of the teacher-orthogonal residuals",
    )
    parser.add_argument(
        "--anchor-base-score",
        metavar="LABEL=NPZ",
        help="measure every change against this scoring of the base instead of the "
        "reference's own base log-probabilities; use it when the arms were scored at a "
        "higher precision than the pass that froze the token IDs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_manifest, reference = load_fisher_reference(args.reference)
    entries = [parse_labeled_path(value) for value in args.score]
    labels = [label for label, _ in entries]
    if len(set(labels)) != len(labels):
        raise ValueError("score labels must be unique")
    if args.reference_label not in labels:
        raise ValueError(f"reference label {args.reference_label!r} was not scored")
    subspace = args.subspace_label or labels
    missing = sorted(set(subspace) - set(labels))
    if missing:
        raise ValueError(f"subspace labels were not scored: {missing}")
    if args.teacher_label is not None and args.teacher_label not in labels:
        raise ValueError(f"teacher label {args.teacher_label!r} was not scored")

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
        shift = base_logprobs - reference["base_logprobs"]
        anchor = {
            "label": anchor_label,
            "path": str(anchor_path),
            "sha256": sha256_file(anchor_path),
            "checkpoint": anchor_manifest.get("checkpoint"),
            "log_probability_shift_rms": float(np.sqrt((shift**2).mean())),
        }
    offsets = reference["position_offsets"]
    domains = reference["domains"]
    slots = position_domains(domains, offsets)

    manifests: dict[str, dict[str, Any]] = {}
    vectors: dict[str, np.ndarray] = {}
    for label, path in entries:
        manifests[label], endpoint_logprobs = load_fisher_score(
            label,
            path,
            reference_path=args.reference,
            reference=reference,
        )
        vectors[label], _ = fisher_centered_vector(base_logprobs, endpoint_logprobs)
        print(f"[vector] {label} {vectors[label].shape}", flush=True)

    domain_slices = {
        "all": np.ones(len(slots), dtype=bool),
        **{
            str(domain): slots == domain
            for domain in dict.fromkeys(domains.tolist())
        },
    }

    report_domains: dict[str, Any] = {}
    for domain, mask in domain_slices.items():
        restricted = {label: vectors[label][mask] for label in labels}
        gram = gram_matrix(restricted)
        subspace_gram = gram[
            np.ix_(
                [labels.index(label) for label in subspace],
                [labels.index(label) for label in subspace],
            )
        ]
        report_domains[domain] = {
            "positions": int(mask.sum()),
            "norms": {
                label: float(math.sqrt(max(gram[index, index], 0.0)))
                for index, label in enumerate(labels)
            },
            "cosine": cosine_table(gram, labels),
            "scalar_fit_against_reference": scalar_fits(
                gram, labels, args.reference_label
            ),
            "spectrum": spectrum(subspace_gram, list(subspace)),
        }
        if args.teacher_label is not None:
            report_domains[domain]["teacher_orthogonal"] = teacher_orthogonal_block(
                gram, labels, args.teacher_label
            )
        print(f"[domain] {domain} positions={int(mask.sum())}", flush=True)

    payload = {
        "schema_version": REPORT_SCHEMA,
        "inputs": {
            "reference": {
                "path": str(args.reference),
                "sha256": sha256_file(args.reference),
                "cohort": reference_manifest.get("cohort"),
                "top_k": reference_manifest.get("scoring", {}).get("top_k"),
            },
            "scores": {
                label: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "checkpoint": manifests[label].get("checkpoint"),
                }
                for label, path in entries
            },
            "reference_label": args.reference_label,
            "subspace_labels": list(subspace),
            "anchor_base_score": anchor,
        },
        "representation": (
            "centered top-K Fisher vector, identical to fisher_centered_vector; "
            "all quantities are exact functions of the Gram matrix of the full "
            "flattened vectors, not of any per-probe scalar reduction"
        ),
        "domains": report_domains,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_json_atomic(args.output, payload)
    print(f"[write] {args.output}", flush=True)


if __name__ == "__main__":
    main()
