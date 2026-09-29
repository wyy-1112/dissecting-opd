#!/usr/bin/env python3
"""Reliability-aware, token-weighted SVD/D-optimal prompt selection.

For each prompt, response-token-mean gradients are first aggregated as

    g_i = sum_r T_ir g_ir / L_i,  where L_i = sum_r T_ir.

The support-size gate measures stable between-prompt energy with weighted
centering.  The selection SVD itself uses the requested uncentered rows
``sqrt(L_i) M_Adam^(1/2) g_i``.  Both are deliberately different from
equal-prompt geometry and the incorrect ``L_i g_i`` geometry, whose covariance
would over-weight length quadratically.

Selection uses discovery rollouts only.  Those rollouts are deterministically
split again into two internal folds.  The symmetrized cross-fold covariance
removes independent rollout noise in expectation; its positive eigenspectrum
defines the stable between-prompt SVD coordinates used by D-optimal selection.
The frozen confirmation rollouts are consulted only after the support and
selected IDs have been fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[3] / "outputs" / "opd_project_root"))
ANALYSIS = Path(__file__).resolve().parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from gradient_geometry_mixed_effects import (  # noqa: E402
    _averaged_gram,
    load_response_sketch_cohort,
    safe_subspace_overlap,
)
from svd_gradient_coverage import (  # noqa: E402
    attach_random_comparison,
    eigensystem,
    greedy_d_optimal,
    random_reference,
    set_metrics,
    sha256_file,
    spectrum_summary,
    top_coordinates,
)


SCHEMA = "opd_reliability_aware_svd_d_optimal_v1"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def projection_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum_projection": float(array.min()),
        "maximum_projection": float(array.max()),
        "per_projection": [float(value) for value in array],
    }


def deterministic_inner_folds(
    prompts: Sequence[str],
    sample_ids: Sequence[Sequence[int]],
    *,
    seed: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    folds = []
    for prompt, ids in zip(prompts, sample_ids, strict=True):
        ordered = list(range(len(ids)))
        random.Random(f"{seed}:{prompt}:reliability-fold").shuffle(ordered)
        midpoint = len(ordered) // 2
        if midpoint < 2 or len(ordered) % 2:
            raise ValueError(
                "reliability fitting requires an even discovery width of at least four"
            )
        folds.append(
            (
                tuple(sorted(ordered[:midpoint])),
                tuple(sorted(ordered[midpoint:])),
            )
        )
    return tuple(folds)


def prompt_token_aggregates(
    response_means: Any,
    response_lengths: Any,
) -> tuple[Any, Any]:
    """Return per-prompt token means g_i and effective-token totals L_i."""
    full = response_means[:, :, :, -1, :].double()
    lengths = response_lengths[:, :, -1].double()
    totals = lengths.sum(dim=1)
    if bool((totals <= 0).any()):
        raise ValueError("every prompt must have positive effective-token count")
    gradients = (full * lengths[None, :, :, None]).sum(dim=2)
    gradients = gradients / totals[None, :, None]
    return gradients, totals


def fold_prompt_token_aggregates(
    response_means: Any,
    response_lengths: Any,
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> tuple[Any, Any, Any, Any]:
    import torch

    first_gradients = []
    second_gradients = []
    first_lengths = []
    second_lengths = []
    full = response_means[:, :, :, -1, :].double()
    lengths = response_lengths[:, :, -1].double()
    for prompt_index, (left, right) in enumerate(folds):
        left_lengths = lengths[prompt_index, list(left)]
        right_lengths = lengths[prompt_index, list(right)]
        left_total = left_lengths.sum()
        right_total = right_lengths.sum()
        first_gradients.append(
            (
                full[:, prompt_index, list(left), :]
                * left_lengths[None, :, None]
            ).sum(dim=1)
            / left_total
        )
        second_gradients.append(
            (
                full[:, prompt_index, list(right), :]
                * right_lengths[None, :, None]
            ).sum(dim=1)
            / right_total
        )
        first_lengths.append(left_total)
        second_lengths.append(right_total)
    # list[prompt](seed, dim) -> (seed, prompt, dim)
    return (
        torch.stack(first_gradients, dim=1),
        torch.stack(first_lengths),
        torch.stack(second_gradients, dim=1),
        torch.stack(second_lengths),
    )


def token_weighted_residual_rows(
    prompt_gradients: Any,
    prompt_lengths: Any,
) -> tuple[Any, Any]:
    """Rows sqrt(L_i) (g_i-g_pool) and the token-weighted pool mean."""
    lengths = prompt_lengths.double()
    pool = (
        prompt_gradients.double() * lengths[None, :, None]
    ).sum(dim=1) / lengths.sum()
    residuals = prompt_gradients.double() - pool[:, None, :]
    return residuals * lengths.sqrt()[None, :, None], pool


def token_weighted_svd_rows(
    prompt_gradients: Any,
    prompt_lengths: Any,
) -> Any:
    """Rows sqrt(L_i) M_Adam^(1/2) g_i used by token-weighted SVD."""
    return (
        prompt_gradients.double()
        * prompt_lengths.double().sqrt()[None, :, None]
    )


def prompt_grams(
    first_rows: Any,
    second_rows: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _averaged_gram(first_rows),
        _averaged_gram(second_rows),
        _averaged_gram(first_rows, second_rows),
    )


def crossfit_spectrum(
    cross_gram: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    symmetric = (cross_gram + cross_gram.T) / 2
    signed, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(signed)[::-1]
    signed = signed[order]
    vectors = vectors[:, order]
    positive = np.maximum(signed, 0.0)
    negative_mass = float(-np.minimum(signed, 0.0).sum())
    positive_trace = float(positive.sum())
    if positive_trace <= 0:
        raise ValueError("cross-fit covariance has no positive stable energy")
    fractions = positive / positive_trace
    participation = float(positive_trace**2 / max(float(np.square(positive).sum()), 1e-24))
    report = {
        "positive_trace": positive_trace,
        "negative_eigenvalue_mass": negative_mass,
        "negative_to_positive_mass_ratio": negative_mass / positive_trace,
        "participation_ratio": participation,
        "top4_stable_energy_fraction": float(fractions[:4].sum()),
        "top8_stable_energy_fraction": float(fractions[:8].sum()),
        "positive_spectrum": spectrum_summary(positive),
        "top_signed_eigenvalues": [float(value) for value in signed[:16]],
        "top_positive_energy_fractions": [
            float(value) for value in fractions[:16]
        ],
        "coordinate_scaling": "unscaled token-weighted cross-fold covariance",
    }
    return report, positive, vectors, signed


def choose_support(
    spectrum: Mapping[str, Any],
    overlap4: Mapping[str, Any],
    *,
    top4_min_energy: float,
    m4_max_participation_ratio: float,
    min_rank4_mean_canonical_cosine: float,
) -> tuple[int, dict[str, Any]]:
    expected_reliable_rank = min(
        4,
        int(spectrum["positive_spectrum"]["rank"]),
    )
    gates = {
        "top4_stable_energy": (
            float(spectrum["top4_stable_energy_fraction"]) >= top4_min_energy
        ),
        "stable_participation_ratio": (
            float(spectrum["participation_ratio"])
            <= m4_max_participation_ratio
        ),
        "rank4_reproducibility": (
            expected_reliable_rank >= 1
            and int(overlap4.get("rank", 0)) >= expected_reliable_rank
            and float(overlap4.get("mean_canonical_cosine", 0.0))
            >= min_rank4_mean_canonical_cosine
        ),
    }
    support = 4 if all(gates.values()) else 8
    return support, {
        "decision": f"M{support}",
        "selection_used_confirmation": False,
        "m4_gates": gates,
        "thresholds": {
            "top4_min_stable_energy_fraction": top4_min_energy,
            "m4_max_stable_participation_ratio": m4_max_participation_ratio,
            "min_rank4_mean_canonical_cosine": (
                min_rank4_mean_canonical_cosine
            ),
            "rank4_available_stable_directions": expected_reliable_rank,
        },
        "fallback": "use M8 whenever any discovery-only M4 gate fails",
    }


def prompt_mean_reproducibility(
    discovery: Any,
    confirmation: Any,
) -> dict[str, Any]:
    left = discovery.double()
    right = confirmation.double()
    numerator = (left * right).sum(dim=2)
    denominator = left.norm(dim=2) * right.norm(dim=2)
    cosine = (numerator / denominator.clamp(min=1e-24)).cpu().numpy()
    return {
        "mean_across_prompts": projection_summary(cosine.mean(axis=1)),
        "minimum_prompt_mean": float(cosine.mean(axis=0).min()),
        "median_prompt_mean": float(np.median(cosine.mean(axis=0))),
    }


def selected_confirmation_report(
    indices: Sequence[int],
    discovery_gradients: Any,
    discovery_lengths: Any,
    confirmation_gradients: Any,
    confirmation_lengths: Any,
    *,
    rank: int,
    ridge: float,
    random_draws: int,
    seed: int,
) -> dict[str, Any]:
    first_rows = token_weighted_svd_rows(
        discovery_gradients,
        discovery_lengths,
    )
    second_rows = token_weighted_svd_rows(
        confirmation_gradients,
        confirmation_lengths,
    )
    first_gram, second_gram, cross_gram = prompt_grams(
        first_rows,
        second_rows,
    )
    second_values, second_vectors = eigensystem(second_gram)
    second_coordinates = top_coordinates(second_values, second_vectors, rank)
    reference = random_reference(
        second_gram,
        second_values,
        second_coordinates,
        support=len(indices),
        ridge=ridge,
        draws=random_draws,
        seed=seed,
    )
    metrics = set_metrics(
        indices,
        second_gram,
        second_values,
        second_coordinates,
        ridge=ridge,
    )
    return {
        "selection_used_for_support_or_ids": False,
        "subspace_reproducibility": safe_subspace_overlap(
            first_gram,
            second_gram,
            cross_gram,
            rank,
        ),
        "selected_set": attach_random_comparison(metrics, reference),
        "random_reference": reference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-max-response-tokens", type=int, default=16384)
    parser.add_argument("--expected-max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--expected-rollouts-per-prompt", type=int, default=32)
    parser.add_argument("--ridge-fraction", type=float, default=1e-3)
    parser.add_argument("--random-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--top4-min-energy", type=float, default=0.60)
    parser.add_argument(
        "--m4-max-participation-ratio",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--min-rank4-mean-canonical-cosine",
        type=float,
        default=0.50,
    )
    parser.add_argument("--prompt-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    import torch

    arguments = parse_args()
    if arguments.ridge_fraction <= 0 or arguments.random_draws < 2:
        raise SystemExit(
            "ridge-fraction must be positive and random-draws at least two"
        )
    if not 0 < arguments.top4_min_energy <= 1:
        raise SystemExit("--top4-min-energy must be in (0, 1]")
    if arguments.m4_max_participation_ratio < 4:
        raise SystemExit("--m4-max-participation-ratio must be at least four")
    if not 0 <= arguments.min_rank4_mean_canonical_cosine <= 1:
        raise SystemExit(
            "--min-rank4-mean-canonical-cosine must be in [0, 1]"
        )
    cohort_manifest = json.loads(
        arguments.cohort_manifest.read_text(encoding="utf-8")
    )
    if (
        int(cohort_manifest.get("max_tokens", -1))
        != arguments.expected_max_response_tokens
    ):
        raise SystemExit(
            "cohort response cap does not match the expected training cap: "
            f"{cohort_manifest.get('max_tokens')} != "
            f"{arguments.expected_max_response_tokens}"
        )
    expected_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "samples_per_prompt": arguments.expected_rollouts_per_prompt,
    }
    if (
        int(cohort_manifest.get("max_prompt_tokens", -1))
        != arguments.expected_max_prompt_tokens
        or int(cohort_manifest.get("max_model_len", -1))
        != (
            arguments.expected_max_response_tokens
            + arguments.expected_max_prompt_tokens
        )
        or int(cohort_manifest.get("samples_per_prompt", -1))
        != arguments.expected_rollouts_per_prompt
        or cohort_manifest.get("sampling") != expected_sampling
    ):
        raise SystemExit(
            "cohort prompt, context, rollout count, or sampling protocol "
            "does not match training"
        )

    cohort = load_response_sketch_cohort(
        arguments.sketch_dir,
        label=arguments.label,
        split_manifest=arguments.split,
    )
    if cohort.prefixes != ("full",):
        raise SystemExit(
            "training-aligned selection requires full-response-only sketches; "
            f"found {cohort.prefixes}"
        )

    discovery_gradients, discovery_lengths = prompt_token_aggregates(
        cohort.discovery,
        cohort.discovery_counts,
    )
    confirmation_gradients, confirmation_lengths = prompt_token_aggregates(
        cohort.confirmation,
        cohort.confirmation_counts,
    )
    raw_discovery_gradients, raw_discovery_lengths = prompt_token_aggregates(
        cohort.raw_discovery,
        cohort.discovery_counts,
    )
    raw_confirmation_gradients, raw_confirmation_lengths = (
        prompt_token_aggregates(
            cohort.raw_confirmation,
            cohort.confirmation_counts,
        )
    )
    if not np.array_equal(
        discovery_lengths.cpu().numpy(),
        raw_discovery_lengths.cpu().numpy(),
    ) or not np.array_equal(
        confirmation_lengths.cpu().numpy(),
        raw_confirmation_lengths.cpu().numpy(),
    ):
        raise AssertionError("raw and Adam-metric prompt lengths differ")
    folds = deterministic_inner_folds(
        cohort.prompts,
        cohort.discovery_samples,
        seed=arguments.seed,
    )
    fold_a_gradients, fold_a_lengths, fold_b_gradients, fold_b_lengths = (
        fold_prompt_token_aggregates(
            cohort.discovery,
            cohort.discovery_counts,
            folds,
        )
    )
    fold_a_rows, fold_a_pool = token_weighted_residual_rows(
        fold_a_gradients,
        fold_a_lengths,
    )
    fold_b_rows, fold_b_pool = token_weighted_residual_rows(
        fold_b_gradients,
        fold_b_lengths,
    )
    fold_a_update_rows = token_weighted_svd_rows(
        fold_a_gradients,
        fold_a_lengths,
    )
    fold_b_update_rows = token_weighted_svd_rows(
        fold_b_gradients,
        fold_b_lengths,
    )
    first_gram, second_gram, cross_gram = prompt_grams(
        fold_a_rows,
        fold_b_rows,
    )
    spectrum, positive_values, vectors, signed_values = crossfit_spectrum(
        cross_gram
    )
    update_first_gram, update_second_gram, update_cross_gram = prompt_grams(
        fold_a_update_rows,
        fold_b_update_rows,
    )
    (
        update_spectrum,
        update_positive_values,
        update_vectors,
        update_signed_values,
    ) = crossfit_spectrum(update_cross_gram)
    overlap4 = safe_subspace_overlap(
        first_gram,
        second_gram,
        cross_gram,
        4,
    )
    overlap8 = safe_subspace_overlap(
        first_gram,
        second_gram,
        cross_gram,
        8,
    )
    update_overlap4 = safe_subspace_overlap(
        update_first_gram,
        update_second_gram,
        update_cross_gram,
        4,
    )
    update_overlap8 = safe_subspace_overlap(
        update_first_gram,
        update_second_gram,
        update_cross_gram,
        8,
    )
    support, support_gate = choose_support(
        spectrum,
        overlap4,
        top4_min_energy=arguments.top4_min_energy,
        m4_max_participation_ratio=arguments.m4_max_participation_ratio,
        min_rank4_mean_canonical_cosine=(
            arguments.min_rank4_mean_canonical_cosine
        ),
    )
    rank = min(support, int(np.count_nonzero(update_positive_values > 0)))
    coordinates = update_vectors[:, :rank] * np.sqrt(
        update_positive_values[:rank]
    )[None, :]
    # Cross-fold coordinates carry sqrt(sqrt(L_i^A L_i^B)) exposure.
    # Remove that exposure to recover stable z_i, then apply the full
    # discovery L_i explicitly in the D-optimal information matrix.
    prompt_lengths_np = discovery_lengths.cpu().numpy()
    crossfit_lengths = np.sqrt(
        fold_a_lengths.cpu().numpy() * fold_b_lengths.cpu().numpy()
    )
    z_coordinates = coordinates / np.sqrt(crossfit_lengths)[:, None]
    d_optimal_coordinates = (
        np.sqrt(prompt_lengths_np)[:, None] * z_coordinates
    )
    explicit_information = np.einsum(
        "n,ni,nj->ij",
        prompt_lengths_np,
        z_coordinates,
        z_coordinates,
        optimize=True,
    )
    if not np.allclose(
        d_optimal_coordinates.T @ d_optimal_coordinates,
        explicit_information,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError("D-optimal rows do not implement sum_i L_i z_i z_i^T")
    full_information_trace = float(np.square(d_optimal_coordinates).sum())
    ridge = arguments.ridge_fraction * full_information_trace / rank
    selected = greedy_d_optimal(
        d_optimal_coordinates,
        support,
        ridge=ridge,
    )
    stable_gram = d_optimal_coordinates @ d_optimal_coordinates.T
    stable_values, stable_vectors = eigensystem(stable_gram)
    stable_coordinates = top_coordinates(stable_values, stable_vectors, rank)
    discovery_reference = random_reference(
        stable_gram,
        stable_values,
        stable_coordinates,
        support=support,
        ridge=ridge,
        draws=arguments.random_draws,
        seed=arguments.seed + 17,
    )
    discovery_metrics = set_metrics(
        selected,
        stable_gram,
        stable_values,
        stable_coordinates,
        ridge=ridge,
    )

    inner_fold_ids = {
        str(prompt): {
            "fold_a": [
                int(cohort.discovery_samples[index][position])
                for position in folds[index][0]
            ],
            "fold_b": [
                int(cohort.discovery_samples[index][position])
                for position in folds[index][1]
            ],
        }
        for index, prompt in enumerate(cohort.prompts)
    }
    prompt_artifact = {
        "schema_version": "opd_prompt_token_aggregates_v1",
        "problem_ids": list(cohort.prompts),
        "projection_seeds": list(cohort.projection_seeds),
        "definition": {
            "g_i": "sum_r T_ir g_ir / L_i",
            "L_i": "sum_r T_ir",
            "stored_representation": (
                "three independent CountSketch projections of the "
                "parameter-space Adam-metric gradient"
            ),
            "svd_row": (
                "sqrt(L_i) * M_Adam^(1/2) * g_i"
            ),
            "d_optimal_information": (
                "lambda*I + sum_i L_i*z_i*z_i^T"
            ),
        },
        "inputs": {
            "split": str(arguments.split.resolve()),
            "split_sha256": sha256_file(arguments.split),
            "cohort_manifest": str(arguments.cohort_manifest.resolve()),
            "cohort_manifest_sha256": sha256_file(arguments.cohort_manifest),
            "projection_identity": cohort.projection_identity,
            "training_contract": cohort.training_contract,
        },
        "discovery": {
            "g_i": discovery_gradients.float().cpu(),
            "raw_g_i": raw_discovery_gradients.float().cpu(),
            "L_i": discovery_lengths.long().cpu(),
            "response_lengths": cohort.discovery_counts[:, :, -1].long().cpu(),
            "sample_ids": [list(values) for values in cohort.discovery_samples],
        },
        "confirmation": {
            "g_i": confirmation_gradients.float().cpu(),
            "raw_g_i": raw_confirmation_gradients.float().cpu(),
            "L_i": confirmation_lengths.long().cpu(),
            "response_lengths": cohort.confirmation_counts[:, :, -1].long().cpu(),
            "sample_ids": [list(values) for values in cohort.confirmation_samples],
        },
        "reliability_folds": {
            "assignment": inner_fold_ids,
            "fold_a_L_i": fold_a_lengths.long().cpu(),
            "fold_b_L_i": fold_b_lengths.long().cpu(),
            "crossfit_effective_L_i": torch.from_numpy(crossfit_lengths),
            "fold_a_pool_mean": fold_a_pool.float().cpu(),
            "fold_b_pool_mean": fold_b_pool.float().cpu(),
            "signed_between_prompt_crossfit_eigenvalues": torch.from_numpy(
                signed_values
            ),
            "signed_update_crossfit_eigenvalues": torch.from_numpy(
                update_signed_values
            ),
        },
        "stable_coordinates": {
            "rank": rank,
            "z_i": torch.from_numpy(z_coordinates),
            "sqrt_L_i_z_i": torch.from_numpy(d_optimal_coordinates),
            "z_i_exposure_correction": (
                "divide cross-fold coordinates by "
                "sqrt(sqrt(L_i_fold_a*L_i_fold_b)); then multiply by "
                "sqrt(full_discovery_L_i) for D-optimal information"
            ),
            "ridge_lambda": ridge,
            "ridge_fraction_of_mean_information_eigenvalue": (
                arguments.ridge_fraction
            ),
        },
    }
    arguments.prompt_artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact_temporary = arguments.prompt_artifact.with_suffix(
        arguments.prompt_artifact.suffix + ".tmp"
    )
    torch.save(prompt_artifact, artifact_temporary)
    artifact_temporary.replace(arguments.prompt_artifact)
    report = {
        "schema_version": SCHEMA,
        "analysis_regime": "prospective_pretraining_selection",
        "method": {
            "selection": (
                "greedy_D_optimal_on_positive_noise_corrected_"
                "discovery_cross_fold_SVD_coordinates"
            ),
            "prompt_estimator": (
                "g_i=sum_r T_ir*g_ir/L_i, L_i=sum_r T_ir"
            ),
            "svd_rows": (
                "sqrt(L_i)*M_Adam^(1/2)*g_i"
            ),
            "d_optimal_information": (
                "logdet(lambda*I + sum_{i in S} L_i*z_i*z_i^T)"
            ),
            "crossfit_exposure_correction": (
                "cross-fold coordinates are divided by "
                "sqrt(sqrt(L_i^A*L_i^B)) before full-discovery L_i is "
                "applied in D-optimal scoring"
            ),
            "noise_correction": (
                "symmetrized covariance between two disjoint internal "
                "discovery folds; independent rollout noise vanishes in expectation"
            ),
            "support_gate_geometry": (
                "stable between-prompt energy uses token-weighted centering "
                "g_i-g_pool; D-optimal selection uses the uncentered SVD rows"
            ),
            "confirmation_policy": (
                "confirmation rollouts do not choose support, rank, or IDs"
            ),
            "global_gradient_clipping": (
                "not applied per response; training applies one scalar after "
                "batch aggregation, which does not alter SVD directions"
            ),
        },
        "inputs": {
            "sketch_dir": str(arguments.sketch_dir.resolve()),
            "split": str(arguments.split.resolve()),
            "split_sha256": sha256_file(arguments.split),
            "cohort_manifest": str(arguments.cohort_manifest.resolve()),
            "cohort_manifest_sha256": sha256_file(arguments.cohort_manifest),
            "prompt_aggregate_artifact": str(
                arguments.prompt_artifact.resolve()
            ),
            "prompt_aggregate_artifact_sha256": sha256_file(
                arguments.prompt_artifact
            ),
            "label": arguments.label,
            "projection_identity": cohort.projection_identity,
            "training_contract": cohort.training_contract,
        },
        "cohort": {
            "prompts": len(cohort.prompts),
            "rollouts_per_discovery_half": int(cohort.discovery.shape[2]),
            "rollouts_per_confirmation_half": int(cohort.confirmation.shape[2]),
            "max_response_tokens": int(cohort_manifest["max_tokens"]),
            "discovery_effective_tokens": {
                "total": int(discovery_lengths.sum()),
                "per_prompt_mean": float(discovery_lengths.double().mean()),
                "per_prompt_minimum": int(discovery_lengths.min()),
                "per_prompt_maximum": int(discovery_lengths.max()),
            },
            "confirmation_effective_tokens": {
                "total": int(confirmation_lengths.sum()),
                "per_prompt_mean": float(confirmation_lengths.double().mean()),
                "per_prompt_minimum": int(confirmation_lengths.min()),
                "per_prompt_maximum": int(confirmation_lengths.max()),
            },
        },
        "discovery_only_reliability_fit": {
            "inner_folds_sha256": canonical_hash(inner_fold_ids),
            "inner_folds": inner_fold_ids,
            "noise_corrected_between_prompt_spectrum": spectrum,
            "rank4_subspace_reproducibility": overlap4,
            "rank8_subspace_reproducibility": overlap8,
            "noise_corrected_token_weighted_update_spectrum": update_spectrum,
            "update_rank4_subspace_reproducibility": update_overlap4,
            "update_rank8_subspace_reproducibility": update_overlap8,
        },
        "support_gate": support_gate,
        "selection": {
            "support": support,
            "rank": rank,
            "ridge_lambda": ridge,
            "ridge_fraction_of_mean_information_eigenvalue": (
                arguments.ridge_fraction
            ),
            "problem_effective_tokens": {
                str(prompt): int(discovery_lengths[index])
                for index, prompt in enumerate(cohort.prompts)
            },
            "selected_length_profile": {
                "total_effective_tokens": int(
                    discovery_lengths[selected].sum()
                ),
                "per_prompt_effective_tokens": [
                    int(discovery_lengths[index]) for index in selected
                ],
                "per_prompt_token_shares": [
                    float(
                        discovery_lengths[index]
                        / discovery_lengths[selected].sum()
                    )
                    for index in selected
                ],
            },
            "problem_ids": [
                str(cohort.prompts[index]) for index in selected
            ],
            "discovery_metrics": attach_random_comparison(
                discovery_metrics,
                discovery_reference,
            ),
            "random_reference": discovery_reference,
        },
        "confirmation": {
            "prompt_mean_reproducibility": prompt_mean_reproducibility(
                discovery_gradients,
                confirmation_gradients,
            ),
            **selected_confirmation_report(
                selected,
                discovery_gradients,
                discovery_lengths,
                confirmation_gradients,
                confirmation_lengths,
                rank=rank,
                ridge=ridge,
                random_draws=arguments.random_draws,
                seed=arguments.seed + 29,
            ),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    report = json_safe(report)
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "support": support,
                "rank": rank,
                "problem_ids": report["selection"]["problem_ids"],
                "support_gate": support_gate,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
