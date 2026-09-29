#!/usr/bin/env python3
"""Decompose prompt/rollout OPD gradients under three aggregation regimes.

The response-level random-effects model is

    x_ir = mu + a_i + epsilon_ir,

where ``i`` indexes prompts and ``r`` indexes disjoint on-policy rollouts.
Every reported geometry is computed independently in each CountSketch
projection and then averaged; projection spread remains in the report.

Three views use exactly the same responses:

* ``equal_prompt``: one response-token-mean gradient per rollout;
* ``token_weighted``: response gradients weighted exactly as VERL's global
  token-mean batch reduction;
* ``position_reliability_weighted``: disjoint token-block gradient sums
  weighted by discovery-only aligned contribution per token and split-half
  reproducibility.

Global norm clipping is deliberately not applied to individual ``x_ir``:
training clips the aggregate batch gradient, and per-response clipping would
change the estimand. The raw-gradient pooled clip factor is applied as one
common scalar to each split's additive Adam-metric features and also reported.
Adam must already have been applied in parameter space before
CountSketch, except at fresh optimizer step zero where the pre-update metric is
isotropic and therefore differs only by a common scalar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[3] / "outputs" / "opd_project_root"))
ANALYSIS = Path(__file__).resolve().parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from svd_gradient_coverage import (  # noqa: E402
    eigensystem,
    spectrum_summary,
    subspace_overlap,
)


SCHEMA = "opd_prompt_rollout_gradient_geometry_v1"
EXPECTED_OBJECTIVE = "fixed_base_detached_k1_policy_gradient_v1"
EXPECTED_REDUCTION = "response_token_mean"
FRESH_ADAM_METRIC = "fresh_adam_pre_update_isotropic_v1"
PRECONDITIONED_ADAM_METRIC = "parameter_space_adam_preconditioned_v1"


@dataclass(frozen=True)
class ResponseSketchCohort:
    prompts: tuple[str, ...]
    discovery_samples: tuple[tuple[int, ...], ...]
    confirmation_samples: tuple[tuple[int, ...], ...]
    prefixes: tuple[str, ...]
    projection_seeds: tuple[int, ...]
    discovery: Any
    confirmation: Any
    raw_discovery: Any
    raw_confirmation: Any
    discovery_counts: Any
    confirmation_counts: Any
    projection_identity: Mapping[str, Any]
    training_contract: Mapping[str, Any]
    source_files: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError("rank lists must contain positive integers")
    return tuple(dict.fromkeys(values))


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _projection_identity(payload: Mapping[str, Any], dimension: int) -> dict[str, Any]:
    family = payload.get("projection_family")
    if not isinstance(family, Mapping):
        raise ValueError("multi-projection family metadata is required")
    layout = payload.get("parameter_layout")
    return {
        "contract": family.get("contract"),
        "dimension": int(dimension),
        "seeds": [int(value) for value in family.get("seeds", ())],
        "projection_ids": list(family.get("projection_ids", ())),
        "parameter_layout_sha256": (
            _canonical_json_hash(layout) if layout is not None else None
        ),
    }


def _training_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = payload.get("optimizer_geometry")
    return {
        "objective_contract": payload.get("objective_contract"),
        "gradient_semantics": payload.get("gradient_semantics"),
        "loss_reduction": payload.get("loss_reduction"),
        "batch_aggregation": payload.get("batch_aggregation"),
        "k1_clamp": payload.get("k1_clamp"),
        "mask_contract": payload.get("mask_contract"),
        "ppo_contract": payload.get("ppo_contract"),
        "rollout_correction_contract": payload.get(
            "rollout_correction_contract"
        ),
        "optimizer_geometry": optimizer,
    }


def validate_training_contract(
    contract: Mapping[str, Any],
    *,
    allow_legacy_euclidean: bool,
) -> None:
    if contract.get("objective_contract") != EXPECTED_OBJECTIVE:
        raise ValueError(
            "gradient sketches do not use the fixed detached K1 policy-gradient "
            f"contract: {contract.get('objective_contract')!r}"
        )
    if contract.get("loss_reduction") != EXPECTED_REDUCTION:
        raise ValueError("gradient sketches are not response-token means")
    if not math.isclose(
        float(contract.get("k1_clamp", float("nan"))),
        10.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("K1 loss clamp must exactly match training value 10")

    optimizer = contract.get("optimizer_geometry")
    kind = optimizer.get("adam_metric_contract") if isinstance(optimizer, Mapping) else None
    if kind not in {FRESH_ADAM_METRIC, PRECONDITIONED_ADAM_METRIC}:
        if allow_legacy_euclidean:
            return
        raise ValueError(
            "official geometry requires parameter-space Adam metric metadata; "
            "regenerate sketches or pass --allow-legacy-euclidean only for diagnostics"
        )
    if contract.get("mask_contract") is None:
        raise ValueError("official geometry requires an explicit response/loss-mask contract")
    if contract.get("ppo_contract") is None:
        raise ValueError("official geometry requires an explicit PPO clipping contract")
    rollout_correction = contract.get("rollout_correction_contract")
    if (
        not isinstance(rollout_correction, Mapping)
        or not rollout_correction.get("enabled", False)
        or rollout_correction.get("mode") != "token"
        or float(rollout_correction.get("threshold", float("nan"))) != 5.0
        or rollout_correction.get("batch_normalize") is not False
    ):
        raise ValueError(
            "official geometry requires the training-aligned token rollout "
            "importance correction (threshold=5, no batch normalization)"
        )


def load_response_sketch_cohort(
    sketch_dir: Path,
    *,
    label: str,
    split_manifest: Path,
    allow_legacy_euclidean: bool = False,
) -> ResponseSketchCohort:
    """Load rectangular response-level sketches without averaging rollouts."""
    import torch

    matrices: dict[tuple[str, int], Any] = {}
    raw_matrices: dict[tuple[str, int], Any] = {}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    prefixes: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    identity: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    source_files: list[str] = []

    for path in sorted(sketch_dir.glob("shard*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if label not in payload.get("sketches", {}):
            raise ValueError(f"{path}: missing teacher label {label!r}")
        block = payload["sketches"][label].float()
        raw_payload = payload.get("raw_gradient_sketches", {})
        if label in raw_payload:
            raw_block = raw_payload[label].float()
        elif allow_legacy_euclidean:
            raw_block = block
        else:
            raise ValueError(
                f"{path}: raw gradient sketches are required to apply "
                "pre-Adam global norm clipping"
            )
        if raw_block.shape != block.shape:
            raise ValueError(f"{path}: raw and Adam-metric sketch shapes differ")
        if block.ndim != 4:
            raise ValueError(
                f"{path}: expected [response, projection, prefix, dimension], "
                f"found rank {block.ndim}"
            )
        shard_prefixes = tuple(str(value) for value in payload.get("prefixes", ()))
        shard_seeds = tuple(
            int(value)
            for value in payload.get("projection_family", {}).get("seeds", ())
        )
        if len(shard_prefixes) != block.shape[2]:
            raise ValueError(f"{path}: prefix metadata does not match sketch tensor")
        if len(shard_seeds) != block.shape[1]:
            raise ValueError(f"{path}: projection seeds do not match sketch tensor")
        shard_identity = _projection_identity(payload, block.shape[-1])
        shard_contract = _training_contract(payload)
        if prefixes and shard_prefixes != prefixes:
            raise ValueError(f"{path}: prefix family differs across shards")
        if seeds and shard_seeds != seeds:
            raise ValueError(f"{path}: projection family differs across shards")
        if identity is not None and shard_identity != identity:
            raise ValueError(f"{path}: projection identity differs across shards")
        if contract is not None and shard_contract != contract:
            raise ValueError(f"{path}: training contract differs across shards")
        prefixes = shard_prefixes
        seeds = shard_seeds
        identity = shard_identity
        contract = shard_contract

        shard_records = payload.get("records", ())
        if len(shard_records) != block.shape[0]:
            raise ValueError(f"{path}: record and response counts differ")
        for index, raw_record in enumerate(shard_records):
            record = dict(raw_record)
            key = (str(record["problem_id"]), int(record["sample_index"]))
            if key in matrices:
                raise ValueError(f"duplicate response key {key}")
            matrices[key] = block[index]
            raw_matrices[key] = raw_block[index]
            records[key] = record
        source_files.append(str(path.resolve()))

    if not matrices or identity is None or contract is None:
        raise FileNotFoundError(f"no shard*.pt under {sketch_dir}")
    validate_training_contract(
        contract,
        allow_legacy_euclidean=allow_legacy_euclidean,
    )

    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    assignment = split.get("assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError(f"{split_manifest}: missing assignment")
    prompts = tuple(
        sorted(
            (str(value) for value in assignment),
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
    )
    widths = {
        (
            len(assignment[prompt]["discovery"]),
            len(assignment[prompt]["confirmation"]),
        )
        for prompt in prompts
    }
    if len(widths) != 1:
        raise ValueError("split manifest is not rectangular")
    discovery_width, confirmation_width = next(iter(widths))
    if discovery_width != confirmation_width or discovery_width < 2:
        raise ValueError("equal split halves with at least two rollouts are required")

    def materialize(
        half: str,
        values_by_key: Mapping[tuple[str, int], Any],
    ) -> tuple[Any, Any, tuple[tuple[int, ...], ...]]:
        prompt_blocks = []
        prompt_counts = []
        all_sample_ids: list[tuple[int, ...]] = []
        for prompt in prompts:
            sample_ids = tuple(int(value) for value in assignment[prompt][half])
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError(f"{prompt}: duplicate {half} sample IDs")
            all_sample_ids.append(sample_ids)
            response_blocks = []
            response_counts = []
            for sample_id in sample_ids:
                key = (prompt, sample_id)
                if key not in values_by_key:
                    raise ValueError(f"{split_manifest}: response {key} absent from sketches")
                record = records[key]
                counts_by_prefix = record.get("prefix_token_counts")
                if not isinstance(counts_by_prefix, Mapping):
                    raise ValueError(f"{key}: missing prefix token counts")
                counts = [int(counts_by_prefix[prefix]) for prefix in prefixes]
                if any(value < 1 for value in counts):
                    raise ValueError(f"{key}: non-positive prefix token count")
                if any(right < left for left, right in zip(counts, counts[1:])):
                    raise ValueError(f"{key}: prefix token counts are not monotone")
                if counts[-1] != int(record["response_tokens"]):
                    raise ValueError(f"{key}: full-prefix token count mismatch")
                response_blocks.append(values_by_key[key])
                response_counts.append(counts)
            # [rollout, seed, prefix, dimension]
            prompt_blocks.append(torch.stack(response_blocks))
            prompt_counts.append(torch.tensor(response_counts, dtype=torch.float64))
        # [prompt, rollout, seed, prefix, dimension] -> [seed, prompt, rollout, prefix, dimension]
        values = torch.stack(prompt_blocks).permute(2, 0, 1, 3, 4).contiguous()
        counts = torch.stack(prompt_counts)
        return values, counts, tuple(all_sample_ids)

    discovery, discovery_counts, discovery_samples = materialize(
        "discovery",
        matrices,
    )
    confirmation, confirmation_counts, confirmation_samples = materialize(
        "confirmation",
        matrices,
    )
    raw_discovery, raw_discovery_counts, raw_discovery_samples = materialize(
        "discovery",
        raw_matrices,
    )
    raw_confirmation, raw_confirmation_counts, raw_confirmation_samples = materialize(
        "confirmation",
        raw_matrices,
    )
    if (
        not np.array_equal(discovery_counts.numpy(), raw_discovery_counts.numpy())
        or not np.array_equal(
            confirmation_counts.numpy(),
            raw_confirmation_counts.numpy(),
        )
        or discovery_samples != raw_discovery_samples
        or confirmation_samples != raw_confirmation_samples
    ):
        raise AssertionError("raw and Adam-metric cohort materialization diverged")
    discovery_keys = {
        (prompt, sample)
        for prompt, samples in zip(prompts, discovery_samples, strict=True)
        for sample in samples
    }
    confirmation_keys = {
        (prompt, sample)
        for prompt, samples in zip(prompts, confirmation_samples, strict=True)
        for sample in samples
    }
    if discovery_keys & confirmation_keys:
        raise ValueError("discovery and confirmation responses overlap")

    return ResponseSketchCohort(
        prompts=prompts,
        discovery_samples=discovery_samples,
        confirmation_samples=confirmation_samples,
        prefixes=prefixes,
        projection_seeds=seeds,
        discovery=discovery,
        confirmation=confirmation,
        raw_discovery=raw_discovery,
        raw_confirmation=raw_confirmation,
        discovery_counts=discovery_counts,
        confirmation_counts=confirmation_counts,
        projection_identity=identity,
        training_contract=contract,
        source_files=tuple(source_files),
    )


def reconstruct_block_sums(cumulative: Any, counts: Any) -> tuple[Any, Any]:
    """Recover disjoint masked-token gradient sums from cumulative means."""
    import torch

    if cumulative.ndim != 5 or counts.ndim != 3:
        raise ValueError("unexpected cumulative sketch/count rank")
    if tuple(cumulative.shape[1:4]) != tuple(counts.shape):
        raise ValueError("cumulative sketch and token-count shapes differ")
    count_tensor = counts.to(dtype=cumulative.dtype, device=cumulative.device)
    cumulative_sums = cumulative * count_tensor[None, ..., None]
    zeros = torch.zeros_like(cumulative_sums[:, :, :, :1, :])
    block_sums = cumulative_sums - torch.cat(
        (zeros, cumulative_sums[:, :, :, :-1, :]),
        dim=3,
    )
    zero_counts = torch.zeros_like(counts[:, :, :1])
    block_counts = counts - torch.cat((zero_counts, counts[:, :, :-1]), dim=2)
    return block_sums, block_counts


def _averaged_gram(left: Any, right: Any | None = None) -> np.ndarray:
    values = left.double()
    other = values if right is None else right.double()
    return (
        values.shape[0] ** -1
        * np.asarray(
            __import__("torch").einsum("snd,smd->nm", values, other).cpu()
        )
    )


def _cosine_per_projection(left: Any, right: Any) -> np.ndarray:
    numerator = (left.double() * right.double()).sum(dim=-1)
    denominator = left.double().norm(dim=-1) * right.double().norm(dim=-1)
    return (numerator / denominator.clamp(min=1e-24)).cpu().numpy()


def projection_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum_projection": float(array.min()),
        "maximum_projection": float(array.max()),
        "per_projection": [float(value) for value in array],
    }


def safe_subspace_overlap(
    first_gram: np.ndarray,
    second_gram: np.ndarray,
    cross_gram: np.ndarray,
    rank: int,
) -> dict[str, Any]:
    first_values = eigensystem(first_gram)[0]
    second_values = eigensystem(second_gram)[0]
    first_positive = int(
        np.count_nonzero(first_values > max(first_values[0] * 1e-12, 1e-15))
    )
    second_positive = int(
        np.count_nonzero(second_values > max(second_values[0] * 1e-12, 1e-15))
    )
    effective = min(rank, first_positive, second_positive)
    if effective < 1:
        return {
            "rank": 0,
            "mean_canonical_cosine": 0.0,
            "mean_squared_canonical_cosine": 0.0,
            "minimum_canonical_cosine": 0.0,
            "canonical_cosines": [],
        }
    return subspace_overlap(
        first_gram,
        second_gram,
        cross_gram,
        effective,
    )


def fit_position_reliability_profile(
    discovery: Any,
    counts: Any,
    prefixes: Sequence[str],
    *,
    subspace_rank: int,
) -> dict[str, Any]:
    """Fit block weights on two internal discovery folds only."""
    import torch

    if discovery.shape[2] < 4 or discovery.shape[2] % 2:
        raise ValueError(
            "position/reliability fitting requires an even discovery width >= 4"
        )
    block_sums, block_counts = reconstruct_block_sums(discovery, counts)
    fold_width = discovery.shape[2] // 2
    fold_a = block_sums[:, :, :fold_width]
    fold_b = block_sums[:, :, fold_width:]
    full_pool = block_sums.sum(dim=3).mean(dim=(1, 2))
    full_norm_sq = full_pool.double().square().sum(dim=1).clamp(min=1e-24)
    total_tokens = float(block_counts.sum())
    block_token_totals = block_counts.sum(dim=(0, 1)).cpu().numpy()

    rows: list[dict[str, Any]] = []
    raw_scores = []
    previous = "0"
    for block_index, prefix in enumerate(prefixes):
        contribution = block_sums[:, :, :, block_index].mean(dim=(1, 2))
        aligned = (
            (contribution.double() * full_pool.double()).sum(dim=1)
            / full_norm_sq
        ).cpu().numpy()
        token_fraction = float(block_token_totals[block_index] / total_tokens)
        efficiency = float(aligned.mean() / token_fraction) if token_fraction else 0.0

        prompt_a = fold_a[:, :, :, block_index].mean(dim=2)
        prompt_b = fold_b[:, :, :, block_index].mean(dim=2)
        mean_a = prompt_a.mean(dim=1)
        mean_b = prompt_b.mean(dim=1)
        mean_cosines = _cosine_per_projection(mean_a, mean_b)
        centered_a = prompt_a - mean_a[:, None, :]
        centered_b = prompt_b - mean_b[:, None, :]
        prompt_all = block_sums[:, :, :, block_index].mean(dim=2)
        common_all = prompt_all.mean(dim=1)
        residual_all = prompt_all - common_all[:, None, :]
        common_energy = common_all.double().square().sum(dim=1)
        residual_energy = residual_all.double().square().sum(dim=2).mean(dim=1)
        common_share = float(
            (
                common_energy
                / (common_energy + residual_energy).clamp(min=1e-24)
            )
            .mean()
            .cpu()
        )
        first_gram = _averaged_gram(centered_a)
        second_gram = _averaged_gram(centered_b)
        cross_gram = _averaged_gram(centered_a, centered_b)
        effective_rank = min(subspace_rank, centered_a.shape[1] - 1)
        overlap = safe_subspace_overlap(
            first_gram,
            second_gram,
            cross_gram,
            effective_rank,
        )
        mean_reliability = max(float(mean_cosines.mean()), 0.0)
        residual_reliability = max(
            float(overlap["mean_squared_canonical_cosine"]),
            0.0,
        )
        reliability = (
            common_share * mean_reliability
            + (1.0 - common_share) * residual_reliability
        )
        raw_score = max(efficiency, 0.0) * reliability
        raw_scores.append(raw_score)
        rows.append(
            {
                "block": f"{previous}:{prefix}",
                "token_fraction": token_fraction,
                "aligned_fraction": projection_summary(aligned),
                "aligned_fraction_per_token": efficiency,
                "common_mean_reproducibility": projection_summary(mean_cosines),
                "residual_subspace_reproducibility": overlap,
                "common_energy_share": common_share,
                "reliability": reliability,
                "raw_weight_score": raw_score,
            }
        )
        previous = prefix

    scores = np.asarray(raw_scores, dtype=np.float64)
    token_fractions = np.asarray(
        [float(row["token_fraction"]) for row in rows],
        dtype=np.float64,
    )
    normalization = float(scores @ token_fractions)
    if normalization <= 0:
        raise ValueError("position/reliability profile has no positive stable signal")
    weights = scores / normalization
    for row, weight in zip(rows, weights, strict=True):
        row["normalized_token_weight"] = float(weight)
    return {
        "fit_data": (
            "discovery rollouts split deterministically into two equal internal folds"
        ),
        "confirmation_used_for_weight_fitting": False,
        "formula": (
            "max(aligned_fraction_per_token,0) * "
            "[common_energy_share*max(common_mean_cosine,0) + "
            "(1-common_energy_share)*mean_squared_residual_canonical_cosine], "
            "then token-normalize"
        ),
        "subspace_rank": int(subspace_rank),
        "weights": [float(value) for value in weights],
        "blocks": rows,
    }


def equal_prompt_features(cumulative: Any) -> Any:
    return cumulative[:, :, :, -1, :].double()


def token_weighted_features(cumulative: Any, counts: Any) -> tuple[Any, float]:
    full = equal_prompt_features(cumulative)
    tokens = counts[:, :, -1].double()
    scale = float(tokens.mean())
    return full * (tokens / scale)[None, ..., None], scale


def position_reliability_features(
    cumulative: Any,
    counts: Any,
    weights: Sequence[float],
) -> tuple[Any, float]:
    import torch

    block_sums, block_counts = reconstruct_block_sums(cumulative, counts)
    weight = torch.as_tensor(
        weights,
        dtype=block_sums.dtype,
        device=block_sums.device,
    )
    if weight.numel() != block_sums.shape[3]:
        raise ValueError("position weights do not match token blocks")
    weighted_sums = (block_sums * weight[None, None, None, :, None]).sum(dim=3)
    weighted_tokens = (block_counts * weight[None, None, :]).sum(dim=2)
    scale = float(weighted_tokens.mean())
    if scale <= 0:
        raise ValueError("position/reliability weighted token scale is non-positive")
    return weighted_sums.double() / scale, scale


def _spectrum_from_prompt_effects(prompt_effects: Any) -> dict[str, Any]:
    prompt_count = prompt_effects.shape[1]
    gram = _averaged_gram(prompt_effects) / max(prompt_count - 1, 1)
    values, _ = eigensystem(gram)
    return {
        "trace": float(values.sum()),
        **spectrum_summary(values),
    }


def mixed_effects(features: Any) -> dict[str, Any]:
    """Balanced random-effects decomposition for one response feature regime."""
    prompt_count = int(features.shape[1])
    rollout_count = int(features.shape[2])
    if prompt_count < 2 or rollout_count < 2:
        raise ValueError("mixed-effects decomposition requires I>=2 and R>=2")

    values = features.double()
    grand = values.mean(dim=(1, 2))
    prompt_means = values.mean(dim=2)
    prompt_effects = prompt_means - grand[:, None, :]
    noise = values - prompt_means[:, :, None, :]

    common = grand.square().sum(dim=1).cpu().numpy()
    between_empirical = (
        prompt_effects.square().sum(dim=2).mean(dim=1).cpu().numpy()
    )
    within_empirical = noise.square().sum(dim=3).mean(dim=(1, 2)).cpu().numpy()
    total = values.square().sum(dim=3).mean(dim=(1, 2)).cpu().numpy()
    identity_error = total - common - between_empirical - within_empirical

    ss_between = prompt_effects.square().sum(dim=(1, 2)).cpu().numpy()
    ss_within = noise.square().sum(dim=(1, 2, 3)).cpu().numpy()
    ms_between = rollout_count * ss_between / (prompt_count - 1)
    ms_within = ss_within / (prompt_count * (rollout_count - 1))
    between_debiased = (ms_between - ms_within) / rollout_count
    common_debiased = (
        common
        - between_debiased / prompt_count
        - ms_within / (prompt_count * rollout_count)
    )

    denominator = np.maximum(total, 1e-24)
    return {
        "prompts": prompt_count,
        "rollouts_per_prompt": rollout_count,
        "common_mean_energy": projection_summary(common),
        "common_mean_energy_debiased": projection_summary(common_debiased),
        "common_mean_energy_debiased_nonnegative": projection_summary(
            np.maximum(common_debiased, 0.0)
        ),
        "common_mean_energy_fraction": projection_summary(common / denominator),
        "between_prompt_empirical_energy": projection_summary(between_empirical),
        "between_prompt_empirical_fraction": projection_summary(
            between_empirical / denominator
        ),
        "within_prompt_rollout_noise_energy": projection_summary(within_empirical),
        "within_prompt_rollout_noise_fraction": projection_summary(
            within_empirical / denominator
        ),
        "random_effects_covariance_trace": {
            "between_prompt_debiased": projection_summary(between_debiased),
            "between_prompt_debiased_nonnegative": projection_summary(
                np.maximum(between_debiased, 0.0)
            ),
            "within_prompt": projection_summary(ms_within),
            "formula": (
                "sigma_a^2=(MS_between-MS_within)/R; "
                "sigma_epsilon^2=MS_within"
            ),
        },
        "total_response_second_moment": projection_summary(total),
        "anova_identity_error": projection_summary(identity_error),
        "between_prompt_covariance_spectrum": _spectrum_from_prompt_effects(
            prompt_effects
        ),
    }


def stable_residual_spectrum(
    first_gram: np.ndarray,
    second_gram: np.ndarray,
    cross_gram: np.ndarray,
    *,
    rank: int,
) -> dict[str, Any]:
    """Energy-weighted effective rank of reproducible principal directions."""
    first_values, first_vectors = eigensystem(first_gram)
    second_values, second_vectors = eigensystem(second_gram)
    first_positive = int(
        np.count_nonzero(first_values > max(first_values[0] * 1e-12, 1e-15))
    )
    second_positive = int(
        np.count_nonzero(second_values > max(second_values[0] * 1e-12, 1e-15))
    )
    effective = min(rank, first_positive, second_positive)
    if effective < 1:
        return {"rank_cap": int(rank), "effective_rank": 0}

    left = first_vectors[:, :effective] / np.sqrt(
        first_values[:effective]
    )[None, :]
    right = second_vectors[:, :effective] / np.sqrt(
        second_values[:effective]
    )[None, :]
    bridge = left.T @ cross_gram @ right
    u, canonical, vh = np.linalg.svd(bridge, full_matrices=False)
    canonical = np.clip(canonical, 0.0, 1.0)
    discovery_energy = np.diag(
        u.T @ np.diag(first_values[:effective]) @ u
    )
    confirmation_energy = np.diag(
        vh @ np.diag(second_values[:effective]) @ vh.T
    )
    stable_energy = np.square(canonical) * np.sqrt(
        np.maximum(discovery_energy * confirmation_energy, 0.0)
    )
    total = float(stable_energy.sum())
    squared = float(np.square(stable_energy).sum())
    return {
        "rank_cap": int(rank),
        "effective_rank": int(effective),
        "participation_ratio": total * total / squared if squared else 0.0,
        "stable_rank": (
            total / float(stable_energy.max())
            if stable_energy.size and stable_energy.max() > 0
            else 0.0
        ),
        "stable_energy": total,
        "directions_canonical_cosine_ge_0_5": int(
            np.count_nonzero(canonical >= 0.5)
        ),
        "canonical_cosines": [float(value) for value in canonical],
        "stable_energy_fractions": (
            [float(value) for value in stable_energy / total] if total else []
        ),
        "formula": (
            "e_k=canonical_cosine_k^2*sqrt(discovery_energy_k*"
            "confirmation_energy_k); effective rank=(sum e)^2/sum(e^2)"
        ),
    }


def prompt_mean_cosines(
    prompts: Sequence[str],
    discovery: Any,
    confirmation: Any,
) -> dict[str, Any]:
    first = discovery.double().mean(dim=2)
    second = confirmation.double().mean(dim=2)
    first_pool = first.mean(dim=1)
    second_pool = second.mean(dim=1)
    first_to_second = _cosine_per_projection(
        first,
        second_pool[:, None, :],
    )
    second_to_first = _cosine_per_projection(
        second,
        first_pool[:, None, :],
    )
    rows = []
    for index, prompt in enumerate(prompts):
        rows.append(
            {
                "problem_id": str(prompt),
                "discovery_prompt_to_confirmation_pool": projection_summary(
                    first_to_second[:, index]
                ),
                "confirmation_prompt_to_discovery_pool": projection_summary(
                    second_to_first[:, index]
                ),
            }
        )
    pooled = _cosine_per_projection(first_pool, second_pool)
    return {
        "full_pool_mean_discovery_confirmation_cosine": projection_summary(pooled),
        "per_prompt": rows,
        "cross_fit_contract": (
            "each prompt half is compared with the opposite half's full-pool mean"
        ),
    }


def cross_fit_between_covariance(
    cross_gram: np.ndarray,
    prompt_count: int,
) -> dict[str, Any]:
    """Noise-independent between-prompt covariance from disjoint rollout halves."""
    symmetric = (cross_gram + cross_gram.T) / (2 * max(prompt_count - 1, 1))
    signed = np.linalg.eigvalsh(symmetric)[::-1]
    positive = np.maximum(signed, 0.0)
    negative = np.minimum(signed, 0.0)
    return {
        "signed_trace": float(signed.sum()),
        "positive_trace": float(positive.sum()),
        "negative_eigenvalue_mass": float(-negative.sum()),
        "positive_spectrum": spectrum_summary(positive),
        "top_signed_eigenvalues": [float(value) for value in signed[:16]],
        "contract": (
            "symmetrized discovery-confirmation cross covariance; independent "
            "rollout noise has zero expectation and negative mass diagnoses "
            "finite-sample instability"
        ),
    }


def cross_half_geometry(
    prompts: Sequence[str],
    discovery: Any,
    confirmation: Any,
    *,
    ranks: Sequence[int],
    stable_rank_cap: int,
) -> dict[str, Any]:
    first = discovery.double().mean(dim=2)
    second = confirmation.double().mean(dim=2)
    first_centered = first - first.mean(dim=1, keepdim=True)
    second_centered = second - second.mean(dim=1, keepdim=True)
    first_gram = _averaged_gram(first_centered)
    second_gram = _averaged_gram(second_centered)
    cross_gram = _averaged_gram(first_centered, second_centered)
    maximum = len(prompts) - 1
    return {
        "cross_fit_between_prompt_covariance": cross_fit_between_covariance(
            cross_gram,
            len(prompts),
        ),
        "discovery_confirmation_subspace_stability": {
            str(rank): safe_subspace_overlap(
                first_gram,
                second_gram,
                cross_gram,
                min(rank, maximum),
            )
            for rank in ranks
            if rank <= maximum
        },
        "stable_residual_effective_rank": stable_residual_spectrum(
            first_gram,
            second_gram,
            cross_gram,
            rank=min(stable_rank_cap, maximum),
        ),
        "prompt_gradient_to_full_pool_mean": prompt_mean_cosines(
            prompts,
            discovery,
            confirmation,
        ),
    }


def pooled_clip_factors(raw_features: Any, clip_norm: float) -> np.ndarray:
    pooled = raw_features.double().mean(dim=(1, 2))
    norms = pooled.norm(dim=1).cpu().numpy()
    return np.minimum(1.0, clip_norm / np.maximum(norms, 1e-24))


def pooled_clip_report(raw_features: Any, clip_norm: float) -> dict[str, Any]:
    pooled = raw_features.double().mean(dim=(1, 2))
    norms = pooled.norm(dim=1).cpu().numpy()
    factors = pooled_clip_factors(raw_features, clip_norm)
    return {
        "clip_norm": float(clip_norm),
        "raw_aggregate_preclip_norm": projection_summary(norms),
        "aggregate_clip_factor": projection_summary(factors),
        "contract": (
            "factor is estimated from independent raw-gradient CountSketches "
            "before Adam preconditioning and applied once to the pooled batch; "
            "individual x_ir are never clipped"
        ),
    }


def regime_report(
    prompts: Sequence[str],
    discovery: Any,
    confirmation: Any,
    raw_discovery: Any,
    raw_confirmation: Any,
    *,
    ranks: Sequence[int],
    stable_rank_cap: int,
    clip_norm: float,
    scaling: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    discovery_factors = pooled_clip_factors(raw_discovery, clip_norm)
    confirmation_factors = pooled_clip_factors(raw_confirmation, clip_norm)
    clipped_discovery = discovery * torch.as_tensor(
        discovery_factors,
        dtype=discovery.dtype,
        device=discovery.device,
    )[:, None, None, None]
    clipped_confirmation = confirmation * torch.as_tensor(
        confirmation_factors,
        dtype=confirmation.dtype,
        device=confirmation.device,
    )[:, None, None, None]
    return {
        "scaling": dict(scaling),
        "geometry_coordinates": (
            "Adam-metric gradients after the one pooled raw-gradient clipping "
            "factor for that split and projection"
        ),
        "discovery": mixed_effects(clipped_discovery),
        "confirmation": mixed_effects(clipped_confirmation),
        "cross_half": cross_half_geometry(
            prompts,
            clipped_discovery,
            clipped_confirmation,
            ranks=ranks,
            stable_rank_cap=stable_rank_cap,
        ),
        "global_gradient_clipping": {
            "discovery": pooled_clip_report(raw_discovery, clip_norm),
            "confirmation": pooled_clip_report(raw_confirmation, clip_norm),
        },
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    base = ROOT / "data/opd/coverage_selection_v2"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sketch-dir",
        type=Path,
        default=base / "sketches_geometry_v1",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=base / "split_manifest.json",
    )
    parser.add_argument("--label", default="nonhom_30b")
    parser.add_argument("--ranks", default="1,2,4,8,16,32")
    parser.add_argument("--stable-rank-cap", type=int, default=32)
    parser.add_argument("--position-subspace-rank", type=int, default=4)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--allow-legacy-euclidean",
        action="store_true",
        help="diagnostic only; official reports require Adam metric metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/analysis/gradient_geometry_96_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.clip_norm <= 0:
        raise SystemExit("--clip-norm must be positive")
    if arguments.stable_rank_cap < 1 or arguments.position_subspace_rank < 1:
        raise SystemExit("rank caps must be positive")
    ranks = parse_int_list(arguments.ranks)
    cohort = load_response_sketch_cohort(
        arguments.sketch_dir,
        label=arguments.label,
        split_manifest=arguments.split,
        allow_legacy_euclidean=arguments.allow_legacy_euclidean,
    )
    if len(cohort.prefixes) < 2:
        raise SystemExit(
            "position/reliability geometry requires multi-prefix sketches; "
            f"found prefixes={cohort.prefixes}"
        )

    equal_discovery = equal_prompt_features(cohort.discovery)
    equal_confirmation = equal_prompt_features(cohort.confirmation)
    raw_equal_discovery = equal_prompt_features(cohort.raw_discovery)
    raw_equal_confirmation = equal_prompt_features(cohort.raw_confirmation)
    token_discovery, discovery_token_scale = token_weighted_features(
        cohort.discovery,
        cohort.discovery_counts,
    )
    token_confirmation, confirmation_token_scale = token_weighted_features(
        cohort.confirmation,
        cohort.confirmation_counts,
    )
    raw_token_discovery, raw_discovery_token_scale = token_weighted_features(
        cohort.raw_discovery,
        cohort.discovery_counts,
    )
    raw_token_confirmation, raw_confirmation_token_scale = (
        token_weighted_features(
            cohort.raw_confirmation,
            cohort.confirmation_counts,
        )
    )
    if (
        discovery_token_scale != raw_discovery_token_scale
        or confirmation_token_scale != raw_confirmation_token_scale
    ):
        raise AssertionError("raw and Adam-metric token scales differ")
    position_profile = fit_position_reliability_profile(
        cohort.discovery,
        cohort.discovery_counts,
        cohort.prefixes,
        subspace_rank=arguments.position_subspace_rank,
    )
    position_discovery, discovery_position_scale = (
        position_reliability_features(
            cohort.discovery,
            cohort.discovery_counts,
            position_profile["weights"],
        )
    )
    position_confirmation, confirmation_position_scale = (
        position_reliability_features(
            cohort.confirmation,
            cohort.confirmation_counts,
            position_profile["weights"],
        )
    )
    raw_position_discovery, raw_discovery_position_scale = (
        position_reliability_features(
            cohort.raw_discovery,
            cohort.discovery_counts,
            position_profile["weights"],
        )
    )
    raw_position_confirmation, raw_confirmation_position_scale = (
        position_reliability_features(
            cohort.raw_confirmation,
            cohort.confirmation_counts,
            position_profile["weights"],
        )
    )
    if (
        discovery_position_scale != raw_discovery_position_scale
        or confirmation_position_scale != raw_confirmation_position_scale
    ):
        raise AssertionError("raw and Adam-metric position scales differ")

    report = {
        "schema_version": SCHEMA,
        "analysis_regime": "method_calibration_on_frozen_96_prompt_pool",
        "inputs": {
            "sketch_dir": str(arguments.sketch_dir.resolve()),
            "sketch_files": list(cohort.source_files),
            "split_manifest": str(arguments.split.resolve()),
            "split_manifest_sha256": sha256_file(arguments.split),
            "label": arguments.label,
            "projection_identity": cohort.projection_identity,
            "training_contract": cohort.training_contract,
        },
        "cohort": {
            "prompts": len(cohort.prompts),
            "rollouts_per_half": int(cohort.discovery.shape[2]),
            "same_response_keys_for_all_feature_regimes": True,
            "prefixes": list(cohort.prefixes),
            "projection_seeds": list(cohort.projection_seeds),
        },
        "method": {
            "random_effects_model": "x_ir = mu + a_i + epsilon_ir",
            "adam_metric": (
                "applied in parameter space before CountSketch; fresh step-0 "
                "Adam is isotropic up to a common scalar"
            ),
            "nonlinear_optimizer_operations": (
                "PPO clipping is inactive at the fresh ratio; global norm "
                "clipping uses one factor from pooled raw-gradient sketches, "
                "preserving response-feature additivity"
            ),
            "confirmation_policy": (
                "confirmation responses never fit position/reliability weights"
            ),
        },
        "position_reliability_profile": position_profile,
        "feature_regimes": {
            "equal_prompt": regime_report(
                cohort.prompts,
                equal_discovery,
                equal_confirmation,
                raw_equal_discovery,
                raw_equal_confirmation,
                ranks=ranks,
                stable_rank_cap=arguments.stable_rank_cap,
                clip_norm=arguments.clip_norm,
                scaling={
                    "definition": "response-token-mean K1 gradient",
                    "prompt_measure": "uniform over prompts and rollouts",
                },
            ),
            "token_weighted": regime_report(
                cohort.prompts,
                token_discovery,
                token_confirmation,
                raw_token_discovery,
                raw_token_confirmation,
                ranks=ranks,
                stable_rank_cap=arguments.stable_rank_cap,
                clip_norm=arguments.clip_norm,
                scaling={
                    "definition": (
                        "x_ir=(masked_response_tokens/half_mean_tokens)*"
                        "response_token_mean_gradient"
                    ),
                    "discovery_mean_tokens": discovery_token_scale,
                    "confirmation_mean_tokens": confirmation_token_scale,
                    "batch_equivalence": (
                        "mean_ir(x_ir) equals VERL global token-mean gradient"
                    ),
                },
            ),
            "position_reliability_weighted": regime_report(
                cohort.prompts,
                position_discovery,
                position_confirmation,
                raw_position_discovery,
                raw_position_confirmation,
                ranks=ranks,
                stable_rank_cap=arguments.stable_rank_cap,
                clip_norm=arguments.clip_norm,
                scaling={
                    "definition": (
                        "weighted sum of disjoint masked-token block gradients "
                        "divided by half mean weighted-token count"
                    ),
                    "discovery_mean_weighted_tokens": discovery_position_scale,
                    "confirmation_mean_weighted_tokens": confirmation_position_scale,
                },
            ),
        },
    }
    atomic_json(arguments.output, report)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "prompts": len(cohort.prompts),
                "rollouts_per_half": int(cohort.discovery.shape[2]),
                "regimes": list(report["feature_regimes"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
