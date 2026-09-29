#!/usr/bin/env python3
"""Use reproducible gradient singular directions to guide prompt selection.

The selection unit is a prompt, not one rollout.  Each prompt's response
gradients are split into two disjoint halves.  The first half discovers an
SVD basis and selects a D-optimal set; the second half measures whether the
selected set still covers the population gradient subspace out of sample.

For a prompt x with top-r SVD coordinates z_x and an already selected set S,
the next-prompt utility is

    log(1 + z_x^T (ridge I + sum_{s in S} z_s z_s^T)^-1 z_x)

optionally divided by rollout cost.  This rewards a prompt only for a stable
direction that the selected prompts do not already cover.  It therefore
differs from top-k gradient alignment, which can select many redundant
prompts, and from ranking by gradient norm or response length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[3] / "outputs" / "opd_project_root"))
SCHEMA = "opd_svd_gradient_coverage_v1"


@dataclass(frozen=True)
class GradientHalves:
    prompts: tuple[str, ...]
    first: Any
    second: Any
    mean_response_tokens: np.ndarray
    projection_seeds: tuple[int, ...]
    samples_per_half: int
    projection_identity: Mapping[str, Any] | None = None
    confirmation_mean_response_tokens: np.ndarray | None = None


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError("integer lists must contain positive values")
    return tuple(dict.fromkeys(values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _projection_seeds(payload: Mapping[str, Any]) -> tuple[int, ...]:
    family = payload.get("projection_family")
    if isinstance(family, Mapping):
        seeds = tuple(int(seed) for seed in family.get("seeds", ()))
        if seeds:
            return seeds
    projection = payload.get("projection")
    if isinstance(projection, Mapping) and projection.get("seed") is not None:
        return (int(projection["seed"]),)
    return ()


def _projection_identity(
    payload: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    dimension: int,
) -> dict[str, Any]:
    """Canonical identity that prevents mixing layouts or projection families."""
    family = payload.get("projection_family")
    projection = payload.get("projection")
    layout = payload.get("parameter_layout")
    layout_sha256 = None
    if layout is not None:
        canonical_layout = json.dumps(
            layout,
            sort_keys=True,
            separators=(",", ":"),
        )
        layout_sha256 = hashlib.sha256(canonical_layout.encode()).hexdigest()
    if isinstance(family, Mapping):
        projection_ids = list(family.get("projection_ids", ()))
        specifications = family.get("projections")
        contract = family.get("contract")
    else:
        projection_ids = [
            value
            for value in (
                payload.get("projection_id"),
                projection.get("projection_id")
                if isinstance(projection, Mapping)
                else None,
            )
            if value is not None
        ]
        specifications = projection if isinstance(projection, Mapping) else None
        contract = (
            projection.get("contract")
            if isinstance(projection, Mapping)
            else None
        )
    return {
        "schema_version": payload.get("schema_version", payload.get("kind")),
        "contract": contract,
        "dimension": int(dimension),
        "seeds": [int(seed) for seed in seeds],
        "projection_ids": projection_ids,
        "specifications": specifications,
        "parameter_layout_sha256": layout_sha256,
        "objective_contract": payload.get("objective_contract"),
        "gradient_semantics": payload.get("gradient_semantics"),
        "loss_reduction": payload.get("loss_reduction"),
    }


def _select_prefix(
    tensor: Any,
    payload: Mapping[str, Any],
    *,
    prefix: str,
    path: Path,
) -> Any:
    """Return (responses, projection seeds, dimension) for v1 or v2 shards."""
    if tensor.ndim == 2:
        if prefix != "full":
            raise ValueError(f"{path}: v1 sketch only contains the full response")
        return tensor.unsqueeze(1)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim != 4:
        raise ValueError(f"{path}: unexpected sketch rank {tensor.ndim}")
    prefixes = tuple(str(value) for value in payload.get("prefixes", ()))
    if prefix not in prefixes:
        raise ValueError(f"{path}: prefix {prefix!r} is absent from {prefixes}")
    return tensor[:, :, prefixes.index(prefix), :]


def load_gradient_halves(
    sketch_dir: Path,
    *,
    label: str,
    normalization: str,
    prefix: str = "full",
    split_manifest: Path | None = None,
) -> GradientHalves:
    """Load shards and form disjoint per-prompt gradient means."""
    import torch

    records: list[dict[str, Any]] = []
    blocks: list[Any] = []
    seeds: tuple[int, ...] = ()
    projection_identity: dict[str, Any] | None = None
    for path in sorted(sketch_dir.glob("shard*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if label not in payload["sketches"]:
            raise ValueError(f"{path}: missing teacher label {label!r}")
        block = _select_prefix(
            payload["sketches"][label],
            payload,
            prefix=prefix,
            path=path,
        ).float()
        shard_seeds = _projection_seeds(payload)
        if not shard_seeds:
            shard_seeds = tuple(range(block.shape[1]))
        if len(shard_seeds) != block.shape[1]:
            raise ValueError(f"{path}: projection seed count does not match tensor")
        if seeds and shard_seeds != seeds:
            raise ValueError(f"{path}: projection family differs across shards")
        seeds = shard_seeds
        shard_identity = _projection_identity(
            payload,
            seeds=shard_seeds,
            dimension=block.shape[-1],
        )
        if (
            projection_identity is not None
            and shard_identity != projection_identity
        ):
            raise ValueError(
                f"{path}: projection or parameter-layout identity differs across shards"
            )
        projection_identity = shard_identity
        records.extend(dict(record) for record in payload["records"])
        blocks.append(block)
    if not blocks:
        raise FileNotFoundError(f"no shard*.pt under {sketch_dir}")

    gradients = torch.cat(blocks, dim=0)
    if len(records) != gradients.shape[0]:
        raise ValueError("record and gradient counts differ")
    keys = [
        (str(record["problem_id"]), int(record.get("sample_index", index)))
        for index, record in enumerate(records)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (problem_id, sample_index) records in sketches")
    position = {key: index for index, key in enumerate(keys)}

    by_prompt: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_prompt.setdefault(str(record["problem_id"]), []).append(index)
    for indices in by_prompt.values():
        indices.sort(key=lambda index: int(records[index].get("sample_index", index)))

    assignments: dict[str, tuple[list[int], list[int]]] = {}
    if split_manifest is not None:
        document = json.loads(split_manifest.read_text())
        raw_assignment = document.get("assignment")
        if not isinstance(raw_assignment, Mapping):
            raise ValueError(f"{split_manifest}: missing assignment")
        for prompt, assignment in raw_assignment.items():
            prompt = str(prompt)
            if prompt not in by_prompt:
                raise ValueError(f"{split_manifest}: prompt {prompt} absent from sketches")
            discovery_samples = [int(sample) for sample in assignment["discovery"]]
            confirmation_samples = [
                int(sample) for sample in assignment["confirmation"]
            ]
            if len(set(discovery_samples)) != len(discovery_samples):
                raise ValueError(f"{split_manifest}: duplicate discovery samples for {prompt}")
            if len(set(confirmation_samples)) != len(confirmation_samples):
                raise ValueError(
                    f"{split_manifest}: duplicate confirmation samples for {prompt}"
                )
            if set(discovery_samples) & set(confirmation_samples):
                raise ValueError(f"{split_manifest}: split halves overlap for {prompt}")
            first = [
                position[(prompt, int(sample))]
                for sample in discovery_samples
            ]
            second = [
                position[(prompt, int(sample))]
                for sample in confirmation_samples
            ]
            assignments[prompt] = (first, second)
    else:
        width = min(len(indices) for indices in by_prompt.values())
        half = width // 2
        if half < 1:
            raise ValueError("at least two responses per prompt are required")
        for prompt, indices in by_prompt.items():
            assignments[prompt] = (indices[:half], indices[half : 2 * half])

    prompts = tuple(
        sorted(
            assignments,
            key=lambda value: (
                (0, int(value)) if value.isdigit() else (1, value)
            ),
        )
    )
    widths = {
        (len(assignments[prompt][0]), len(assignments[prompt][1]))
        for prompt in prompts
    }
    if len(widths) != 1:
        raise ValueError(f"prompt split is not rectangular: {sorted(widths)}")
    first_width, second_width = next(iter(widths))
    if first_width != second_width:
        raise ValueError("the two prompt halves must have equal sample counts")

    response_tokens = torch.tensor(
        [float(record["response_tokens"]) for record in records],
        dtype=torch.float32,
    )

    def prompt_mean(indices: Sequence[int]) -> Any:
        block = gradients[list(indices)]
        tokens = response_tokens[list(indices), None, None]
        if normalization == "unit":
            block = block / block.norm(dim=2, keepdim=True).clamp(min=1e-12)
        elif normalization == "per_token":
            block = block / tokens.clamp(min=1)
        elif normalization == "token_sum":
            # Stored sketches are response-token means. Multiplication by response
            # length reconstructs the prompt's expected contribution to a batch
            # numerator under the trainer's token-mean reduction.
            block = block * tokens
        elif normalization != "none":
            raise ValueError(f"unknown normalization {normalization!r}")
        return block.mean(dim=0)

    first = torch.stack([prompt_mean(assignments[prompt][0]) for prompt in prompts])
    second = torch.stack([prompt_mean(assignments[prompt][1]) for prompt in prompts])
    # (prompts, seeds, dimension) -> (seeds, prompts, dimension)
    first = first.permute(1, 0, 2).contiguous()
    second = second.permute(1, 0, 2).contiguous()
    discovery_costs = np.array(
        [
            float(
                response_tokens[assignments[prompt][0]].mean()
            )
            for prompt in prompts
        ],
        dtype=np.float64,
    )
    confirmation_costs = np.array(
        [
            float(response_tokens[assignments[prompt][1]].mean())
            for prompt in prompts
        ],
        dtype=np.float64,
    )
    return GradientHalves(
        prompts=prompts,
        first=first,
        second=second,
        mean_response_tokens=discovery_costs,
        projection_seeds=seeds,
        samples_per_half=first_width,
        projection_identity=projection_identity,
        confirmation_mean_response_tokens=confirmation_costs,
    )


def prompt_grams(
    halves: GradientHalves,
    *,
    normalize_prompts: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average prompt-level Gram matrices over independent projections."""
    import torch

    first = halves.first.double()
    second = halves.second.double()
    if normalize_prompts:
        first = first / first.norm(dim=2, keepdim=True).clamp(min=1e-12)
        second = second / second.norm(dim=2, keepdim=True).clamp(min=1e-12)
    first_gram = torch.einsum("spd,sqd->spq", first, first).mean(dim=0)
    second_gram = torch.einsum("spd,sqd->spq", second, second).mean(dim=0)
    cross_gram = torch.einsum("spd,sqd->spq", first, second).mean(dim=0)
    return (
        first_gram.cpu().numpy(),
        second_gram.cpu().numpy(),
        cross_gram.cpu().numpy(),
    )


def center_gram(gram: np.ndarray) -> np.ndarray:
    row = gram.mean(axis=1, keepdims=True)
    return gram - row - row.T + gram.mean()


def eigensystem(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    return values, vectors[:, order]


def spectrum_summary(values: np.ndarray) -> dict[str, Any]:
    positive = values[values > max(float(values[0]) * 1e-12, 1e-15)]
    trace = float(positive.sum())
    if trace <= 0:
        return {"rank": 0}
    fractions = positive / trace
    cumulative = np.cumsum(fractions)
    return {
        "rank": int(len(positive)),
        "participation_ratio": float(1.0 / np.square(fractions).sum()),
        "stable_rank": float(1.0 / fractions[0]),
        "top_eigenvalue_share": float(fractions[0]),
        "directions_for_50pct": int(np.searchsorted(cumulative, 0.50) + 1),
        "directions_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "top_eigenvalue_shares": [float(value) for value in fractions[:16]],
    }


def top_coordinates(
    values: np.ndarray,
    vectors: np.ndarray,
    rank: int,
) -> np.ndarray:
    rank = min(rank, int(np.count_nonzero(values > max(values[0] * 1e-12, 1e-15))))
    return vectors[:, :rank] * np.sqrt(values[:rank])[None, :]


def subspace_overlap(
    first_gram: np.ndarray,
    second_gram: np.ndarray,
    cross_gram: np.ndarray,
    rank: int,
) -> dict[str, Any]:
    """Canonical cosines between top-r feature-space singular subspaces."""
    first_values, first_vectors = eigensystem(first_gram)
    second_values, second_vectors = eigensystem(second_gram)
    rank = min(
        rank,
        int(np.count_nonzero(first_values > max(first_values[0] * 1e-12, 1e-15))),
        int(np.count_nonzero(second_values > max(second_values[0] * 1e-12, 1e-15))),
    )
    left = first_vectors[:, :rank] / np.sqrt(first_values[:rank])[None, :]
    right = second_vectors[:, :rank] / np.sqrt(second_values[:rank])[None, :]
    bridge = left.T @ cross_gram @ right
    canonical = np.clip(np.linalg.svd(bridge, compute_uv=False), 0.0, 1.0)
    return {
        "rank": rank,
        "mean_canonical_cosine": float(canonical.mean()),
        "mean_squared_canonical_cosine": float(np.square(canonical).mean()),
        "minimum_canonical_cosine": float(canonical.min()),
        "canonical_cosines": [float(value) for value in canonical],
    }


def greedy_d_optimal(
    coordinates: np.ndarray,
    support: int,
    *,
    ridge: float,
    costs: np.ndarray | None = None,
    cost_power: float = 0.0,
) -> list[int]:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if not 1 <= support <= len(coordinates):
        raise ValueError("support is outside the candidate count")
    dimension = coordinates.shape[1]
    inverse = np.eye(dimension, dtype=np.float64) / ridge
    selected: list[int] = []
    available = np.ones(len(coordinates), dtype=bool)
    if costs is None:
        relative_cost = np.ones(len(coordinates), dtype=np.float64)
    else:
        relative_cost = np.asarray(costs, dtype=np.float64) / np.median(costs)
        if np.any(relative_cost <= 0):
            raise ValueError("costs must be positive")

    for _ in range(support):
        quadratic = np.einsum(
            "ni,ij,nj->n",
            coordinates,
            inverse,
            coordinates,
            optimize=True,
        )
        gain = np.log1p(np.maximum(quadratic, 0.0))
        score = gain / np.power(relative_cost, cost_power)
        score[~available] = -np.inf
        chosen = int(np.argmax(score))
        vector = coordinates[chosen]
        transformed = inverse @ vector
        denominator = 1.0 + float(vector @ transformed)
        inverse -= np.outer(transformed, transformed) / denominator
        inverse = (inverse + inverse.T) / 2
        selected.append(chosen)
        available[chosen] = False
    return selected


def set_metrics(
    indices: Sequence[int],
    gram: np.ndarray,
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    ridge: float,
) -> dict[str, float]:
    selected = np.asarray(indices, dtype=np.int64)
    block = gram[np.ix_(selected, selected)]
    count = len(selected)
    if count > 1:
        within = float((block.sum() - np.trace(block)) / (count * (count - 1)))
    else:
        within = 0.0

    pool = np.full(len(gram), 1.0 / len(gram))
    arm = np.zeros(len(gram), dtype=np.float64)
    arm[selected] = 1.0 / count
    numerator = float(arm @ gram @ pool)
    denominator = math.sqrt(max(float(arm @ gram @ arm) * float(pool @ gram @ pool), 1e-24))

    selected_coordinates = coordinates[selected]
    information = selected_coordinates.T @ selected_coordinates
    soft_projector = information @ np.linalg.inv(
        information + ridge * np.eye(information.shape[0])
    )
    diagonal_projector = np.diag(soft_projector)
    retained = float(values[: coordinates.shape[1]] @ diagonal_projector)
    total = float(values[: coordinates.shape[1]].sum())

    sign, logdet = np.linalg.slogdet(
        np.eye(count) + selected_coordinates @ selected_coordinates.T / ridge
    )
    if sign <= 0:
        raise ValueError("D-optimal information matrix is not positive definite")
    block_values = np.maximum(np.linalg.eigvalsh((block + block.T) / 2), 0.0)
    block_trace = float(block_values.sum())
    fractions = block_values / block_trace if block_trace else block_values
    return {
        "mean_gradient_cosine_to_pool": numerator / denominator,
        "within_set_cosine": within,
        "top_r_subspace_energy_coverage": retained / total if total else float("nan"),
        "d_opt_logdet": float(logdet),
        "set_participation_ratio": (
            float(1.0 / np.square(fractions).sum()) if block_trace else float("nan")
        ),
    }


def random_reference(
    gram: np.ndarray,
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    support: int,
    ridge: float,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    generator = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(draws):
        indices = generator.choice(len(gram), size=support, replace=False)
        for name, value in set_metrics(
            indices,
            gram,
            values,
            coordinates,
            ridge=ridge,
        ).items():
            collected.setdefault(name, []).append(float(value))
    return {
        name: {
            "mean": float(np.mean(observations)),
            "std": float(np.std(observations, ddof=1)),
            "p05": float(np.percentile(observations, 5)),
            "p95": float(np.percentile(observations, 95)),
        }
        for name, observations in collected.items()
    }


def attach_random_comparison(
    metrics: dict[str, float],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    output: dict[str, Any] = dict(metrics)
    output["vs_random"] = {}
    for name, value in metrics.items():
        baseline = reference[name]
        std = float(baseline["std"])
        output["vs_random"][name] = {
            "random_mean": float(baseline["mean"]),
            "random_p05": float(baseline["p05"]),
            "random_p95": float(baseline["p95"]),
            "z_score": (value - float(baseline["mean"])) / std if std else float("nan"),
        }
    return output


def load_retrospective_arms(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    document = json.loads(path.read_text())
    arms: dict[str, list[str]] = {}
    for name, entry in document.get("arms", {}).items():
        values = entry["problem_ids"] if isinstance(entry, Mapping) else entry
        arms[str(name)] = [str(value) for value in values]
    return arms


def observed_math(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    document = json.loads(path.read_text())
    return {
        str(name): float(entry["mean_at_16"])
        for name, entry in document.get("math_arm_summary", {}).items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sketch-dir",
        type=Path,
        default=ROOT / "data/opd/gradient_spectrum_v1/sketches_384",
    )
    parser.add_argument("--label", default="nonhom_30b")
    parser.add_argument(
        "--normalization",
        choices=("unit", "per_token", "token_sum", "none"),
        default="unit",
    )
    parser.add_argument("--prefix", default="full")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--ranks", default="4,8,16,32,64")
    parser.add_argument("--supports", default="4,8,16")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cost-power", type=float, default=1.0)
    parser.add_argument("--random-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--retrospective-arms", type=Path)
    parser.add_argument("--observed-results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/analysis/svd_gradient_coverage_v1.json",
    )
    arguments = parser.parse_args()
    ranks = parse_int_list(arguments.ranks)
    supports = parse_int_list(arguments.supports)
    if arguments.cost_power < 0:
        raise SystemExit("--cost-power must be non-negative")
    if arguments.random_draws < 1:
        raise SystemExit("--random-draws must be positive")

    halves = load_gradient_halves(
        arguments.sketch_dir,
        label=arguments.label,
        normalization=arguments.normalization,
        prefix=arguments.prefix,
        split_manifest=arguments.split_manifest,
    )
    first_gram, second_gram, cross_gram = prompt_grams(halves)
    first_values, first_vectors = eigensystem(first_gram)
    second_values, second_vectors = eigensystem(second_gram)
    retrospective = load_retrospective_arms(arguments.retrospective_arms)
    observed = observed_math(arguments.observed_results)
    position = {prompt: index for index, prompt in enumerate(halves.prompts)}

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "method": {
            "selection": "greedy_D_optimal_on_discovery_top_r_SVD_coordinates",
            "marginal_gain": (
                "log(1 + z_x^T (ridge I + sum_s z_s z_s^T)^-1 z_x)"
            ),
            "cost_aware_gain": "marginal_gain / (prompt_tokens / median_tokens)^cost_power",
            "validation": "all reported selected-set metrics use disjoint confirmation responses",
        },
        "inputs": {
            "sketch_dir": str(arguments.sketch_dir.resolve()),
            "label": arguments.label,
            "normalization": arguments.normalization,
            "prefix": arguments.prefix,
            "split_manifest": (
                str(arguments.split_manifest.resolve())
                if arguments.split_manifest is not None
                else None
            ),
            "split_manifest_sha256": (
                sha256_file(arguments.split_manifest)
                if arguments.split_manifest is not None
                else None
            ),
        },
        "cohort": {
            "prompts": len(halves.prompts),
            "samples_per_half": halves.samples_per_half,
            "projection_seeds": list(halves.projection_seeds),
            "projection_identity": halves.projection_identity,
            "response_tokens": {
                "mean": float(halves.mean_response_tokens.mean()),
                "median": float(np.median(halves.mean_response_tokens)),
                "minimum": float(halves.mean_response_tokens.min()),
                "maximum": float(halves.mean_response_tokens.max()),
            },
        },
        "spectrum": {
            "discovery_uncentered": spectrum_summary(first_values),
            "confirmation_uncentered": spectrum_summary(second_values),
            "discovery_centered": spectrum_summary(eigensystem(center_gram(first_gram))[0]),
            "confirmation_centered": spectrum_summary(eigensystem(center_gram(second_gram))[0]),
        },
        "ranks": {},
    }

    for rank in ranks:
        effective_rank = min(rank, len(first_values), len(second_values))
        first_coordinates = top_coordinates(first_values, first_vectors, effective_rank)
        second_coordinates = top_coordinates(second_values, second_vectors, effective_rank)
        rank_report: dict[str, Any] = {
            "subspace_reproducibility": subspace_overlap(
                first_gram,
                second_gram,
                cross_gram,
                effective_rank,
            ),
            "supports": {},
        }
        for support in supports:
            if support > len(halves.prompts):
                continue
            reference = random_reference(
                second_gram,
                second_values,
                second_coordinates,
                support=support,
                ridge=arguments.ridge,
                draws=arguments.random_draws,
                seed=arguments.seed + 1009 * effective_rank + support,
            )
            support_report: dict[str, Any] = {"random_confirmation": reference}
            for name, cost_power in (
                ("svd_d_optimal", 0.0),
                ("svd_d_optimal_cost_aware", arguments.cost_power),
            ):
                selected = greedy_d_optimal(
                    first_coordinates,
                    support,
                    ridge=arguments.ridge,
                    costs=halves.mean_response_tokens,
                    cost_power=cost_power,
                )
                confirmation = set_metrics(
                    selected,
                    second_gram,
                    second_values,
                    second_coordinates,
                    ridge=arguments.ridge,
                )
                oracle = greedy_d_optimal(
                    second_coordinates,
                    support,
                    ridge=arguments.ridge,
                    costs=halves.mean_response_tokens,
                    cost_power=cost_power,
                )
                support_report[name] = {
                    "problem_ids": [halves.prompts[index] for index in selected],
                    "mean_response_tokens": float(
                        halves.mean_response_tokens[selected].mean()
                    ),
                    "confirmation": attach_random_comparison(confirmation, reference),
                    "selection_jaccard_with_confirmation_oracle": (
                        len(set(selected) & set(oracle)) / len(set(selected) | set(oracle))
                    ),
                }
            if retrospective:
                support_report["retrospective_arms"] = {}
                for name, prompt_ids in retrospective.items():
                    if len(prompt_ids) != support or any(
                        prompt not in position for prompt in prompt_ids
                    ):
                        continue
                    indices = [position[prompt] for prompt in prompt_ids]
                    entry = {
                        "problem_ids": prompt_ids,
                        "mean_response_tokens": float(
                            halves.mean_response_tokens[indices].mean()
                        ),
                        "confirmation": attach_random_comparison(
                            set_metrics(
                                indices,
                                second_gram,
                                second_values,
                                second_coordinates,
                                ridge=arguments.ridge,
                            ),
                            reference,
                        ),
                    }
                    if name in observed:
                        entry["observed_id_math_mean_at_16"] = observed[name]
                    support_report["retrospective_arms"][name] = entry
            rank_report["supports"][str(support)] = support_report
        report["ranks"][str(effective_rank)] = rank_report

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "prompts": len(halves.prompts),
                "samples_per_half": halves.samples_per_half,
                "projection_seeds": halves.projection_seeds,
                "spectrum": report["spectrum"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
