#!/usr/bin/env python3
"""Shared loading and Gram-matrix machinery for the coverage-selection arms.

Two things live here because both the discovery search and the confirmation gate need them and
must not disagree about either.

**The v2 tensor is not a v1 tensor.** ``sketch_response_gradients.py`` now writes
``(responses, projection_seeds, prefixes, dimension)`` under schema
``opd_response_prefix_gradient_sketch_v2``, where v1 wrote ``(responses, dimension)``.  Running it
with ``--prefix-lengths full`` reduces the prefix axis to one entry that is the whole-response
gradient v1 measured, but the seed axis stays, and feeding the 4-D tensor to a v1 analyser would
silently reinterpret projection seeds as responses.  ``load_sketches`` collapses the prefix axis and
keeps the seeds separate and explicit.

Projection seeds are averaged over at the level of *cosines*, which removes CountSketch hash noise
from the similarity estimate.  They are not observations: three projections of one response are one
response, and nothing here ever treats the seed axis as a sample axis.

**Everything reduces to a Gram matrix.**  Selection and both bootstraps only ever need inner
products between means of subsets of unit-normalised response gradients, and the inner product of
two such means is the mean of the corresponding block of the Gram matrix.  So the 32,768-dimensional
vectors are touched exactly once, and the tens of thousands of candidate quartets and their
bootstrap resamples then cost small-matrix arithmetic instead of repeated passes over the sketches.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[3] / "outputs" / "opd_project_root"))

COVARIATES = ("response_tokens", "reward", "k1_mean", "teacher_logprob_mean")


@dataclass(frozen=True)
class Cohort:
    """Unit-normalised response gradients for one half, grouped by prompt.

    ``gram`` is ``(seeds, responses, responses)``.  ``rows[p]`` lists the row indices of prompt
    ``p``'s responses, in the order the split manifest gives them, so the A/B sub-split used for
    debiasing is reproducible.
    """

    prompts: list[str]
    rows: dict[str, list[int]]
    gram: np.ndarray
    covariates: dict[str, np.ndarray]
    seeds: tuple[int, ...]


def load_split(path: Path) -> dict[str, dict[str, list[int]]]:
    return json.loads(path.read_text())["assignment"]


def load_sketches(
    sketch_dir: Path, label: str
) -> tuple[list[dict], np.ndarray, tuple[int, ...]]:
    """Return records and a ``(responses, seeds, dimension)`` array of gradients.

    Accepts both schemas so the bridge check can read v1 and v2 side by side.  v1 has no seed axis
    and gets one of width 1.
    """
    records: list[dict] = []
    blocks: list[torch.Tensor] = []
    seeds: tuple[int, ...] = ()
    for path in sorted(sketch_dir.glob("shard*.pt")):
        payload = torch.load(path, weights_only=False)
        records.extend(payload["records"])
        tensor = payload["sketches"][label]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
            shard_seeds = (int(payload["projection"]["seed"]),)
        elif tensor.ndim == 4:
            # (responses, seeds, prefixes, dimension); --prefix-lengths full leaves one prefix.
            if tensor.shape[2] != 1:
                raise SystemExit(
                    f"{path} has {tensor.shape[2]} prefixes; this analysis needs --prefix-lengths full"
                )
            tensor = tensor[:, :, 0, :]
            shard_seeds = tuple(int(seed) for seed in payload["projection_family"]["seeds"])
        else:
            raise SystemExit(f"{path}: unexpected sketch rank {tensor.ndim}")
        if seeds and shard_seeds != seeds:
            raise SystemExit(f"{path}: projection seeds {shard_seeds} != {seeds}")
        seeds = shard_seeds
        blocks.append(tensor)
    if not blocks:
        raise SystemExit(f"no shard*.pt under {sketch_dir}")
    return records, torch.cat(blocks).double().numpy(), seeds


def teacher_summary(record: dict, label: str) -> dict:
    """Per-response teacher statistics, from either schema.

    v1 flattened them onto the record as ``{label}_k1_mean``; v2 nests them per teacher and per
    prefix.  For ``--prefix-lengths full`` the two carry bit-identical values, so downstream code
    should not have to know which it is reading.
    """
    if "teachers" in record:
        return record["teachers"][label]["prefixes"]["full"]
    return {
        "k1_mean": record[f"{label}_k1_mean"],
        "teacher_logprob_mean": record[f"{label}_teacher_logprob_mean"],
    }


def build_cohort(
    sketch_dir: Path,
    split_path: Path | None,
    label: str,
    half: str,
) -> Cohort:
    """Gram matrix and per-prompt covariates for one side of the frozen split.

    ``split_path=None`` takes every response a prompt has, ordered by sample index.  That is for a
    batch collected purely to confirm, where there is nothing to hold out because the selection
    already happened somewhere else.
    """
    records, gradients, seeds = load_sketches(sketch_dir, label)
    if split_path is None:
        grouped: dict[str, list[int]] = {}
        for record in records:
            grouped.setdefault(record["problem_id"], []).append(int(record["sample_index"]))
        assignment = {
            prompt: {"discovery": sorted(samples), "confirmation": []}
            for prompt, samples in grouped.items()
        }
        half = "discovery"
    else:
        assignment = load_split(split_path)

    keep: list[int] = []
    rows: dict[str, list[int]] = {}
    per_prompt: dict[str, dict[str, list[float]]] = {}
    position = {
        (record["problem_id"], int(record["sample_index"])): index
        for index, record in enumerate(records)
    }
    for prompt in sorted(assignment, key=int):
        if half == "all":
            # Only legitimate once the split has served its purpose and a fresh batch is doing the
            # confirming instead; using every response just maximises the estimate's reliability.
            wanted = sorted(
                assignment[prompt]["discovery"] + assignment[prompt]["confirmation"]
            )
        else:
            wanted = assignment[prompt][half]
        indices = []
        for sample in wanted:
            key = (prompt, sample)
            if key not in position:
                raise SystemExit(f"prompt {prompt} sample {sample} absent from {sketch_dir}")
            indices.append(position[key])
        rows[prompt] = list(range(len(keep), len(keep) + len(indices)))
        keep.extend(indices)
        fields = per_prompt.setdefault(prompt, {name: [] for name in COVARIATES})
        for index in indices:
            record = records[index]
            fields["response_tokens"].append(float(record["response_tokens"]))
            fields["reward"].append(float(record["reward"]))
            summary = teacher_summary(record, label)
            fields["k1_mean"].append(float(summary["k1_mean"]))
            fields["teacher_logprob_mean"].append(float(summary["teacher_logprob_mean"]))

    block = gradients[keep]
    norms = np.linalg.norm(block, axis=2, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    block = block / norms
    # (seeds, responses, responses); the only time the 32,768-dimensional vectors are touched.
    gram = np.einsum("nsd,msd->snm", block, block)

    prompts = sorted(rows, key=int)
    covariates = {
        name: np.array(
            [float(np.mean(per_prompt[prompt][name])) for prompt in prompts],
            dtype=np.float64,
        )
        for name in COVARIATES
    }
    return Cohort(prompts, rows, gram, covariates, seeds)


def half_mean_gram(cohort: Cohort, split: int = 4) -> np.ndarray:
    """Inner products between every prompt's two half-means, ``(seeds, 2P, 2P)``.

    Row ``2p`` is prompt ``p``'s first sub-half and ``2p+1`` its second.  Debiasing needs the two
    sub-halves to share no response, which the fixed row order guarantees.
    """
    count = len(cohort.prompts)
    responses = cohort.gram.shape[1]
    weights = np.zeros((2 * count, responses), dtype=np.float64)
    for index, prompt in enumerate(cohort.prompts):
        rows = cohort.rows[prompt]
        for side, chunk in enumerate((rows[:split], rows[split:])):
            weights[2 * index + side, chunk] = 1.0 / len(chunk)
    return np.einsum("ir,srm,jm->sij", weights, cohort.gram, weights)


def cosines_from_gram(inner: np.ndarray) -> np.ndarray:
    """Cosine matrix from a matrix of inner products, per seed."""
    diagonal = np.sqrt(np.maximum(np.einsum("sii->si", inner), 1e-24))
    return inner / diagonal[:, :, None] / diagonal[:, None, :]


def debiased_similarity(cohort: Cohort, split: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Cross-half prompt similarity and per-prompt alignment to the pool mean.

    Both are averaged over projection seeds.  Both take their two operands from disjoint responses,
    so neither is inflated by the shared sampling noise that makes a same-half cosine optimistic.
    """
    inner = half_mean_gram(cohort, split)
    cosine = cosines_from_gram(inner)
    first, second = slice(0, None, 2), slice(1, None, 2)
    across = (cosine[:, first, second] + cosine[:, second, first]) / 2
    similarity = ((across + np.swapaxes(across, 1, 2)) / 2).mean(axis=0)
    np.fill_diagonal(similarity, 1.0)

    count = len(cohort.prompts)
    # Pool mean is the mean of prompt half-means, so every prompt counts once regardless of how
    # long its responses are.
    pooling = np.full(count, 1.0 / count)
    alignment = np.zeros(count, dtype=np.float64)
    for seed in range(inner.shape[0]):
        block = inner[seed]
        for source, target in ((first, second), (second, first)):
            rows = block[source, :][:, target]
            pool_norm = np.sqrt(max(pooling @ (block[target, :][:, target]) @ pooling, 1e-24))
            own = np.sqrt(np.maximum(np.diagonal(block[source, :][:, source]), 1e-24))
            alignment += (rows @ pooling) / own / pool_norm
    alignment /= 2 * inner.shape[0]
    return similarity, alignment
