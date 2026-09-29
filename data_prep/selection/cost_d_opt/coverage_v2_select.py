#!/usr/bin/env python3
"""Choose the two matched quartets on the discovery half, and freeze them.

Reads only the discovery responses.  Writes the arms, the tolerances that were achieved and the
discovery-side statistics, so that the confirmation gate has something fixed to test.

The search is exhaustive rather than sampled: C(96,4) is 3,321,960, which fits in memory, so there
is no question of whether more draws would have found a better pair.

Selection maximises the 95% bootstrap **lower bound** of the separation, not the separation itself.
Picking the largest point estimate out of millions of candidates guarantees an inflated one -- the
winner is whichever pair's sampling noise happened to help most.  Ranking by a lower bound prefers a
separation that survives its own noise, which is the property that has to hold again on the
confirmation half.

Bootstrap resamples responses within each sub-half, independently for the two sub-halves, so the
resampled halves still share no response and the debiased similarity stays debiased.  Since a
sub-half holds four responses, resampling it with replacement has only C(7,3) = 35 outcomes, and all
35 x 35 bilinear forms are tabulated up front; a draw is then a table lookup.
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from math import factorial
from pathlib import Path

import numpy as np

from coverage_v2_common import (
    COVARIATES,
    ROOT,
    build_cohort,
    debiased_similarity,
)

PAIR_INDICES = tuple(itertools.combinations(range(4), 2))
MATCHED = ("alignment", *COVARIATES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    base = ROOT / "data/opd/coverage_selection_v2"
    parser.add_argument("--sketch-dir", type=Path, default=base / "sketches")
    parser.add_argument("--split", type=Path, default=base / "split_manifest.json")
    parser.add_argument("--label", default="nonhom_30b")
    parser.add_argument("--support", type=int, default=4)
    parser.add_argument("--alignment-tolerance", type=float, default=0.005)
    parser.add_argument("--length-tolerance", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--minimum-pairs", type=int, default=200)
    parser.add_argument("--extremes", type=int, default=40_000)
    parser.add_argument("--shortlist", type=int, default=200)
    parser.add_argument("--screen-draws", type=int, default=2_000)
    parser.add_argument("--final-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=base / "frozen_arms.json")
    return parser.parse_args()


def enumerate_quartets(count: int, support: int) -> np.ndarray:
    total = factorial(count) // (factorial(support) * factorial(count - support))
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(count), support)),
        dtype=np.int16,
        count=total * support,
    )
    return flat.reshape(total, support)


def within_similarity(quartets: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    total = np.zeros(len(quartets), dtype=np.float64)
    for left, right in PAIR_INDICES:
        total += similarity[quartets[:, left], quartets[:, right]]
    return total / len(PAIR_INDICES)


def multiplicity_table(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Every way ``size`` draws with replacement can land, with multinomial probabilities."""
    patterns = []
    for combination in itertools.combinations_with_replacement(range(size), size):
        counts = np.bincount(combination, minlength=size)
        patterns.append(counts)
    counts = np.array(patterns, dtype=np.float64)
    weights = np.array(
        [
            factorial(size)
            / np.prod([factorial(int(value)) for value in row])
            / size**size
            for row in counts
        ]
    )
    return counts / size, weights


class BootstrapTable:
    """Tabulated inner products between resampled sub-half means for one candidate pair."""

    def __init__(self, cohort, prompts: list[int], split: int, generator) -> None:
        self.generator = generator
        rows: list[list[int]] = []
        for index in prompts:
            name = cohort.prompts[index]
            member = cohort.rows[name]
            rows.append(member[:split])
            rows.append(member[split:])
        flat = [row for group in rows for row in group]
        block = cohort.gram[:, flat, :][:, :, flat]
        self.halves = len(rows)
        weights, self.probabilities = multiplicity_table(split)
        self.patterns = len(weights)
        # inner[a, b, m, n] = <mean of sub-half a under pattern m, mean of sub-half b under n>,
        # averaged over projection seeds after the cosine is formed, so keep the seed axis here.
        seeds = block.shape[0]
        table = np.zeros((seeds, self.halves, self.halves, self.patterns, self.patterns))
        for a in range(self.halves):
            for b in range(self.halves):
                sub = block[:, a * split : (a + 1) * split, b * split : (b + 1) * split]
                table[:, a, b] = np.einsum("mi,sij,nj->smn", weights, sub, weights)
        self.table = table
        self.norm = np.sqrt(np.maximum(np.einsum("saamm->sam", table), 1e-24))

    def draw(self, draws: int) -> np.ndarray:
        return self.generator.choice(
            self.patterns, size=(draws, self.halves), p=self.probabilities
        )

    def similarity(self, picks: np.ndarray, members: list[int]) -> np.ndarray:
        """Mean cross-half within-set similarity for one quartet, per bootstrap draw."""
        draws = picks.shape[0]
        total = np.zeros(draws)
        for left, right in itertools.combinations(members, 2):
            value = np.zeros(draws)
            for source, target in ((0, 1), (1, 0)):
                a, b = 2 * left + source, 2 * right + target
                m, n = picks[:, a], picks[:, b]
                inner = self.table[:, a, b, m, n]
                value += (
                    inner / self.norm[:, a, m] / self.norm[:, b, n]
                ).mean(axis=0)
            total += value / 2
        return total / len(list(itertools.combinations(members, 2)))


MAX_TABULATED_SPLIT = 5


class DirectBootstrap:
    """Bootstrap by resampling outright, for sub-halves too wide to tabulate.

    ``BootstrapTable`` enumerates every way a sub-half can be resampled, which is C(2k-1, k-1)
    outcomes: 35 at k=4, but 6,435 at k=8, and the table is quadratic in that.  Above a small k the
    resampling has to be done directly.  The sub-Gram is still tiny, so the cost is one small
    ``W G W'`` per chunk of draws rather than anything touching the sketches.
    """

    def __init__(self, cohort, prompts: list[int], split: int, generator) -> None:
        self.generator = generator
        self.split = split
        rows: list[list[int]] = []
        for index in prompts:
            member = cohort.rows[cohort.prompts[index]]
            rows.append(member[:split])
            rows.append(member[split:])
        flat = [row for group in rows for row in group]
        self.halves = len(rows)
        # Blocked as (seeds, halves, split, halves, split).  A half-mean only draws on its own eight
        # responses, so contracting against the blocks costs a twentieth of padding the weights out
        # to the full width and letting most of the multiplications hit zeros.
        self.gram = (
            cohort.gram[:, flat, :][:, :, flat]
            .reshape(-1, self.halves, split, self.halves, split)
            .copy()
        )

    def gaps(self, draws: int, chunk: int = 2_000) -> np.ndarray:
        pairs_low = list(itertools.combinations(range(4), 2))
        pairs_high = list(itertools.combinations(range(4, 8), 2))
        out: list[np.ndarray] = []
        for start in range(0, draws, chunk):
            size = min(chunk, draws - start)
            counts = self.generator.multinomial(
                self.split, np.full(self.split, 1.0 / self.split), size=(size, self.halves)
            ).astype(np.float64) / self.split
            inner = np.einsum(
                "bip,sipjq,bjq->bsij", counts, self.gram, counts, optimize=True
            )
            norm = np.sqrt(np.maximum(np.einsum("bsii->bsi", inner), 1e-24))
            cosine = (inner / norm[:, :, :, None] / norm[:, :, None, :]).mean(axis=1)
            values = []
            for pairs in (pairs_low, pairs_high):
                total = np.zeros(size)
                for left, right in pairs:
                    total += (
                        cosine[:, 2 * left, 2 * right + 1] + cosine[:, 2 * left + 1, 2 * right]
                    ) / 2
                values.append(total / len(pairs))
            out.append(values[1] - values[0])
        return np.concatenate(out)


def separation_bounds(
    cohort, low: np.ndarray, high: np.ndarray, split: int, draws: int, generator
) -> tuple[float, float, float]:
    prompts = [int(value) for value in low] + [int(value) for value in high]
    if split <= MAX_TABULATED_SPLIT:
        table = BootstrapTable(cohort, prompts, split, generator)
        picks = table.draw(draws)
        gap = table.similarity(picks, [4, 5, 6, 7]) - table.similarity(picks, [0, 1, 2, 3])
    else:
        gap = DirectBootstrap(cohort, prompts, split, generator).gaps(draws)
    return float(np.percentile(gap, 5)), float(gap.mean()), float(np.percentile(gap, 95))


def main() -> int:
    arguments = parse_args()
    if arguments.output.exists():
        raise SystemExit(f"{arguments.output} exists; arms are frozen and must not be re-selected")

    cohort = build_cohort(
        arguments.sketch_dir, arguments.split, arguments.label, half="discovery"
    )
    split = len(cohort.rows[cohort.prompts[0]]) // 2
    similarity, alignment = debiased_similarity(cohort, split)
    count = len(cohort.prompts)
    print(f"[cohort] {count} prompts x {2 * split} discovery responses, seeds={cohort.seeds}")

    features = {"alignment": alignment, **cohort.covariates}
    quartets = enumerate_quartets(count, arguments.support)
    s_bar = within_similarity(quartets, similarity)
    means = {name: values[quartets].mean(axis=1) for name, values in features.items()}
    print(f"[enumerate] {len(quartets):,} quartets, s_bar {s_bar.min():.4f}..{s_bar.max():.4f}")

    # Neither arm may be a freak set in its own right, independent of how well the two match.
    typical = np.ones(len(quartets), dtype=bool)
    for name, values in means.items():
        low, high = np.percentile(values, (10, 90))
        typical &= (values >= low) & (values <= high)
    print(f"[typicality] {typical.sum():,} quartets inside the 10-90% band on all five covariates")

    keep = np.flatnonzero(typical)
    order = keep[np.argsort(s_bar[keep])]
    take = min(arguments.extremes, len(order) // 2)
    low_set, high_set = order[:take], order[-take:]

    # Disjointness of two quartets is a bitwise test once each is a 96-bit membership mask.
    bits = np.zeros((len(quartets), 2), dtype=np.uint64)
    for column in range(arguments.support):
        index = quartets[:, column].astype(np.int64)
        shifted = np.left_shift(np.uint64(1), (index % 64).astype(np.uint64))
        lower = index < 64
        bits[:, 0] |= np.where(lower, shifted, np.uint64(0)).astype(np.uint64)
        bits[:, 1] |= np.where(~lower, shifted, np.uint64(0)).astype(np.uint64)

    scales = {name: float(np.std(values)) for name, values in features.items()}
    tau = arguments.tau
    while True:
        pairs = admissible_pairs(
            low_set, high_set, means, bits, s_bar, arguments, scales, tau
        )
        if len(pairs) >= arguments.minimum_pairs or tau > 1.0:
            break
        tau = round(tau + arguments.tau_step, 4)
    if not pairs:
        raise SystemExit("no admissible matched pair at any tolerance; widen the candidate pool")
    print(f"[matching] tau={tau:.2f} gives {len(pairs):,} admissible disjoint pairs")

    pairs.sort(key=lambda item: -item[2])
    shortlist = pairs[: arguments.shortlist]
    print(
        f"[shortlist] {len(shortlist)} pairs, point Delta s_bar "
        f"{shortlist[-1][2]:.4f}..{shortlist[0][2]:.4f}"
    )

    generator = np.random.default_rng(arguments.seed)
    screened = []
    for rank, (low_index, high_index, gap) in enumerate(shortlist):
        lower, mean, _ = separation_bounds(
            cohort,
            quartets[low_index],
            quartets[high_index],
            split,
            arguments.screen_draws,
            generator,
        )
        screened.append((lower, mean, gap, low_index, high_index))
        if rank % 50 == 0:
            print(f"  screened {rank + 1}/{len(shortlist)}")
    screened.sort(reverse=True)

    finals = []
    for lower, _, gap, low_index, high_index in screened[:10]:
        lower, mean, upper = separation_bounds(
            cohort,
            quartets[low_index],
            quartets[high_index],
            split,
            arguments.final_draws,
            generator,
        )
        finals.append((lower, mean, upper, gap, low_index, high_index))
    finals.sort(reverse=True)
    lower, mean, upper, point, low_index, high_index = finals[0]

    diverse = [cohort.prompts[index] for index in quartets[low_index]]
    redundant = [cohort.prompts[index] for index in quartets[high_index]]
    report = {
        "schema_version": "opd_coverage_frozen_arms_v1",
        "frozen_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "half": "discovery",
        "support": arguments.support,
        "arms": {"grad_diverse": diverse, "grad_redundant": redundant},
        "discovery": {
            "s_bar_diverse": round(float(s_bar[low_index]), 4),
            "s_bar_redundant": round(float(s_bar[high_index]), 4),
            "delta_s_bar": round(float(point), 4),
            "bootstrap_mean": round(mean, 4),
            "bootstrap_ci95": [round(lower, 4), round(upper, 4)],
        },
        "matching": {
            "tau": tau,
            "alignment_tolerance": arguments.alignment_tolerance,
            "length_tolerance": arguments.length_tolerance,
            "achieved": {
                name: round(
                    float(means[name][high_index] - means[name][low_index]), 6
                )
                for name in MATCHED
            },
            "arm_means": {
                name: {
                    "grad_diverse": round(float(means[name][low_index]), 4),
                    "grad_redundant": round(float(means[name][high_index]), 4),
                }
                for name in MATCHED
            },
        },
        "pool": {
            "s_bar_percentiles": {
                str(q): round(float(np.percentile(s_bar, q)), 4) for q in (1, 25, 50, 75, 99)
            },
            "admissible_pairs": len(pairs),
        },
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["discovery"], indent=2))
    print(json.dumps(report["matching"]["achieved"], indent=2))
    print(f"[written] {arguments.output}")
    return 0


def admissible_pairs(low_set, high_set, means, bits, s_bar, arguments, scales, tau):
    """Disjoint (low s_bar, high s_bar) pairs whose covariate means all match.

    Alignment is the tightest constraint, so it does the pruning: both sides are sorted by it and a
    binary search bounds the window each low quartet has to look at.
    """
    alignment_low = means["alignment"][low_set]
    alignment_high = means["alignment"][high_set]
    order_low = np.argsort(alignment_low)
    order_high = np.argsort(alignment_high)
    low_sorted = low_set[order_low]
    high_sorted = high_set[order_high]
    sorted_high_alignment = alignment_high[order_high]

    starts = np.searchsorted(
        sorted_high_alignment, alignment_low[order_low] - arguments.alignment_tolerance, "left"
    )
    stops = np.searchsorted(
        sorted_high_alignment, alignment_low[order_low] + arguments.alignment_tolerance, "right"
    )

    results = []
    for position, low_index in enumerate(low_sorted):
        window = high_sorted[starts[position] : stops[position]]
        if window.size == 0:
            continue
        length_low = means["response_tokens"][low_index]
        length_high = means["response_tokens"][window]
        keep = np.abs(length_high - length_low) <= arguments.length_tolerance * (
            (length_high + length_low) / 2
        )
        for name in ("reward", "k1_mean", "teacher_logprob_mean"):
            keep &= np.abs(means[name][window] - means[name][low_index]) <= tau * scales[name]
        window = window[keep]
        if window.size == 0:
            continue
        disjoint = ((bits[window, 0] & bits[low_index, 0]) == 0) & (
            (bits[window, 1] & bits[low_index, 1]) == 0
        )
        window = window[disjoint]
        for high_index in window:
            results.append(
                (int(low_index), int(high_index), float(s_bar[high_index] - s_bar[low_index]))
            )
    return results


if __name__ == "__main__":
    raise SystemExit(main())
