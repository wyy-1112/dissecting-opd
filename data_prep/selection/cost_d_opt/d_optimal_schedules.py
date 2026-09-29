#!/usr/bin/env python3
"""Materialize an exploratory SVD-D-optimal M=4 arm and matched control.

The treatment IDs are frozen in the discovery-half SVD report.  The control
is sampled without looking at any gradient-geometry objective: it is drawn
from discovery-side quartets matched on alignment, response length, reward,
K1 and teacher log-probability.  Confirmation responses are never consulted
while constructing either training schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[3] / "outputs" / "opd_project_root"))
ANALYSIS = Path(__file__).resolve().parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from coverage_v2_common import build_cohort, debiased_similarity  # noqa: E402
from coverage_v2_select import enumerate_quartets  # noqa: E402


SCHEMA = "opd_svd_d_optimal_m4_v1"
FEATURES = (
    "alignment",
    "response_tokens",
    "reward",
    "k1_mean",
    "teacher_logprob_mean",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)


def selected_problem_ids(
    report: Mapping[str, Any],
    *,
    rank: int,
    support: int,
) -> list[str]:
    if report.get("schema_version") != "opd_svd_gradient_coverage_v1":
        raise ValueError("unexpected SVD report schema")
    entry = report["ranks"][str(rank)]["supports"][str(support)][
        "svd_d_optimal"
    ]
    values = [str(value) for value in entry["problem_ids"]]
    if len(values) != support or len(set(values)) != support:
        raise ValueError("SVD report does not contain one unique support set")
    return values


def feature_matrix(cohort: Any, alignment: np.ndarray) -> np.ndarray:
    columns = {
        "alignment": np.asarray(alignment, dtype=np.float64),
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in cohort.covariates.items()
        },
    }
    return np.column_stack([columns[name] for name in FEATURES])


def choose_matched_control(
    prompts: Sequence[str],
    matrix: np.ndarray,
    treatment_ids: Sequence[str],
    *,
    support: int,
    seed: int,
    alignment_tolerance: float = 0.02,
    length_tolerance: float = 0.10,
    eligible_quartets: np.ndarray | None = None,
) -> tuple[list[str], dict[str, Any]]:
    position = {prompt: index for index, prompt in enumerate(prompts)}
    treatment = np.array([position[value] for value in treatment_ids], dtype=np.int64)
    target = matrix[treatment].mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales <= 0] = 1.0

    quartets = enumerate_quartets(len(prompts), support)
    disjoint = ~np.isin(quartets, treatment).any(axis=1)
    if eligible_quartets is not None:
        eligible_quartets = np.asarray(eligible_quartets, dtype=bool)
        if eligible_quartets.shape != (len(quartets),):
            raise ValueError("eligible quartet mask has the wrong shape")
        disjoint &= eligible_quartets
    means = matrix[quartets].mean(axis=1)
    eligible = (
        disjoint
        & (np.abs(means[:, 0] - target[0]) <= alignment_tolerance)
        & (
            np.abs(means[:, 1] - target[1])
            <= length_tolerance * max(target[1], 1.0)
        )
    )
    tau_used = None
    for tau in (0.25, 0.50, 0.75, 1.00):
        current = eligible.copy()
        for column in range(2, len(FEATURES)):
            current &= (
                np.abs(means[:, column] - target[column])
                <= tau * scales[column]
            )
        if int(current.sum()) >= 100:
            eligible = current
            tau_used = tau
            break
    if tau_used is None:
        raise ValueError("fewer than 100 covariate-matched control quartets")

    candidates = np.flatnonzero(eligible)
    chosen_row = int(random.Random(seed).choice(candidates.tolist()))
    chosen = quartets[chosen_row]
    diagnostics = {
        "selection": "seeded random draw from covariate-matched quartets",
        "seed": seed,
        "eligible_quartets": int(len(candidates)),
        "alignment_tolerance": alignment_tolerance,
        "length_tolerance_fraction": length_tolerance,
        "other_feature_tolerance_sd": tau_used,
        "features": list(FEATURES),
        "treatment_means": {
            name: float(value) for name, value in zip(FEATURES, target)
        },
        "control_means": {
            name: float(value)
            for name, value in zip(FEATURES, means[chosen_row])
        },
        "standardized_mean_differences": {
            name: float((means[chosen_row, index] - target[index]) / scales[index])
            for index, name in enumerate(FEATURES)
        },
    }
    return [str(prompts[index]) for index in chosen], diagnostics


def schedule_rows(
    source: Any,
    problem_ids: Sequence[str],
    *,
    arm: str,
    batch_size: int,
    steps: int,
    training_seed: int,
    selection_schema: str = SCHEMA,
    metadata_key: str = "svd_d_optimal_selection",
    analysis_regime: str = "exploratory_posthoc",
) -> list[Any]:
    by_index: dict[str, Any] = {}
    for position in range(len(source)):
        row = source.iloc[position]
        key = str(row["extra_info"]["index"])
        by_index.setdefault(key, row)
    missing = set(problem_ids) - set(by_index)
    if missing:
        raise ValueError(f"selected prompts absent from source: {sorted(missing)}")
    if batch_size % len(problem_ids):
        raise ValueError("batch size must be divisible by support size")

    repeats = batch_size // len(problem_ids)
    shuffler = random.Random(training_seed)
    rows = []
    for step in range(steps):
        batch = [
            problem_id
            for problem_id in problem_ids
            for _ in range(repeats)
        ]
        shuffler.shuffle(batch)
        for position, problem_id in enumerate(batch):
            row = by_index[problem_id].copy()
            extra = dict(row["extra_info"])
            extra[metadata_key] = {
                "schema_version": selection_schema,
                "analysis_regime": analysis_regime,
                "arm": arm,
                "support_size": len(problem_ids),
                "optimizer_step": step,
                "position_in_step": position,
                "presentation_index": step * batch_size + position,
            }
            extra.pop("math_prompt_diversity", None)
            extra.pop("coverage_selection", None)
            row["extra_info"] = extra
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    base = ROOT / "data/opd/coverage_selection_v2"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--svd-report",
        type=Path,
        default=ROOT / "results/analysis/svd_gradient_coverage_k16_v1.json",
    )
    parser.add_argument("--sketch-dir", type=Path, default=base / "sketches")
    parser.add_argument("--split", type=Path, default=base / "split_manifest.json")
    parser.add_argument("--label", default="nonhom_30b")
    parser.add_argument(
        "--source",
        type=Path,
        default=base / "shortlist_96.parquet",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--support", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--control-seed", type=int, default=20260826)
    parser.add_argument(
        "--analysis-regime",
        default="exploratory_posthoc",
        help="provenance label written to the manifest and every schedule row",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/opd/svd_d_optimal_m4_v1",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = json.loads(arguments.svd_report.read_text())
    treatment = selected_problem_ids(
        report,
        rank=arguments.rank,
        support=arguments.support,
    )
    cohort = build_cohort(
        arguments.sketch_dir,
        arguments.split,
        arguments.label,
        half="discovery",
    )
    split = len(cohort.rows[cohort.prompts[0]]) // 2
    _, alignment = debiased_similarity(cohort, split)
    matrix = feature_matrix(cohort, alignment)
    control, matching = choose_matched_control(
        cohort.prompts,
        matrix,
        treatment,
        support=arguments.support,
        seed=arguments.control_seed,
    )

    import pandas as pd

    source = pd.read_parquet(arguments.source)
    schedule_dir = arguments.output_dir / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    arms = {
        "svd_d_optimal": treatment,
        "random_matched": control,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "analysis_regime": arguments.analysis_regime,
        "candidate_pool": (
            "96-prompt covariate shortlist; conclusions are conditional on this pool"
        ),
        "svd_report": str(arguments.svd_report.resolve()),
        "svd_report_sha256": sha256_file(arguments.svd_report),
        "split_manifest": str(arguments.split.resolve()),
        "split_manifest_sha256": sha256_file(arguments.split),
        "selection_half": "discovery",
        "confirmation_used_for_selection": False,
        "rank": arguments.rank,
        "support": arguments.support,
        "batch_size": arguments.batch_size,
        "steps": arguments.steps,
        "presentations": arguments.batch_size * arguments.steps,
        "training_seed": arguments.training_seed,
        "control_matching": matching,
        "arms": {},
    }
    for arm, problem_ids in arms.items():
        rows = schedule_rows(
            source,
            problem_ids,
            arm=arm,
            batch_size=arguments.batch_size,
            steps=arguments.steps,
            training_seed=arguments.training_seed,
            analysis_regime=arguments.analysis_regime,
        )
        frame = pd.DataFrame(rows).reset_index(drop=True)
        path = schedule_dir / (
            f"{arm}_m{arguments.support}_s{arguments.steps}"
            f"_b{arguments.batch_size}.parquet"
        )
        frame.to_parquet(path, row_group_size=512)
        manifest["arms"][arm] = {
            "problem_ids": problem_ids,
            "rows": len(frame),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }

    manifest_path = arguments.output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "treatment": treatment,
                "control": control,
                "matching": matching,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
