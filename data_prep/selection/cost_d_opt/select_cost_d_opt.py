#!/usr/bin/env python3
"""Freeze rollout-efficient M8 supports and reproducible OPD schedules.

All support IDs are chosen from discovery rollouts only.  Confirmation
rollouts are attached after selection solely to audit the stability of the
estimated generation cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Directory laid out like the original project (data/opd, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[3] / "outputs" / "opd_project_root"))
ANALYSIS = Path(__file__).resolve().parent
DATA_PREP = ROOT / "scripts/data_prep"
for directory in (ANALYSIS, DATA_PREP):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from d_optimal_schedules import schedule_rows  # noqa: E402
from svd_gradient_coverage import greedy_d_optimal  # noqa: E402


SCHEMA = "opd_rollout_efficient_m8_schedules_v1"
PROMPT_ARTIFACT_SCHEMA = "opd_prompt_token_aggregates_v1"
BASELINE_SCHEMA = "opd_dopt_carrier_m8_schedules_v1"
REQUIRED_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, row_group_size=512)
    os.replace(temporary, path)


def prompt_sort_key(prompt: str) -> tuple[int, int | str]:
    if prompt.isdigit():
        return 0, int(prompt)
    return 1, prompt


def sorted_indices(
    indices: Sequence[int],
    values: np.ndarray,
    *,
    descending: bool,
    prompts: Sequence[str],
) -> list[int]:
    direction = -1.0 if descending else 1.0
    return sorted(
        indices,
        key=lambda index: (
            direction * float(values[index]),
            prompt_sort_key(str(prompts[index])),
        ),
    )


def carrier_scores(gradients: Any, lengths: Any) -> np.ndarray:
    values = gradients.double()
    weights = lengths.double()
    pool = (values * weights[None, :, None]).sum(dim=1) / weights.sum()
    numerator = (values * pool[:, None, :]).sum(dim=-1)
    denominator = values.norm(dim=-1) * pool.norm(dim=-1)[:, None]
    return (
        (numerator / denominator.clamp(min=1e-24))
        .mean(dim=0)
        .cpu()
        .numpy()
    )


def normalized_d_opt_logdet(
    coordinates: np.ndarray,
    indices: Sequence[int],
    ridge: float,
) -> float:
    selected = coordinates[np.asarray(indices, dtype=np.int64)]
    matrix = (
        np.eye(len(indices), dtype=np.float64)
        + selected @ selected.T / ridge
    )
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("D-optimal information matrix is not positive definite")
    return float(value)


def cost_metrics(
    response_lengths: np.ndarray,
    indices: Sequence[int],
    *,
    batch_size: int,
    response_cap: int,
) -> dict[str, Any]:
    selected = response_lengths[np.asarray(indices, dtype=np.int64)]
    prompt_means = selected.mean(axis=1)
    return {
        "mean_response_tokens": float(selected.mean()),
        "median_response_tokens": float(np.median(selected)),
        "p90_response_tokens": float(np.quantile(selected, 0.90)),
        "p95_response_tokens": float(np.quantile(selected, 0.95)),
        "maximum_response_tokens": int(selected.max()),
        "response_cap_fraction": float((selected >= response_cap).mean()),
        "prompt_mean_minimum": float(prompt_means.min()),
        "prompt_mean_maximum": float(prompt_means.max()),
        "expected_response_tokens_per_training_batch": float(
            selected.mean() * batch_size
        ),
    }


def support_frame(source: Any, problem_ids: Sequence[str]) -> Any:
    import pandas as pd

    by_id = {
        str(row["extra_info"]["index"]): row
        for row in source.to_dict(orient="records")
    }
    missing = set(problem_ids) - set(by_id)
    if missing:
        raise ValueError(f"support prompts absent from source: {sorted(missing)}")
    return pd.DataFrame([by_id[problem_id] for problem_id in problem_ids])


def validate_schedule(
    frame: Any,
    *,
    problem_ids: Sequence[str],
    batch_size: int,
    steps: int,
) -> None:
    expected = set(problem_ids)
    if len(frame) != batch_size * steps:
        raise AssertionError("schedule row count changed")
    repeats = batch_size // len(problem_ids)
    for step in range(steps):
        batch = frame.iloc[step * batch_size : (step + 1) * batch_size]
        observed = [
            str(extra["index"]) for extra in batch["extra_info"].tolist()
        ]
        counts = {
            problem_id: observed.count(problem_id)
            for problem_id in expected
        }
        if set(observed) != expected or set(counts.values()) != {repeats}:
            raise AssertionError(f"step {step}: support exposure is not balanced")
    invalid = [
        str(row["extra_info"]["index"])
        for row in frame.iloc[:batch_size].to_dict(orient="records")
        if not row["prompt"][0]["content"].endswith(REQUIRED_SUFFIX)
    ]
    if invalid:
        raise ValueError(f"training instruction missing for prompts: {invalid}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-artifact", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support", type=int, default=8)
    parser.add_argument("--short-carrier-quantile", type=float, default=0.25)
    parser.add_argument("--cost-power", type=float, default=1.0)
    parser.add_argument("--response-cap", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--training-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.support != 8:
        raise ValueError("this protocol is frozen for M8")
    if not 0 < arguments.short_carrier_quantile < 1:
        raise ValueError("--short-carrier-quantile must be in (0, 1)")
    if arguments.cost_power <= 0:
        raise ValueError("--cost-power must be positive")
    if arguments.batch_size % arguments.support:
        raise ValueError("batch size must be divisible by support")

    artifact_path = arguments.prompt_artifact.resolve()
    baseline_path = arguments.baseline_manifest.resolve()
    source_path = arguments.source.resolve()
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output_dir}")

    import pandas as pd
    import torch

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )
    if artifact.get("schema_version") != PROMPT_ARTIFACT_SCHEMA:
        raise ValueError("unexpected prompt aggregate artifact schema")
    prompts = [str(value) for value in artifact["problem_ids"]]
    if len(prompts) != 384 or len(set(prompts)) != len(prompts):
        raise ValueError("prompt artifact must contain 384 unique prompts")

    discovery_lengths = (
        artifact["discovery"]["response_lengths"].double().cpu().numpy()
    )
    confirmation_lengths = (
        artifact["confirmation"]["response_lengths"].double().cpu().numpy()
    )
    if discovery_lengths.shape != (384, 16):
        raise ValueError("expected 16 discovery rollouts for each of 384 prompts")
    if confirmation_lengths.shape != discovery_lengths.shape:
        raise ValueError("confirmation response-length matrix shape changed")
    discovery_cost = discovery_lengths.mean(axis=1)
    confirmation_cost = confirmation_lengths.mean(axis=1)
    discovery_carrier = carrier_scores(
        artifact["discovery"]["g_i"],
        artifact["discovery"]["L_i"],
    )
    coordinates = (
        artifact["stable_coordinates"]["sqrt_L_i_z_i"]
        .double()
        .cpu()
        .numpy()
    )
    ridge = float(artifact["stable_coordinates"]["ridge_lambda"])

    all_indices = list(range(len(prompts)))
    shortest = sorted_indices(
        all_indices,
        discovery_cost,
        descending=False,
        prompts=prompts,
    )[: arguments.support]
    short_threshold = float(
        np.quantile(discovery_cost, arguments.short_carrier_quantile)
    )
    short_pool = [
        index
        for index in all_indices
        if float(discovery_cost[index]) <= short_threshold
    ]
    if len(short_pool) < arguments.support:
        raise ValueError("short-carrier candidate pool is smaller than support")
    short_carrier = sorted_indices(
        short_pool,
        discovery_carrier,
        descending=True,
        prompts=prompts,
    )[: arguments.support]
    cost_aware_d_opt = greedy_d_optimal(
        coordinates,
        arguments.support,
        ridge=ridge,
        costs=discovery_cost,
        cost_power=arguments.cost_power,
    )

    arms = {
        "shortest_m8": shortest,
        "short_carrier_m8": short_carrier,
        "cost_aware_d_optimal_m8": cost_aware_d_opt,
    }
    if any(len(set(indices)) != arguments.support for indices in arms.values()):
        raise AssertionError("a selected arm contains duplicate prompts")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != BASELINE_SCHEMA:
        raise ValueError("unexpected baseline schedule manifest schema")
    baseline_arms = {
        arm: [
            prompts.index(str(problem_id))
            for problem_id in baseline["arms"][arm]["problem_ids"]
        ]
        for arm in ("svd_d_optimal_m8", "carrier_top8")
    }
    baseline_d_opt = normalized_d_opt_logdet(
        coordinates,
        baseline_arms["svd_d_optimal_m8"],
        ridge,
    )

    selection_report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "analysis_regime": "prospective_pretraining_selection",
        "selection_used_confirmation": False,
        "cost_definition": (
            "mean student response tokens across the 16 frozen discovery "
            "rollouts generated with the training sampling protocol"
        ),
        "methods": {
            "shortest_m8": "eight smallest discovery mean response lengths",
            "short_carrier_m8": (
                "highest discovery carrier cosine among prompts in the "
                f"shortest {arguments.short_carrier_quantile:.0%} by discovery "
                "mean response length"
            ),
            "cost_aware_d_optimal_m8": (
                "greedy D-optimal marginal logdet gain divided by relative "
                f"discovery rollout cost^{arguments.cost_power:g}"
            ),
        },
        "parameters": {
            "support": arguments.support,
            "short_carrier_quantile": arguments.short_carrier_quantile,
            "short_carrier_cost_threshold": short_threshold,
            "short_carrier_candidate_prompts": len(short_pool),
            "cost_power": arguments.cost_power,
            "response_cap": arguments.response_cap,
            "ridge_lambda": ridge,
        },
        "candidate_pool": {
            "prompts": len(prompts),
            "discovery_mean_response_tokens": float(discovery_lengths.mean()),
            "confirmation_mean_response_tokens": float(
                confirmation_lengths.mean()
            ),
            "discovery_confirmation_prompt_cost_correlation": float(
                np.corrcoef(discovery_cost, confirmation_cost)[0, 1]
            ),
        },
        "baseline_arms": {},
        "selected_arms": {},
    }

    def arm_report(indices: Sequence[int]) -> dict[str, Any]:
        d_opt = normalized_d_opt_logdet(coordinates, indices, ridge)
        return {
            "problem_ids": [prompts[index] for index in indices],
            "discovery_cost": cost_metrics(
                discovery_lengths,
                indices,
                batch_size=arguments.batch_size,
                response_cap=arguments.response_cap,
            ),
            "confirmation_cost_audit_only": cost_metrics(
                confirmation_lengths,
                indices,
                batch_size=arguments.batch_size,
                response_cap=arguments.response_cap,
            ),
            "discovery_d_opt_logdet": d_opt,
            "fraction_of_baseline_d_opt_logdet": d_opt / baseline_d_opt,
            "discovery_mean_carrier_cosine": float(
                discovery_carrier[np.asarray(indices, dtype=np.int64)].mean()
            ),
        }

    for arm, indices in baseline_arms.items():
        selection_report["baseline_arms"][arm] = arm_report(indices)
    for arm, indices in arms.items():
        selection_report["selected_arms"][arm] = arm_report(indices)

    output_dir.mkdir(parents=True)
    support_dir = output_dir / "supports"
    schedule_dir = output_dir / "schedules"
    support_dir.mkdir()
    schedule_dir.mkdir()
    selection_path = output_dir / "selection.json"
    atomic_json(selection_path, selection_report)

    source = pd.read_parquet(source_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "validated",
        "analysis_regime": "prospective_pretraining_selection",
        "selection_used_confirmation": False,
        "training_seed": arguments.training_seed,
        "batch_size": arguments.batch_size,
        "optimizer_steps": arguments.steps,
        "total_presentations_per_arm": arguments.batch_size * arguments.steps,
        "prompt_artifact": {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
        },
        "baseline_manifest": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "source": {
            "path": str(source_path),
            "rows": len(source),
            "sha256": sha256_file(source_path),
        },
        "arms": {},
    }
    for arm, indices in arms.items():
        problem_ids = [prompts[index] for index in indices]
        support = support_frame(source, problem_ids)
        support_path = support_dir / f"{arm}.parquet"
        atomic_parquet(support, support_path)
        rows = schedule_rows(
            source,
            problem_ids,
            arm=arm,
            batch_size=arguments.batch_size,
            steps=arguments.steps,
            training_seed=arguments.training_seed,
            selection_schema=SCHEMA,
            metadata_key="rollout_efficient_m8_selection",
            analysis_regime="prospective_pretraining_selection",
        )
        frame = pd.DataFrame(rows)
        validate_schedule(
            frame,
            problem_ids=problem_ids,
            batch_size=arguments.batch_size,
            steps=arguments.steps,
        )
        schedule_path = (
            schedule_dir
            / f"{arm}_s{arguments.steps}_b{arguments.batch_size}_seed{arguments.training_seed}.parquet"
        )
        atomic_parquet(frame, schedule_path)
        manifest["arms"][arm] = {
            "problem_ids": problem_ids,
            "selection_metrics": selection_report["selected_arms"][arm],
            "support": {
                "path": str(support_path),
                "rows": len(support),
                "sha256": sha256_file(support_path),
            },
            "schedule": {
                "path": str(schedule_path),
                "rows": len(frame),
                "sha256": sha256_file(schedule_path),
            },
            "presentations_per_prompt": (
                arguments.batch_size // arguments.support * arguments.steps
            ),
        }

    manifest_path = output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        output_dir / "SCHEDULES_VALIDATED.json",
        {
            "schema_version": SCHEMA,
            "status": "validated",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "selected_arms": {
                    arm: selection_report["selected_arms"][arm]
                    for arm in arms
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
