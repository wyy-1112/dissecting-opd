#!/usr/bin/env python3
"""Fail closed unless an OPD run has a complete, successful checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--allow-incomplete-run",
        action="store_true",
        help=(
            "accept a checkpoint from a run that later died. A crash at a subsequent "
            "step does not retroactively corrupt an earlier checkpoint, so every "
            "shard and metadata check below still has to pass; only the run-level "
            "success markers are waived. A still-running driver is never accepted, "
            "because its checkpoint may be mid-write."
        ),
    )
    parser.add_argument(
        "--allow-intermediate-step",
        action="store_true",
        help=(
            "accept an earlier checkpoint of a run that finished successfully. Only "
            "the 'step is the latest one' equality is relaxed to 'step exists'; the "
            "run-level success markers and every shard and metadata check still have "
            "to pass. Use this to read a mid-training policy, not to rescue a crash."
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    run_dir = arguments.run_dir.expanduser().resolve()
    state = run_dir / "state"
    if (state / "driver.running").exists():
        raise SystemExit(f"run is still active: {state / 'driver.running'}")
    if not arguments.allow_incomplete_run and (state / "driver.failed").exists():
        raise SystemExit(f"run failed: {state / 'driver.failed'}")

    latest_path = run_dir / "latest_checkpointed_iteration.txt"
    try:
        latest = int(latest_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"invalid latest-checkpoint marker: {latest_path}") from error
    if arguments.allow_incomplete_run or arguments.allow_intermediate_step:
        if arguments.step > latest:
            raise SystemExit(
                f"step {arguments.step} is beyond the last checkpoint {latest}: {run_dir}"
            )
    elif latest != arguments.step:
        raise SystemExit(
            f"latest checkpoint is {latest}, expected {arguments.step}: {run_dir}"
        )

    manifest_path = run_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid run manifest: {manifest_path}") from error
    if manifest and not arguments.allow_incomplete_run and manifest.get("status") != "success":
        raise SystemExit(
            f"run manifest is not successful: {manifest.get('status')!r}"
        )

    actor = run_dir / f"global_step_{arguments.step}" / "actor"
    fsdp_path = actor / "fsdp_config.json"
    try:
        fsdp = json.loads(fsdp_path.read_text(encoding="utf-8"))
        world_size = int(fsdp["world_size"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid FSDP config: {fsdp_path}") from error
    if world_size <= 0:
        raise SystemExit(f"invalid FSDP world size: {world_size}")

    model_shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    expected_names = {
        f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
    }
    actual_names = {path.name for path in model_shards}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise SystemExit(
            f"incomplete model shards in {actor}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    empty = [str(path) for path in model_shards if path.stat().st_size <= 0]
    if empty:
        raise SystemExit(f"empty model shards: {empty}")

    for relative in (
        "huggingface/config.json",
        "huggingface/tokenizer_config.json",
    ):
        path = actor / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise SystemExit(f"missing checkpoint metadata: {path}")

    print(
        json.dumps(
            {
                "status": "validated",
                "run_dir": str(run_dir),
                "step": arguments.step,
                "world_size": world_size,
                "model_shards": len(model_shards),
                "manifest_status": manifest.get("status"),
                "allow_incomplete_run": arguments.allow_incomplete_run,
                "latest_checkpointed_iteration": latest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
