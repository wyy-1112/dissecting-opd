#!/usr/bin/env python3
"""Run one OPD experiment from a release config.

    python opd/run.py configs/<group>/<pair>/<run>.yaml            # train
    python opd/run.py <config> --dry-run                           # print environment
    python opd/run.py <config> --cfg-only                          # compose hydra config only
    python opd/run.py <config> --gpus 8                            # one 8-GPU node
    python opd/run.py configs/teacher_training/<model>.yaml        # teacher GRPO / student SFT

Path variables used inside configs:
    ${REPO_ROOT}    this repository (set automatically)
    ${MODEL_ROOT}   directory holding the student/teacher checkpoints described in
                    models/README.md (default: $REPO_ROOT/models, which has no weights)
    ${OUTPUT_ROOT}  where run directories are written (default: $REPO_ROOT/outputs)
    ${POOL_ROOT}    directory with source prompt pools that are not shipped in
                    data/pools (default: $REPO_ROOT/data/pools)

If a config has ``train_schedule`` (a directory produced by
data_prep/schedule_ids.py), the exact training parquet is rebuilt into
``<that dir>/schedule.parquet`` before launching, and TRAIN_FILE points to it.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def expand(value, variables: dict[str, str]):
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: variables.get(m.group(1), os.environ.get(m.group(1), m.group(0))), value)
    return value


def check_resolved(values: dict[str, str] | list[str]) -> None:
    items = values.items() if isinstance(values, dict) else enumerate(values)
    unresolved = [f"{k}={v}" for k, v in items if re.search(r"\$\{\w+\}", v)]
    if unresolved:
        raise SystemExit("unresolved variables (export them first): " + ", ".join(unresolved))


def check_weights(env: dict[str, str]) -> None:
    for key in ("STUDENT_MODEL", "TEACHER_MODEL", "MODEL", "MODEL_PATH"):
        if key not in env:
            continue
        path = Path(env[key])
        if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
            raise SystemExit(f"{key}={path} has no weight files; see models/README.md")


def fit_single_node(env: dict[str, str], gpus: int) -> None:
    """Place a run on one node; OPD runs with a separate teacher pool split the GPUs evenly.

    Only the GPU layout changes: batch sizes and all training settings stay as configured.
    """
    if "NNODES" in env:  # teacher training (GRPO / SFT): one worker group
        rollout_tp = int(env.get("ROLLOUT_TP", "1"))
        if gpus % rollout_tp:
            raise SystemExit(f"--gpus {gpus}: ROLLOUT_TP={rollout_tp} does not divide {gpus} GPUs")
        env.update(NNODES="1", NPROC_PER_NODE=str(gpus))
        print(f"[single node] {gpus} GPUs", flush=True)
        return
    separate_pool = env.get("DISTILLATION_ENABLE_RESOURCE_POOL", "True").lower() == "true"
    if separate_pool and gpus % 2:
        raise SystemExit(f"--gpus {gpus}: a separate teacher pool needs an even number of GPUs")
    student = teacher = gpus // 2 if separate_pool else gpus
    teacher_tp = int(env.get("TEACHER_TP", "1"))
    if teacher % teacher_tp:
        raise SystemExit(f"--gpus {gpus}: TEACHER_TP={teacher_tp} does not divide {teacher} teacher GPUs")
    if int(env["TRAIN_BATCH_SIZE"]) % student:
        raise SystemExit(f"--gpus {gpus}: TRAIN_BATCH_SIZE={env['TRAIN_BATCH_SIZE']} is not divisible by {student}")
    env.update(
        STUDENT_NNODES="1",
        TEACHER_NNODES="1",
        STUDENT_GPUS_PER_NODE=str(student),
        TEACHER_GPUS_PER_NODE=str(teacher),
        DISTILLATION_ENABLE_RESOURCE_POOL=str(separate_pool),
    )
    layout = f"student {student} + teacher {teacher}" if separate_pool else f"student and teacher share {gpus}"
    print(f"[single node] {layout} GPUs", flush=True)


def rebuild_schedule(spec: dict, variables: dict[str, str]) -> str:
    ids_dir = Path(expand(spec["ids"], variables))
    target = ids_dir / "schedule.parquet"
    if target.exists():
        return str(target)
    rows = [str(ids_dir / "support.parquet")] if (ids_dir / "support.parquet").exists() else []
    rows += [expand(p, variables) for p in spec.get("pools", [])]
    cmd = [sys.executable, str(REPO_ROOT / "data_prep" / "schedule_ids.py"), "rebuild", str(ids_dir), "--rows", *rows,
           "--out", str(target)]
    print("[rebuild]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cfg-only", action="store_true", help="compose and print the hydra config, do not train")
    parser.add_argument("--gpus", type=int, help="run on a single node with this many GPUs (e.g. 8) instead of the configured layout")
    parser.add_argument("overrides", nargs="*", help="extra hydra overrides appended to the driver call")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    variables = {
        "REPO_ROOT": str(REPO_ROOT),
        "MODEL_ROOT": os.environ.get("MODEL_ROOT", str(REPO_ROOT / "models")),
        "OUTPUT_ROOT": os.environ.get("OUTPUT_ROOT", str(REPO_ROOT / "outputs")),
        "POOL_ROOT": os.environ.get("POOL_ROOT", str(REPO_ROOT / "data" / "pools")),
    }
    env = {k: str(expand(v, variables)) for k, v in config["env"].items() if v is not None}
    if args.gpus:
        fit_single_node(env, args.gpus)
    extra = [expand(o, variables) for o in config.get("extra_overrides", [])] + list(args.overrides)
    if not args.dry_run:
        check_resolved(env)
        check_resolved(extra)
    if not args.dry_run and not args.cfg_only:
        check_weights(env)
    if config.get("train_schedule") and not args.dry_run:
        env["TRAIN_FILE"] = rebuild_schedule(config["train_schedule"], variables)
    elif config.get("train_schedule"):
        env["TRAIN_FILE"] = str(Path(expand(config["train_schedule"]["ids"], variables)) / "schedule.parquet")

    if args.dry_run:
        for key in sorted(env):
            print(f"{key}={env[key]}")
        return

    driver = REPO_ROOT / config.get("driver", "opd/train_opd.sh")
    if args.cfg_only:
        extra += ["--cfg", "job", "--resolve"]
    os.makedirs(env["OUTPUT_DIR"], exist_ok=True)
    full_env = {**os.environ, **env}
    os.execvpe("bash", ["bash", str(driver), *extra], full_env)


if __name__ == "__main__":
    main()
