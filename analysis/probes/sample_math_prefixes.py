#!/usr/bin/env python3
"""Sample a prompts x responses cohort under the initial student, for gradient-spectrum work.

The question this cohort serves is whether the OPD gradient's span is set by the prompt set or
by the token-level sampling inside each response, so the cohort has to be a rectangle: the same
K samples for every prompt, all drawn from one policy.  That policy is the *initial* student,
because an offline selection rule can only be built from the policy it starts with.

Prompts come from a frozen shortlist derived from the m-sweep schedule.  The
prospective reliability-SVD run uses the same response cap as training; changing
that cap creates a different length-weighted gradient geometry and is therefore
not a valid shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "opd_gradient_spectrum_cohort_v1"


def load_reward_fn(path: Path):
    spec = importlib.util.spec_from_file_location("math_reward_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        default=str(
            ROOT
            / "data/schedules/support_size/math_grpo500_to_qwen3_4b"
            / "deepmath_m3840_s15/schedule.parquet"
        ),
    )
    parser.add_argument(
        "--student",
        default=os.path.join(os.environ.get("MODEL_ROOT", str(ROOT / "models")), "qwen3_4b"),
    )
    parser.add_argument("--prompts", type=int, default=96)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--required-question-suffix", default=None)
    parser.add_argument("--reward-fn", default=str(ROOT / "opd/rewards/math_reward.py"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        help="Read the frozen ordered prompt IDs from this manifest.",
    )
    parser.add_argument(
        "--write-candidate-manifest",
        type=Path,
        help="Freeze the selected ordered prompt IDs before GPU generation.",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs/gradient_spectrum_v1/cohort"),
    )
    args = parser.parse_args()
    if args.shards <= 0 or not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must be in [0, --shards)")
    if args.candidate_manifest and args.write_candidate_manifest:
        raise SystemExit(
            "--candidate-manifest and --write-candidate-manifest are mutually exclusive"
        )
    if args.write_candidate_manifest and not args.validate_only:
        raise SystemExit("--write-candidate-manifest requires --validate-only")

    import pandas as pd

    frame = pd.read_parquet(args.schedule)
    unique: dict[int, dict] = {}
    for row in frame.itertuples(index=False):
        index = int(row.extra_info["index"])
        if index not in unique:
            unique[index] = {
                "problem_id": str(index),
                "question": row.prompt[0]["content"],
                "ground_truth": row.reward_model["ground_truth"],
                "data_source": row.data_source,
            }
    ordered = [unique[key] for key in sorted(unique)]
    by_problem_id = {record["problem_id"]: record for record in ordered}
    if args.candidate_manifest:
        frozen = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
        if frozen.get("schema_version") != "opd_gradient_candidate_pool_v1":
            raise SystemExit("unexpected candidate-manifest schema")
        if int(frozen.get("count", -1)) != args.prompts:
            raise SystemExit("candidate-manifest count differs from --prompts")
        if int(frozen.get("selection_seed", -1)) != args.seed:
            raise SystemExit("candidate-manifest seed differs from --seed")
        frozen_ids = [str(value) for value in frozen["problem_ids"]]
        if len(frozen_ids) != len(set(frozen_ids)):
            raise SystemExit("candidate-manifest contains duplicate problem IDs")
        missing = [problem_id for problem_id in frozen_ids if problem_id not in by_problem_id]
        if missing:
            raise SystemExit(
                f"candidate-manifest IDs are absent from the schedule: {missing[:5]}"
            )
        globally_selected = [by_problem_id[problem_id] for problem_id in frozen_ids]
    else:
        rng = random.Random(args.seed)
        globally_selected = rng.sample(ordered, args.prompts)
        frozen_ids = [record["problem_id"] for record in globally_selected]

    selected = []
    for selection_index, record in enumerate(globally_selected):
        record["selection_index"] = selection_index
        if selection_index % args.shards != args.shard:
            continue
        selected.append(record)
    if not selected:
        raise SystemExit("this shard has no selected prompts")
    if args.required_question_suffix is not None:
        invalid = [
            record["problem_id"]
            for record in globally_selected
            if not record["question"].endswith(args.required_question_suffix)
        ]
        if invalid:
            raise SystemExit(
                "questions do not preserve the required training instruction: "
                + ",".join(invalid)
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    for record in globally_selected:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        record["prompt_text"] = text
        record["prompt_token_ids"] = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(record["prompt_token_ids"]) > args.max_prompt_tokens:
            raise SystemExit(
                f"prompt {record['problem_id']} has "
                f"{len(record['prompt_token_ids'])} tokens, exceeding "
                f"training limit {args.max_prompt_tokens}"
            )
    if args.write_candidate_manifest:
        canonical = json.dumps(
            frozen_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        candidate_manifest = {
            "schema_version": "opd_gradient_candidate_pool_v1",
            "frozen_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_schedule": str(Path(args.schedule).resolve()),
            "selection_method": "seeded uniform sample without replacement",
            "selection_seed": args.seed,
            "count": len(frozen_ids),
            "problem_ids": frozen_ids,
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
        args.write_candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_candidate_manifest.write_text(
            json.dumps(candidate_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "prompts": len(globally_selected),
                    "samples_per_prompt": args.samples,
                    "max_response_tokens": args.max_tokens,
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "maximum_observed_prompt_tokens": max(
                        len(record["prompt_token_ids"])
                        for record in globally_selected
                    ),
                    "required_question_suffix": args.required_question_suffix,
                    "candidate_manifest": (
                        str(args.write_candidate_manifest)
                        if args.write_candidate_manifest
                        else (
                            str(args.candidate_manifest)
                            if args.candidate_manifest
                            else None
                        )
                    ),
                    "status": "valid",
                },
                indent=2,
            )
        )
        return 0

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.student,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + args.max_prompt_tokens,
        trust_remote_code=True,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    outputs = llm.generate(
        [record["prompt_text"] for record in selected],
        SamplingParams(
            n=args.samples,
            temperature=1.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )

    compute_score = load_reward_fn(Path(args.reward_fn))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / "cohort.jsonl"
    kept_samples = 0
    truncated = 0
    with cohort_path.open("w") as stream:
        for record, output in zip(selected, outputs, strict=True):
            samples = []
            for completion in output.outputs:
                finished = completion.finish_reason == "stop"
                truncated += int(not finished)
                score = compute_score(
                    record["data_source"],
                    completion.text,
                    record["ground_truth"],
                )
                if isinstance(score, dict):
                    score = score.get("score", 0.0)
                samples.append(
                    {
                        "response_token_ids": list(completion.token_ids),
                        "response_tokens": len(completion.token_ids),
                        "reward": float(score),
                        "truncated": not finished,
                    }
                )
            kept_samples += len(samples)
            stream.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA,
                        "problem_id": record["problem_id"],
                        "question": record["question"],
                        "ground_truth": record["ground_truth"],
                        "data_source": record["data_source"],
                        "selection_index": record["selection_index"],
                        "prompt_text": record["prompt_text"],
                        "prompt_token_ids": record["prompt_token_ids"],
                        "samples": samples,
                    }
                )
                + "\n"
            )

    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA,
        "schedule": args.schedule,
        "student": args.student,
        "prompts": len(selected),
        "global_prompts": len(globally_selected),
        "samples_per_prompt": args.samples,
        "samples_total": kept_samples,
        "truncated_samples": truncated,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_tokens + args.max_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "maximum_observed_prompt_tokens": max(
            len(record["prompt_token_ids"]) for record in selected
        ),
        "required_question_suffix": args.required_question_suffix,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "samples_per_prompt": args.samples,
        },
        "prompt_rendering": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "note": (
                "DeepSeek-R1-Distill's template always opens the native "
                "thinking block; the extra template argument is ignored there"
            ),
        },
        "engine": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        "selection_seed": args.seed,
        "candidate_manifest": (
            str(args.candidate_manifest.resolve())
            if args.candidate_manifest
            else None
        ),
        "shard": args.shard,
        "shards": args.shards,
        "selection_indices": [
            int(record["selection_index"]) for record in selected
        ],
        "problem_ids": [record["problem_id"] for record in selected],
        "cohort_sha256": digest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "problem_ids"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
