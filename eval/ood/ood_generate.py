"""Generate one shard of an OOD benchmark with vLLM.

One process owns one GPU (tensor parallel 1) and handles a round-robin slice of the
tasks, so a benchmark is spread across the whole node by launching several of these.
Sampling defaults match the G-OPD paper's code eval: temperature 1.0, top_p 1.0,
max_tokens 16384.

Usage:
  python3 scripts/eval/ood_generate.py \
      --bench livecodebench_v6 --model <hf_dir> --out <shard.jsonl> \
      --n 8 --shard-id 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ood_common import BENCHMARKS, build_prompt, load_tasks, sha256_text, shard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", required=True, choices=BENCHMARKS)
    p.add_argument("--model", required=True, help="HF model directory")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--out", required=True, help="output jsonl for this shard")
    p.add_argument("--n", type=int, default=8, help="samples per task")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="explicit vLLM model dtype; scaling interventions require float32",
    )
    p.add_argument("--seed", type=int, default=42)
    # Qwen's own recommendation for thinking mode is temperature 0.6, top_p 0.95, top_k 20,
    # min_p 0, and never greedy.  The defaults here stay at the G-OPD code-eval settings so
    # existing non-thinking numbers reproduce; a thinking run passes all of them explicitly.
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--top-k", type=int, default=-1, help="-1 disables, Qwen recommends 20")
    p.add_argument("--min-p", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"[skip] {out} already exists")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks = shard(load_tasks(args.bench), args.shard_id, args.num_shards)
    print(
        f"[{args.bench}] shard {args.shard_id}/{args.num_shards}: "
        f"{len(tasks)} tasks x n={args.n} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    prompts = [
        build_prompt(tokenizer, t.user_content, enable_thinking=args.enable_thinking)
        for t in tasks
    ]

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed + args.shard_id,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
    )

    started = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - started

    tmp = out.with_suffix(out.suffix + ".partial")
    n_truncated = 0
    total_tokens = 0
    total_samples = 0
    with tmp.open("w") as fh:
        for task, prompt, result in zip(tasks, prompts, outputs):
            completions = [c.text for c in result.outputs]
            finish_reasons = [c.finish_reason for c in result.outputs]
            token_counts = [len(c.token_ids) for c in result.outputs]
            n_truncated += sum(1 for r in finish_reasons if r == "length")
            total_tokens += sum(token_counts)
            total_samples += len(completions)
            fh.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "prompt_sha256": sha256_text(prompt),
                        "meta": task.meta,
                        "outputs": completions,
                        "finish_reasons": finish_reasons,
                        "output_tokens": token_counts,
                    }
                )
                + "\n"
            )
    tmp.rename(out)

    print(
        f"[done] {args.bench} shard {args.shard_id}: {len(tasks)} tasks, "
        f"{total_samples} samples, avg_len={total_tokens / max(total_samples, 1):.1f}, "
        f"truncated={n_truncated}/{total_samples}, {elapsed / 60:.1f} min -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
