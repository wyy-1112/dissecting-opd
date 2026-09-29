#!/usr/bin/env python3
"""Build the Eurus code split into a GRPO-ready parquet for the Qwen3-4B code teacher.

G-OPD trains its code-domain teacher on the code half of PRIME-RL/Eurus-2-RL-Data, whose
prompts already end with the fenced-code instruction that our code benchmarks use. Two
things need fixing before verl can train on it: the rows carry PRIME's [ASSESS]/[ADVANCE]
system prompt, which has nothing to do with Qwen3 non-thinking, and a minority of rows ship
call-based tests that our stdio judge cannot run.

Usage:
  python data_prep/build_eurus_code_pool.py --audit-only   # report what would be kept
  python data_prep/build_eurus_code_pool.py                # write data/pools/eurus_code{,_dev}.parquet

The upstream snapshot (PRIME-RL/Eurus-2-RL-Data@9776b13) is downloaded unless --eurus-dir points
at a local copy; the tokenizer only measures prompt length (default $MODEL_ROOT/qwen3_4b).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
EURUS_REPO = "PRIME-RL/Eurus-2-RL-Data"
EURUS_REVISION = "9776b13264b5aaa0b16495fcf086a0a8d86fd655"
DEFAULT_TOKENIZER = str(Path(os.environ["MODEL_ROOT"]) / "qwen3_4b") if os.environ.get("MODEL_ROOT") else "Qwen/Qwen3-4B"
OUTPUT_NAMES = {"train": "eurus_code", "validation": "eurus_code_dev"}
# The trailing line our code evals add on top of the Eurus instruction; keeping training and
# evaluation prompts identical avoids teaching the teacher a format it is never asked for.
THINK_FIRST_LINE = "\nYou need to think first then write the Python code."

SCHEMA = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
    ("extra_info", pa.struct([("index", pa.int64()), ("split", pa.string())])),
])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_ground_truth(raw: str, store_tests: int) -> tuple[bool, str, str]:
    """Accept only the stdio ``inputs``/``outputs`` schema our judge understands.

    Keeps at most ``store_tests`` cases. The judge already truncates to its own max_tests, so
    the rest is dead weight, and a few problems ship enough of it that the whole column stops
    fitting in one Arrow chunk, which makes the parquet unreadable for verl's loader.
    """
    try:
        gt = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return False, "invalid_json", ""
    if not isinstance(gt, dict):
        return False, "not_object", ""
    if "fn_name" in gt:
        return False, "call_based", ""
    ins, outs = gt.get("inputs"), gt.get("outputs")
    if not isinstance(ins, list) or not isinstance(outs, list):
        return False, "missing_arrays", ""
    if not ins or len(ins) != len(outs):
        return False, "length_mismatch", ""
    trimmed = json.dumps({"inputs": ins[:store_tests], "outputs": outs[:store_tests]})
    return True, "ok", trimmed


def iter_code_rows(path: Path):
    f = pq.ParquetFile(path)
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg)
        cols = {name: t.column(name) for name in t.schema.names}
        for i, ability in enumerate(cols["ability"].to_pylist()):
            if ability != "code":
                continue
            yield {name: cols[name][i].as_py() for name in t.schema.names}


def build(split_path: Path, split_name: str, args, tokenizer) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    rows: list[dict] = []
    pending: list[dict] = []

    for row in iter_code_rows(split_path):
        stats["code_rows"] += 1
        ok, reason, ground_truth = check_ground_truth(
            row["reward_model"]["ground_truth"], args.store_tests
        )
        if not ok:
            stats[f"drop_{reason}"] += 1
            continue
        user = [m for m in row["prompt"] if m["role"] == "user"]
        if len(user) != 1:
            stats["drop_prompt_shape"] += 1
            continue
        content = user[0]["content"]
        if args.append_think_line and not content.endswith(THINK_FIRST_LINE):
            content += THINK_FIRST_LINE
        pending.append({
            "data_source": row["data_source"],
            "content": content,
            "ground_truth": ground_truth,
        })

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["content"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in pending
    ]
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]

    for p, n_tok in zip(pending, lengths):
        if n_tok > args.max_prompt_tokens:
            stats["drop_overlong_prompt"] += 1
            continue
        rows.append({
            "data_source": p["data_source"],
            "prompt": [{"content": p["content"], "role": "user"}],
            "ability": "code",
            "reward_model": {"ground_truth": p["ground_truth"], "style": "rule"},
            "extra_info": {"index": len(rows), "split": split_name},
        })
        stats["kept"] += 1
        stats["kept_tokens"] += n_tok
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eurus-dir", default="", help="local snapshot of the upstream dataset (default: download)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data/pools"))
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--store-tests", type=int, default=16,
                    help="tests kept per problem; the judge scores the first max_tests of them")
    ap.add_argument("--row-group-size", type=int, default=512)
    ap.add_argument("--append-think-line", action=argparse.BooleanOptionalAction, default=True,
                    help="match the code eval template exactly (used for the released data)")
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.eurus_dir:
        eurus = Path(args.eurus_dir)
    else:
        from huggingface_hub import snapshot_download

        eurus = Path(snapshot_download(EURUS_REPO, repo_type="dataset", revision=EURUS_REVISION,
                                       allow_patterns=["train.parquet", "validation.parquet"]))

    for split_file, split_name in (("train.parquet", "train"), ("validation.parquet", "validation")):
        path = eurus / split_file
        rows, stats = build(path, split_name, args, tokenizer)
        avg = stats["kept_tokens"] / stats["kept"] if stats["kept"] else 0
        print(f"=== {split_file} ===")
        print(f"  code rows {stats['code_rows']}, kept {stats['kept']}, mean prompt {avg:.0f} tokens")
        for k in sorted(k for k in stats if k.startswith("drop_")):
            print(f"    {k}: {stats[k]}")
        print(f"  sources: {dict(Counter(r['data_source'] for r in rows))}")

        if args.audit_only:
            continue

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if split_name == "validation":
            import random

            random.Random(args.seed).shuffle(rows)
            rows = rows[: args.val_size]
            for i, r in enumerate(rows):
                r["extra_info"]["index"] = i
        out_path = out_dir / f"{OUTPUT_NAMES[split_name]}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SCHEMA),
            out_path,
            row_group_size=args.row_group_size,
        )
        readable = pq.read_table(out_path).num_rows  # verl's loader chokes on chunked nesting
        print(f"  [write] {out_path} ({len(rows)} rows, {readable} read back)")

        manifest = out_dir / f"{out_path.stem}.manifest.json"
        manifest.write_text(json.dumps({
            "contract": "eurus_code_grpo_v1",
            "upstream": {
                "repo": "PRIME-RL/Eurus-2-RL-Data",
                "revision": EURUS_REVISION,
                "file": path.name,
                "sha256": sha256(path),
            },
            "filters": {
                "ability": "code",
                "ground_truth_schema": "stdio inputs/outputs, call-based dropped",
                "tests_stored_per_problem": args.store_tests,
                "max_prompt_tokens": args.max_prompt_tokens,
                "prompt_template": "qwen3 chat template, enable_thinking=False",
                "system_prompt": "PRIME action-format system message dropped",
                "append_think_line": args.append_think_line,
            },
            "rows": len(rows),
            "stats": dict(stats),
        }, indent=2, ensure_ascii=False))
        print(f"  [write] {manifest}")


if __name__ == "__main__":
    main()
