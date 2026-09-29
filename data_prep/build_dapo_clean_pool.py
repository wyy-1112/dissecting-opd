#!/usr/bin/env python3
"""Freeze a deduplicated, decontaminated DAPO-Math pool in this repo's OPD schema.

The released DAPO-Math-17k parquet holds 1,791,700 rows because every problem is
repeated for the rollout schedule, and it wraps each problem in DAPO's own
"Answer: $Answer" instruction.  JustRL, however, was RL-trained on these problems under
the boxed instruction (its model card's usage block, and its 1.5-4.4 percent
missing-boxed rate against our base student's 10-17.5 percent).  So the pool has to be
unwrapped and re-templated onto the boxed suffix this repo already uses everywhere else,
which also keeps the DAPO arms template-identical to the existing DeepMath arms and
leaves problem provenance as the only difference between them.

Pipeline: unwrap -> normalize -> text dedup -> drop answer-conflicting statements ->
benchmark decontamination -> tokenized-prefix-hash dedup -> freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

# Directory laid out like the original project (data/opd, data/rl, results/...).
ROOT = Path(os.environ.get("OPD_PROJECT_ROOT", Path(__file__).resolve().parents[1] / "outputs" / "opd_project_root"))

SCHEMA = "dapo_math_clean_pool_v1"

# The canonical instruction used by every math pool, validation set, and eval protocol in
# this repo.  Held identical to the DeepMath L6 pool on purpose.
PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

BENCHMARKS = ("aime24", "aime25", "hmmt25_feb", "hmmt25_nov")
BENCHMARK_DIR = ROOT / "data/rl/gopd_math_val"

NGRAM = 8
CONTAINMENT_THRESHOLD = 0.5

WHITESPACE = re.compile(r"\s+")
NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_text(text: str) -> str:
    """Whitespace- and unicode-canonical form; case and symbols are preserved.

    Math statements are case sensitive (variables) and symbol sensitive, so this stays
    conservative and is used as the primary dedup key.
    """
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def loose_key(text: str) -> str:
    """Aggressive alphanumeric form, used only where recall matters (contamination)."""
    return NON_ALNUM.sub("", unicodedata.normalize("NFKC", text).lower())


def normalize_answer(answer: str) -> str:
    value = WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(answer))).strip()
    while len(value) > 1 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return value


def word_ngrams(text: str, size: int = NGRAM) -> set[str]:
    words = loose_key_words(text)
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def loose_key_words(text: str) -> list[str]:
    lowered = unicodedata.normalize("NFKC", text).lower()
    return [token for token in NON_ALNUM.split(lowered) if token]


def unwrap(content: str) -> tuple[str, str]:
    """Return (bare problem, how it was unwrapped)."""
    text = content
    had_prefix = text.startswith(DAPO_PREFIX)
    if had_prefix:
        text = text[len(DAPO_PREFIX) :]
    had_suffix = text.endswith(DAPO_SUFFIX)
    if had_suffix:
        text = text[: -len(DAPO_SUFFIX)]
    if had_prefix and had_suffix:
        mode = "both"
    elif had_prefix or had_suffix:
        mode = "partial"
    else:
        mode = "neither"
    return text.strip(), mode


def iter_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    handle = pq.ParquetFile(path)
    columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    for batch in handle.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def load_benchmarks() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name in BENCHMARKS:
        path = BENCHMARK_DIR / f"{name}.parquet"
        rows = pq.read_table(path).to_pylist()
        items = []
        for row in rows:
            content = row["prompt"][0]["content"]
            if content.endswith(PROMPT_SUFFIX):
                content = content[: -len(PROMPT_SUFFIX)]
            items.append(
                {
                    "id": str(row.get("id", row["extra_info"].get("index"))),
                    "problem": content.strip(),
                    "answer": normalize_answer(row["reward_model"]["ground_truth"]),
                }
            )
        out[name] = items
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/path/to/code/"
            "G-OPD-Training-Data/DAPO-Math-17k/dapo-math-17k.parquet"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT
        / "data/models/opd_deepseek_r1_justrl_1p5b/student_deepseek_r1_distill_qwen_1p5b",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/opd/dapo_math_clean_v1"
    )
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--allow-overwrite", action="store_true")
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists() and not arguments.allow_overwrite:
        raise FileExistsError(f"refusing to overwrite frozen pool: {output_dir}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(arguments.tokenizer))
    template_hash = hashlib.sha256(
        (tokenizer.chat_template or "").encode("utf-8")
    ).hexdigest()

    def tokenized_prefix_hash(problem: str) -> tuple[str, int]:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem + PROMPT_SUFFIX}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
        payload = ",".join(str(value) for value in ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), len(ids)

    # ---- stage 1: unwrap and collapse the released repetitions -------------------
    groups: dict[str, dict[str, Any]] = {}
    unwrap_modes: Counter[str] = Counter()
    data_sources: Counter[str] = Counter()
    abilities: Counter[str] = Counter()
    reward_styles: Counter[str] = Counter()
    total_rows = 0
    multi_turn_rows = 0

    for row in iter_rows(source, arguments.batch_size):
        total_rows += 1
        prompt = row["prompt"]
        if len(prompt) != 1 or prompt[0].get("role") != "user":
            multi_turn_rows += 1
            continue
        problem, mode = unwrap(prompt[0]["content"])
        unwrap_modes[mode] += 1
        data_sources[row.get("data_source")] += 1
        abilities[row.get("ability")] += 1
        reward_styles[row["reward_model"].get("style")] += 1
        answer = normalize_answer(row["reward_model"]["ground_truth"])
        key = normalize_text(problem)
        entry = groups.get(key)
        if entry is None:
            groups[key] = {
                "problem": problem,
                "answers": {answer},
                "rows": 1,
                "source_ids": {str(row["extra_info"].get("index"))},
            }
        else:
            entry["answers"].add(answer)
            entry["rows"] += 1
            if len(entry["source_ids"]) < 8:
                entry["source_ids"].add(str(row["extra_info"].get("index")))

    print(f"rows read                     {total_rows:>9,d}")
    print(f"  non single-turn (skipped)   {multi_turn_rows:>9,d}")
    print(f"  unwrap modes               {dict(unwrap_modes)}")
    print(f"  data_source                {dict(data_sources)}")
    print(f"  ability                    {dict(abilities)}")
    print(f"  reward style               {dict(reward_styles)}")
    print(f"unique problems after text dedup {len(groups):>6,d}")

    # ---- stage 2: drop statements whose duplicates disagree on the answer --------
    conflicts = []
    unique: dict[str, dict[str, Any]] = {}
    for key, entry in groups.items():
        if len(entry["answers"]) > 1:
            conflicts.append(
                {
                    "problem": entry["problem"][:600],
                    "answers": sorted(entry["answers"]),
                    "rows": entry["rows"],
                    "source_ids": sorted(entry["source_ids"]),
                }
            )
            continue
        entry["answer"] = next(iter(entry["answers"]))
        unique[key] = entry
    print(f"  answer-conflicting statements dropped {len(conflicts):>5,d}")
    print(f"survivors                     {len(unique):>9,d}")

    # ---- stage 3: benchmark decontamination -------------------------------------
    benchmarks = load_benchmarks()
    bench_exact: dict[str, tuple[str, str]] = {}
    bench_loose: dict[str, tuple[str, str]] = {}
    bench_grams: list[tuple[str, str, set[str], str]] = []
    for name, items in benchmarks.items():
        for item in items:
            bench_exact[normalize_text(item["problem"])] = (name, item["id"])
            bench_loose[loose_key(item["problem"])] = (name, item["id"])
            bench_grams.append(
                (name, item["id"], word_ngrams(item["problem"]), item["answer"])
            )
    print(
        f"benchmark items loaded        "
        f"{sum(len(v) for v in benchmarks.values()):>9,d} "
        f"({', '.join(f'{k}={len(v)}' for k, v in benchmarks.items())})"
    )

    # Invert the benchmark n-grams once so each pool problem is a single set lookup
    # rather than a scan over all benchmark items.
    gram_index: dict[str, set[int]] = defaultdict(set)
    for position, (_, _, grams, _) in enumerate(bench_grams):
        for gram in grams:
            gram_index[gram].add(position)

    contaminated = []
    clean: dict[str, dict[str, Any]] = {}
    for key, entry in unique.items():
        problem = entry["problem"]
        reason = None
        match = None
        if key in bench_exact:
            reason, match = "exact_text", bench_exact[key]
        elif loose_key(problem) in bench_loose:
            reason, match = "loose_text", bench_loose[loose_key(problem)]
        else:
            grams = word_ngrams(problem)
            hits: Counter[int] = Counter()
            for gram in grams:
                for position in gram_index.get(gram, ()):
                    hits[position] += 1
            for position, shared in hits.most_common(4):
                name, item_id, bench_gram_set, _ = bench_grams[position]
                if not bench_gram_set:
                    continue
                bench_containment = shared / len(bench_gram_set)
                pool_containment = shared / max(len(grams), 1)
                if max(bench_containment, pool_containment) >= CONTAINMENT_THRESHOLD:
                    reason = f"ngram_containment_{max(bench_containment, pool_containment):.2f}"
                    match = (name, item_id)
                    break
        if reason is not None:
            contaminated.append(
                {
                    "problem": problem[:600],
                    "answer": entry["answer"],
                    "reason": reason,
                    "benchmark": match[0],
                    "benchmark_id": match[1],
                }
            )
            continue
        clean[key] = entry
    print(f"  benchmark-contaminated dropped        {len(contaminated):>5,d}")
    print(f"survivors                     {len(clean):>9,d}")

    # ---- stage 4: tokenized prefix hash, and a second dedup pass on it ----------
    by_prefix: dict[str, str] = {}
    prefix_collisions = []
    final: list[dict[str, Any]] = []
    for key in sorted(clean):
        entry = clean[key]
        digest, token_count = tokenized_prefix_hash(entry["problem"])
        if digest in by_prefix:
            prefix_collisions.append(
                {
                    "kept": by_prefix[digest][:300],
                    "dropped": entry["problem"][:300],
                    "tokenized_prefix_sha256": digest,
                }
            )
            continue
        by_prefix[digest] = entry["problem"]
        final.append(
            {
                "problem": entry["problem"],
                "answer": entry["answer"],
                "tokenized_prefix_sha256": digest,
                "prompt_token_count": token_count,
                "released_rows": entry["rows"],
                "source_ids": sorted(entry["source_ids"]),
            }
        )
    print(f"  tokenized-prefix collisions dropped   {len(prefix_collisions):>5,d}")
    print(f"FINAL POOL                    {len(final):>9,d}")

    # ---- freeze ----------------------------------------------------------------
    import pandas as pd

    rows = [
        {
            "data_source": "DAPO-Math-17k",
            "prompt": [
                {"content": item["problem"] + PROMPT_SUFFIX, "role": "user"}
            ],
            "ability": "math",
            "reward_model": {"ground_truth": item["answer"], "style": "rule"},
            "extra_info": {"index": position, "split": "train"},
        }
        for position, item in enumerate(final)
    ]
    frame = pd.DataFrame(
        rows, columns=["data_source", "prompt", "ability", "reward_model", "extra_info"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "pool.parquet"
    temporary = pool_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, row_group_size=512)
    os.replace(temporary, pool_path)

    index_path = output_dir / "pool_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for position, item in enumerate(final):
            handle.write(
                json.dumps(
                    {
                        "index": position,
                        "tokenized_prefix_sha256": item["tokenized_prefix_sha256"],
                        "prompt_token_count": item["prompt_token_count"],
                        "answer": item["answer"],
                        "released_rows": item["released_rows"],
                        "source_ids": item["source_ids"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    audit = {
        "schema_version": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "rows": total_rows,
            "sha256": sha256_file(source),
        },
        "template": {
            "prompt_suffix": PROMPT_SUFFIX,
            "chat_template_sha256": template_hash,
            "tokenizer": str(arguments.tokenizer),
            "justrl_template_evidence": [
                "model card usage block renders '<problem>\\n\\nPlease reason step by "
                "step, and put your final answer within \\\\boxed{}.'",
                "teacher missing_boxed_rate 0.0146-0.0437 vs base student 0.1021-0.1750 "
                "under this repo's boxed eval protocol",
            ],
            "note": (
                "DAPO's released wrapper asks for 'Answer: $Answer' and is stripped; the "
                "boxed suffix here is identical to the DeepMath L6 pool so that the DAPO "
                "arms differ from the existing arms only in problem provenance"
            ),
        },
        "stages": {
            "rows_read": total_rows,
            "non_single_turn_skipped": multi_turn_rows,
            "unwrap_modes": dict(unwrap_modes),
            "unique_after_text_dedup": len(groups),
            "answer_conflicts_dropped": len(conflicts),
            "benchmark_contaminated_dropped": len(contaminated),
            "tokenized_prefix_collisions_dropped": len(prefix_collisions),
            "final_pool": len(final),
        },
        "decontamination": {
            "benchmarks": {k: len(v) for k, v in benchmarks.items()},
            "ngram_words": NGRAM,
            "containment_threshold": CONTAINMENT_THRESHOLD,
            "matches": contaminated,
        },
        "answer_conflicts": conflicts[:200],
        "answer_conflicts_truncated": len(conflicts) > 200,
        "tokenized_prefix_collisions": prefix_collisions[:200],
        "source_metadata": {
            "data_source": dict(data_sources),
            "ability": dict(abilities),
            "reward_style": dict(reward_styles),
        },
        "outputs": {
            "pool": {
                "path": str(pool_path),
                "rows": len(frame),
                "sha256": sha256_file(pool_path),
            },
            "pool_index": {
                "path": str(index_path),
                "sha256": sha256_file(index_path),
            },
        },
    }
    audit_path = output_dir / "audit.json"
    atomic_json(audit_path, audit)
    atomic_json(
        output_dir / "POOL_VALIDATED.json",
        {
            "schema_version": SCHEMA,
            "status": "validated",
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "pool": str(pool_path),
            "pool_rows": len(frame),
            "pool_sha256": audit["outputs"]["pool"]["sha256"],
        },
    )
    print(f"\nwrote {pool_path} ({len(frame):,d} rows)")
    print(f"wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
