"""Score the generated shards of one OOD benchmark for one checkpoint.

Judging reuses the original graders so the numbers stay comparable with published ones:
HumanEval+ and MBPP+ go through the G-OPD evalplus fork (tree-sitter sanitiser plus
base and extra tests), LiveCodeBench through ``lcb_runner``'s last-code-fence extraction
and its sandboxed test runner, and GPQA through the R1 recipe's answer regex.

Reported per benchmark:
  mean@n  average fraction of the n samples that pass, averaged over tasks
          (this is the unbiased pass@1 estimate, matching how the math evals are read)
  hit@n   fraction of tasks solved by at least one of the n samples

Usage:
  python3 scripts/eval/ood_score.py --bench humaneval_plus \
      --gen-dir <dir with shard_*.jsonl> --label <model_label> --out <summary.json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ood_common import (
    BENCHMARKS,
    EXPECTED_TASK_COUNTS,
    GPQA_ANSWER_PATTERN,
    HUMANEVAL_PLUS_JSONL,
    LCB_RELEASE_VERSION,
    MBPP_PLUS_JSONL,
    add_gopd_paths,
    load_tasks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", required=True, choices=BENCHMARKS)
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--work-dir",
        help="directory for intermediate judge artifacts; defaults to --gen-dir",
    )
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--lcb-timeout", type=int, default=60, help="per-test timeout, G-OPD uses 60")
    p.add_argument(
        "--strip-think",
        action="store_true",
        help="score only the text after the last </think>; for thinking-mode runs",
    )
    p.add_argument(
        "--lcb-global-timeout",
        type=int,
        default=600,
        help="cap on judging one generation across all of its tests",
    )
    return p.parse_args()


def load_generations(gen_dir: Path, bench: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    shards = sorted(gen_dir.glob("shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"no shard_*.jsonl under {gen_dir}")
    for shard_file in shards:
        with shard_file.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["task_id"] in records:
                    raise SystemExit(f"duplicate task {rec['task_id']} across shards")
                records[rec["task_id"]] = rec
    expected = EXPECTED_TASK_COUNTS[bench]
    if len(records) != expected:
        raise SystemExit(f"{bench}: have {len(records)} tasks, expected {expected}")
    sizes = {len(r["outputs"]) for r in records.values()}
    if len(sizes) != 1:
        raise SystemExit(f"{bench}: inconsistent sample counts per task: {sorted(sizes)}")
    return records


THINK_CLOSE = "</think>"


def strip_think_blocks(records: dict[str, dict]) -> dict:
    """Keep only what a thinking model wrote after it stopped thinking.

    Scoring has to see the answer, not the reasoning: GPQA's pattern would otherwise match a
    letter the model tried and discarded mid-trace, and the code extractors would settle on a
    draft inside the think block.  A response with no closing tag never reached an answer --
    it spent its whole budget reasoning -- so it becomes empty and scores as a failure, which
    is the honest reading rather than a parse of its scratch work.
    """
    missing = 0
    total = 0
    for record in records.values():
        stripped = []
        for text in record["outputs"]:
            total += 1
            index = text.rfind(THINK_CLOSE)
            if index < 0:
                missing += 1
                stripped.append("")
            else:
                stripped.append(text[index + len(THINK_CLOSE) :].lstrip())
        record["outputs"] = stripped
    return {
        "stripped": True,
        "samples": total,
        "no_closing_tag_fraction": round(missing / total, 4) if total else 0.0,
    }


def generation_stats(records: dict[str, dict]) -> dict:
    lengths = [t for r in records.values() for t in r["output_tokens"]]
    truncated = sum(1 for r in records.values() for f in r["finish_reasons"] if f == "length")
    return {
        "avg_output_tokens": round(statistics.fmean(lengths), 1),
        "truncated_fraction": round(truncated / len(lengths), 4),
    }


def summarise(per_task_passes: dict[str, list[bool]]) -> dict:
    means = [sum(p) / len(p) for p in per_task_passes.values()]
    hits = [any(p) for p in per_task_passes.values()]
    n = len(next(iter(per_task_passes.values())))
    return {
        "n": n,
        "tasks": len(per_task_passes),
        f"mean@{n}": round(statistics.fmean(means), 4),
        f"hit@{n}": round(sum(hits) / len(hits), 4),
    }


# --------------------------------------------------------------------------------------
# GPQA
# --------------------------------------------------------------------------------------


# A model distilled on math carries the \boxed{} habit into GPQA and writes "$$\boxed{C}$$"
# where the R1 recipe expects "Answer: C".  The strict pattern scores those zero even when the
# letter is right, which cost the 30B-thinking students roughly 25 points and put one of them
# below the 25% random floor -- a shape no loss of knowledge can produce.  The strict number stays
# the headline so rows remain comparable to the published recipe; the permissive one separates a
# formatting shift from a capability change, and the gap between them is the size of that shift.
#
# Neither half carries an inline (?i): Python rejects a global flag that is not at the start of
# the expression, which an alternation of two (?i)-prefixed patterns necessarily produces.
GPQA_BOXED_PATTERN = r"\\boxed\{\s*(?:\\(?:text|mathrm|mathbf)\s*\{)?\s*\(?\$?([A-D])\b"


def score_gpqa(records: dict[str, dict]) -> tuple[dict, dict]:
    pattern = re.compile(GPQA_ANSWER_PATTERN)
    strict_body = GPQA_ANSWER_PATTERN.removeprefix("(?i)")
    permissive = re.compile(
        f"(?:{strict_body})|(?:{GPQA_BOXED_PATTERN})", re.IGNORECASE
    )
    per_task: dict[str, list[bool]] = {}
    per_task_last: dict[str, list[bool]] = {}
    per_task_permissive: dict[str, list[bool]] = {}
    unparsed = 0
    unparsed_permissive = 0
    total = 0
    for task_id, rec in records.items():
        gold = rec["meta"]["answer"]
        first_hits, last_hits, permissive_hits = [], [], []
        for text in rec["outputs"]:
            matches = pattern.findall(text)
            total += 1
            if not matches:
                unparsed += 1
            first_hits.append(bool(matches) and matches[0].upper() == gold)
            last_hits.append(bool(matches) and matches[-1].upper() == gold)
            # Alternation yields one group per branch, exactly one of which is non-empty.
            loose = ["".join(groups) for groups in permissive.findall(text)]
            if not loose:
                unparsed_permissive += 1
            permissive_hits.append(bool(loose) and loose[0].upper() == gold)
        per_task[task_id] = first_hits
        per_task_last[task_id] = last_hits
        per_task_permissive[task_id] = permissive_hits
    n = len(next(iter(per_task.values())))
    extra = {
        "unparsed_fraction": round(unparsed / total, 4),
        "mean_with_last_match": round(
            statistics.fmean([sum(p) / len(p) for p in per_task_last.values()]), 4
        ),
        "answer_pattern": GPQA_ANSWER_PATTERN,
        f"mean@{n}_any_format": round(
            statistics.fmean([sum(p) / len(p) for p in per_task_permissive.values()]), 4
        ),
        f"hit@{n}_any_format": round(
            sum(1 for p in per_task_permissive.values() if any(p)) / len(per_task_permissive), 4
        ),
        "unparsed_fraction_any_format": round(unparsed_permissive / total, 4),
    }
    return per_task, extra


# --------------------------------------------------------------------------------------
# EvalPlus: HumanEval+ and MBPP+
# --------------------------------------------------------------------------------------


# ``code_extract`` enumerates every contiguous line span and AST-parses each one,
# so its fallback cost is quadratic in line count.  EvalPlus tasks need only short
# functions; 120 lines still leaves a generous recovery window while bounding one
# malformed response to 7,140 candidate spans.
MAX_SANITIZE_LINES = 120
_CODE_FENCE = re.compile(r"```(?:[Pp]ython)?\n(.*?)```", re.DOTALL)


def _bounded_sanitize_candidate(
    text: str, entry_point: str
) -> tuple[str, bool]:
    """Select one likely answer block without EvalPlus's quadratic scan.

    EvalPlus's ``extract_target_code_or_empty`` unexpectedly calls ``code_extract``
    before its tree-sitter dependency pass.  ``code_extract`` enumerates every
    contiguous line span, so even calling the nominal extractor on the full response
    retains the quadratic bottleneck.  The benchmark prompt explicitly asks for final
    fenced code, so prefer the last fence containing the requested definition,
    otherwise the last fence, and cap only that candidate.
    """
    blocks = _CODE_FENCE.findall(text)
    matching_blocks = [
        block
        for block in blocks
        if re.search(
            rf"\b(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(",
            block,
        )
    ]
    candidate = (
        matching_blocks[-1]
        if matching_blocks
        else blocks[-1]
        if blocks
        else text
    )
    lines = candidate.split("\n")
    if len(lines) <= MAX_SANITIZE_LINES:
        return candidate, False

    definition_lines = [
        index
        for index, line in enumerate(lines)
        if re.search(
            rf"\b(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(",
            line,
        )
    ]
    if definition_lines:
        start = max(0, definition_lines[-1] - 20)
        start = min(start, len(lines) - MAX_SANITIZE_LINES)
    else:
        start = 0
    return "\n".join(lines[start : start + MAX_SANITIZE_LINES]), True


def _sanitize_one(job: tuple[str, str, str]) -> tuple[str, str, bool, bool]:
    task_id, text, entry_point = job
    import evalplus.sanitize as sanitize_module

    candidate, was_reduced = _bounded_sanitize_candidate(text, entry_point)
    # Keep EvalPlus's tree-sitter dependency extraction, but bind its internal
    # ``code_extract`` call to the deterministic candidate selected above.
    original_code_extract = sanitize_module.code_extract
    sanitize_module.code_extract = lambda _text: candidate
    try:
        recovered = sanitize_module.extract_target_code_or_empty(
            candidate, entry_point
        ).strip()
    finally:
        sanitize_module.code_extract = original_code_extract
    return task_id, recovered or candidate, was_reduced, True


def score_evalplus(
    records: dict[str, dict], out_dir: Path, workers: int, benchmark: str
) -> tuple[dict, dict]:
    if benchmark == "humaneval_plus":
        os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(HUMANEVAL_PLUS_JSONL)
        dataset = "humaneval"
    elif benchmark == "mbpp_plus":
        os.environ["MBPP_OVERRIDE_PATH"] = str(MBPP_PLUS_JSONL)
        dataset = "mbpp"
    else:
        raise ValueError(f"unsupported EvalPlus benchmark: {benchmark}")
    add_gopd_paths()
    from evalplus.evaluate import evaluate

    entry_points = {t.task_id: t.meta["entry_point"] for t in load_tasks(benchmark)}
    ordered_ids = sorted(records, key=lambda t: int(t.split("/")[1]))
    jobs = [
        (task_id, text, entry_points[task_id])
        for task_id in ordered_ids
        for text in records[task_id]["outputs"]
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sanitized = list(pool.map(_sanitize_one, jobs, chunksize=4))
    n_reduced = sum(
        1 for _, _, was_reduced, _ in sanitized if was_reduced
    )
    n_bounded = sum(
        1 for _, _, _, used_bounded in sanitized if used_bounded
    )

    samples_path = out_dir / "evalplus_samples.jsonl"
    with samples_path.open("w") as fh:
        for task_id, solution, _, _ in sanitized:
            fh.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")

    result_path = out_dir / "evalplus_eval_results.json"
    if result_path.exists():
        result_path.unlink()
    evaluate(
        dataset=dataset,
        samples=str(samples_path),
        base_only=False,
        parallel=workers,
        min_time_limit=10.0,
        gt_time_limit_factor=8.0,
        output_file=str(result_path),
    )

    results = json.loads(result_path.read_text())
    per_task: dict[str, list[bool]] = {}
    base_only_pass = 0
    total = 0
    for task_id, attempts in results["eval"].items():
        passes = []
        for attempt in attempts:
            base_ok = attempt["base_status"] == "pass"
            plus_ok = attempt["plus_status"] == "pass"
            passes.append(base_ok and plus_ok)
            base_only_pass += int(base_ok)
            total += 1
        per_task[task_id] = passes
    extra = {
        "base_tests_mean": round(base_only_pass / total, 4),
        "empty_solution_fraction": round(
            sum(
                1
                for _, solution, _, _ in sanitized
                if not solution.strip()
            )
            / total,
            4,
        ),
        "long_output_reduced_fraction": round(n_reduced / total, 4),
        "bounded_sanitize_fraction": round(n_bounded / total, 4),
        "sanitizer_mode": "last-relevant-fence-tree-sitter",
        "sanitize_candidate_max_lines": MAX_SANITIZE_LINES,
    }
    return per_task, extra


# --------------------------------------------------------------------------------------
# LiveCodeBench v6
# --------------------------------------------------------------------------------------


def _cap_lcb_global_timeout(max_seconds: int) -> None:
    """Bound how long a single generation may occupy the judge.

    Upstream derives its per-generation cap as ``(timeout + 1) * n_tests + 5``, which for a
    LiveCodeBench problem carrying a hundred private tests reaches hours, so one runaway
    generation stalls the whole benchmark. The per-test timeout is left untouched; only the
    overall wait is capped, and exceeding it counts as a failure exactly as upstream does.
    Also avoids upstream's IndexError on the metadata list when the cap does fire.
    """
    import json as _json
    import multiprocessing

    from lcb_runner.evaluation import compute_code_generation_metrics as metrics_mod

    def bounded_check_correctness(sample, generation, timeout, debug=True):
        manager = multiprocessing.Manager()
        result = manager.list()
        metadata_list = manager.list()
        proc = multiprocessing.Process(
            target=metrics_mod._temp_run,
            args=(sample, generation, debug, result, metadata_list, timeout),
        )
        proc.start()
        n_tests = len(_json.loads(sample["input_output"])["inputs"])
        proc.join(timeout=min((timeout + 1) * n_tests + 5, max_seconds))
        if proc.is_alive():
            proc.kill()
        if not result:
            return [-1] * n_tests, {
                "error": "global timeout",
                "error_code": -5,
                "error_message": "GlobalTimeout",
            }
        return result[0], metadata_list[0] if metadata_list else {}

    metrics_mod.check_correctness = bounded_check_correctness


def score_livecodebench(
    records: dict[str, dict], timeout: int, workers: int, global_timeout: int
) -> tuple[dict, dict]:
    add_gopd_paths()
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation import codegen_metrics
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    _cap_lcb_global_timeout(global_timeout)

    problems = load_code_generation_dataset(release_version=LCB_RELEASE_VERSION)
    problems = [p for p in problems if p.question_id in records]
    if len(problems) != len(records):
        raise SystemExit("LiveCodeBench task ids do not line up with the generations")

    samples_list = [p.get_evaluation_sample() for p in problems]
    generations_list = [
        [extract_code(text, LMStyle.Qwen3NonThinking) for text in records[p.question_id]["outputs"]]
        for p in problems
    ]
    empty = sum(1 for gens in generations_list for g in gens if not g.strip())
    total = sum(len(g) for g in generations_list)
    # The G-OPD prompt shows an example fence containing the words "Your code"; a model that
    # echoes it last defeats the last-fence extraction, so track how often that happens.
    echoed = sum(1 for gens in generations_list for g in gens if g.strip() == "Your code")

    _metrics, results, _metadata = codegen_metrics(
        samples_list,
        generations_list,
        k_list=[1],
        num_process_evaluate=workers,
        timeout=timeout,
    )

    per_task: dict[str, list[bool]] = {}
    for idx, problem in enumerate(problems):
        per_task[problem.question_id] = [
            bool(all(g > 0 for g in generation)) for generation in results[idx]
        ]
    extra = {
        "release_version": LCB_RELEASE_VERSION,
        "no_code_extracted_fraction": round(empty / total, 4),
        "placeholder_echo_fraction": round(echoed / total, 4),
        "test_timeout_seconds": timeout,
        "global_timeout_seconds": global_timeout,
    }
    return per_task, extra


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.gen_dir)
    work_dir = Path(args.work_dir) if args.work_dir else gen_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_generations(gen_dir, args.bench)
    think_stats = strip_think_blocks(records) if args.strip_think else None
    if args.bench == "gpqa_diamond":
        per_task, extra = score_gpqa(records)
    elif args.bench in {"humaneval_plus", "mbpp_plus"}:
        per_task, extra = score_evalplus(
            records, work_dir, args.workers, args.bench
        )
    else:
        per_task, extra = score_livecodebench(
            records, args.lcb_timeout, args.workers, args.lcb_global_timeout
        )

    summary = {
        "label": args.label,
        "benchmark": args.bench,
        **summarise(per_task),
        **generation_stats(records),
        "extra": extra,
    }
    if think_stats is not None:
        summary["think_block"] = think_stats
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    (work_dir / "per_task_passes.json").write_text(
        json.dumps(per_task, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
