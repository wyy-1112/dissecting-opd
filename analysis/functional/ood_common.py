"""Shared pieces for the OOD generalisation eval.

The code benchmarks reuse the prompt strings of the G-OPD paper's released eval
harness byte for byte (``code_eval/coding/evalplus`` and
``code_eval/coding/LiveCodeBench``), so numbers are comparable with that paper.
GPQA has no counterpart there, so it uses the R1 multiple-choice template that ships
with the verl recipe vendored in the same tree (``verl/recipe/r1/data_process.py``),
which is also what the public GPQA-Diamond numbers are produced with.

Every prompt goes through the evaluated model's own chat template with
``enable_thinking=False``, matching how the student was trained and how the math
benchmarks were scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
# G-OPD checkout (RUCBM/G-OPD@37371a4) providing the evalplus / LiveCodeBench graders.
GOPD_ROOT = Path(os.environ.get("GOPD_ROOT", PROJ / "external" / "G-OPD"))

EVALPLUS_SRC = GOPD_ROOT / "code_eval/coding/evalplus"
LCB_SRC = GOPD_ROOT / "code_eval/coding/LiveCodeBench"
HUMANEVAL_PLUS_JSONL = GOPD_ROOT / "code_eval/data/HumanEvalPlus.jsonl"
MBPP_PLUS_JSONL = GOPD_ROOT / "code_eval/data/MbppPlus.jsonl"

# The G-OPD script passes --release_version v6, which in this loader means the
# incremental v6 slice (test6.jsonl only, contests after release_v5), not the
# cumulative 1055-problem release_v6.
LCB_DATA_DIR = LCB_SRC / "code_generation_lite"
LCB_RELEASE_VERSION = "v6"

GPQA_JSONL = PROJ / "data/eval/gpqa/gpqa_diamond.jsonl"  # PROJ = release root

BENCHMARKS = ("gpqa_diamond", "humaneval_plus", "mbpp_plus", "livecodebench_v6")

EXPECTED_TASK_COUNTS = {
    "gpqa_diamond": 198,
    "humaneval_plus": 164,
    "mbpp_plus": 378,
    "livecodebench_v6": 175,
}


def add_gopd_paths() -> None:
    """Make the G-OPD forks of evalplus and lcb_runner importable."""
    for src in (EVALPLUS_SRC, LCB_SRC):
        path = str(src)
        if path not in sys.path:
            sys.path.insert(0, path)


# --------------------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------------------

# Verbatim from G-OPD code_eval; shared by both code benchmarks. In the original the
# leading "\n\n" is part of the concatenation, kept explicit at each call site below.
GOPD_CODE_INSTRUCTION = (
    "Write Python code to solve the problem. Present the code in \n"
    "```python\nYour code\n```\n"
    "at the end.\n"
    "You need to think first then write the Python code."
)

# Verbatim from lcb_runner/prompts/code_generation.py, LMStyle.Qwen3NonThinking branch.
_LCB_PREFIX = (
    "You will be given a question (problem specification) and will generate a correct "
    "Python program that matches the specification and passes all tests. You will NOT "
    "return anything except for the program.\n\n"
)

# Verbatim from verl/recipe/r1/data_process.py.
_GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)

# Verbatim from verl/recipe/r1/tasks/gpqa.py (originally OpenAI simple-evals).
GPQA_ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def livecodebench_user_content(question_content: str) -> str:
    return _LCB_PREFIX + f"Question:\n{question_content}\n\n" + "\n\n" + GOPD_CODE_INSTRUCTION


def evalplus_user_content(task_prompt: str) -> str:
    # evalplus/codegen.py feeds `task["prompt"].strip() + "\n"` into make_raw_chat_prompt,
    # which then appends the instruction after a blank line.
    return task_prompt.strip() + "\n" + "\n\n" + GOPD_CODE_INSTRUCTION


def gpqa_user_content(question: str, choices: dict) -> str:
    return _GPQA_QUERY_TEMPLATE.format(
        Question=question, A=choices["A"], B=choices["B"], C=choices["C"], D=choices["D"]
    )


# --------------------------------------------------------------------------------------
# Task loading
# --------------------------------------------------------------------------------------


@dataclass
class Task:
    task_id: str
    user_content: str
    meta: dict


def _load_gpqa() -> list[Task]:
    tasks = []
    with GPQA_JSONL.open() as fh:
        for line in fh:
            rec = json.loads(line)
            tasks.append(
                Task(
                    task_id=rec["task_id"],
                    user_content=gpqa_user_content(rec["question"], rec["choices"]),
                    meta={"answer": rec["answer"], "domain": rec.get("domain")},
                )
            )
    return tasks


def _load_humaneval_plus() -> list[Task]:
    tasks = []
    with HUMANEVAL_PLUS_JSONL.open() as fh:
        for line in fh:
            rec = json.loads(line)
            tasks.append(
                Task(
                    task_id=rec["task_id"],
                    user_content=evalplus_user_content(rec["prompt"]),
                    meta={"entry_point": rec["entry_point"]},
                )
            )
    return tasks


def _load_mbpp_plus() -> list[Task]:
    tasks = []
    with MBPP_PLUS_JSONL.open() as fh:
        for line in fh:
            rec = json.loads(line)
            tasks.append(
                Task(
                    task_id=rec["task_id"],
                    user_content=evalplus_user_content(rec["prompt"]),
                    meta={"entry_point": rec["entry_point"]},
                )
            )
    return tasks


def _load_livecodebench_v6() -> list[Task]:
    add_gopd_paths()
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    problems = load_code_generation_dataset(release_version=LCB_RELEASE_VERSION)
    return [
        Task(
            task_id=p.question_id,
            user_content=livecodebench_user_content(p.question_content),
            meta={"platform": str(p.platform), "difficulty": str(p.difficulty)},
        )
        for p in problems
    ]


_LOADERS = {
    "gpqa_diamond": _load_gpqa,
    "humaneval_plus": _load_humaneval_plus,
    "mbpp_plus": _load_mbpp_plus,
    "livecodebench_v6": _load_livecodebench_v6,
}


def load_tasks(bench: str) -> list[Task]:
    if bench not in _LOADERS:
        raise ValueError(f"unknown benchmark {bench!r}; expected one of {BENCHMARKS}")
    tasks = _LOADERS[bench]()
    expected = EXPECTED_TASK_COUNTS[bench]
    if len(tasks) != expected:
        raise RuntimeError(f"{bench}: loaded {len(tasks)} tasks, expected {expected}")
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{bench}: duplicate task ids")
    return tasks


def shard(tasks: list[Task], shard_id: int, num_shards: int) -> list[Task]:
    """Round-robin slice, so every shard gets a similar mix of easy and hard problems."""
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"bad shard {shard_id}/{num_shards}")
    return tasks[shard_id::num_shards]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_prompt(tokenizer, user_content: str, *, enable_thinking: bool = False) -> str:
    """Apply the model's own chat template.

    Non-thinking is the default because that is how every student here was trained and how
    the math benchmarks were scored.  Thinking mode exists for the stock-model baselines,
    where Qwen's own recommendation is to let the model reason.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
