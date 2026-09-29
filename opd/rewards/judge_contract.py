"""Stable parsing, comparison, and result semantics for code rewards."""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

JUDGE_CONTRACT_VERSION = "opd_code_judge_v1"
FLOAT_REL_TOL = Decimal("1e-6")
FLOAT_ABS_TOL = Decimal("1e-6")

RESULT_KEYS = (
    "judge_contract_version",
    "score",
    "acc",
    "all_pass",
    "format_ok",
    "passed",
    "total",
    "timeouts",
    "runtime_errors",
    "compile_errors",
    "output_limit_errors",
    "candidate_errors",
    "judge_errors",
    "status",
    "diagnostics",
    "results",
)

_FENCE_RE = re.compile(
    r"```[ \t]*([^\n`]*)\r?\n(.*?)```",
    flags=re.DOTALL,
)
_INTEGER_RE = re.compile(r"[+-]?\d+\Z")


class JudgeContractError(ValueError):
    """Invalid judge input rather than a candidate-program failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def extract_fenced_code(text: str) -> str | None:
    """Return the last Python fence, or otherwise the last Markdown fence."""
    if not isinstance(text, str):
        return None
    blocks = list(_FENCE_RE.finditer(text))
    if not blocks:
        return None
    python_blocks = [
        block
        for block in blocks
        if block.group(1).strip().lower() in {"python", "py", "python3"}
    ]
    selected = (python_blocks or blocks)[-1]
    return selected.group(2)


def parse_stdio_tests(
    ground_truth: str | Mapping[str, Any],
    *,
    max_tests: int = 10,
) -> tuple[list[Any], list[Any]]:
    """Parse and validate the canonical ``inputs``/``outputs`` stdio schema."""
    try:
        value = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (json.JSONDecodeError, TypeError) as error:
        raise JudgeContractError("invalid_json", "ground truth is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise JudgeContractError("invalid_schema", "ground truth must be an object")
    inputs = value.get("inputs")
    outputs = value.get("outputs")
    if (
        not isinstance(inputs, Sequence)
        or isinstance(inputs, (str, bytes, bytearray))
        or not isinstance(outputs, Sequence)
        or isinstance(outputs, (str, bytes, bytearray))
    ):
        raise JudgeContractError(
            "invalid_schema",
            "ground truth inputs and outputs must be arrays",
        )
    if max_tests < 1:
        raise JudgeContractError("invalid_max_tests", "max_tests must be positive")
    parsed_inputs = list(inputs)[:max_tests]
    parsed_outputs = list(outputs)[:max_tests]
    if not parsed_inputs:
        raise JudgeContractError("empty_tests", "ground truth has no tests")
    if len(parsed_inputs) != len(parsed_outputs):
        raise JudgeContractError(
            "length_mismatch",
            "ground truth inputs and outputs have different lengths",
        )
    return parsed_inputs, parsed_outputs


def _numeric_tokens_equivalent(actual: str, expected: str) -> bool:
    if _INTEGER_RE.fullmatch(actual) and _INTEGER_RE.fullmatch(expected):
        def normalize_integer(token: str) -> tuple[bool, str]:
            negative = token.startswith("-")
            digits = token.lstrip("+-").lstrip("0") or "0"
            return (negative and digits != "0"), digits

        return normalize_integer(actual) == normalize_integer(expected)
    try:
        actual_number = Decimal(actual)
        expected_number = Decimal(expected)
    except InvalidOperation:
        return False
    if not actual_number.is_finite() or not expected_number.is_finite():
        return False
    difference = abs(actual_number - expected_number)
    scale = max(abs(actual_number), abs(expected_number))
    return difference <= max(FLOAT_ABS_TOL, FLOAT_REL_TOL * scale)


def outputs_equivalent(actual: str, expected: str) -> bool:
    """Compare ordered whitespace tokens; integers exact, floats tolerant."""
    actual_tokens = str(actual).strip().split()
    expected_tokens = str(expected).strip().split()
    if actual_tokens == expected_tokens:
        return True
    if len(actual_tokens) != len(expected_tokens):
        return False
    return all(
        _numeric_tokens_equivalent(left, right)
        for left, right in zip(actual_tokens, expected_tokens)
    )


def make_result(
    *,
    format_ok: float = 1.0,
    passed: int = 0,
    total: int = 0,
    timeouts: int = 0,
    runtime_errors: int = 0,
    compile_errors: int = 0,
    output_limit_errors: int = 0,
    candidate_errors: int | None = None,
    judge_errors: int = 0,
    status: str | None = None,
    diagnostics: Sequence[str] | None = None,
    results: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build the sole public result schema while preserving ``K / N``."""
    passed = max(0, int(passed))
    total = max(0, int(total))
    passed = min(passed, total)
    timeouts = max(0, int(timeouts))
    runtime_errors = max(0, int(runtime_errors))
    compile_errors = max(0, int(compile_errors))
    output_limit_errors = max(0, int(output_limit_errors))
    judge_errors = max(0, int(judge_errors))
    if candidate_errors is None:
        candidate_errors = (
            timeouts + runtime_errors + compile_errors + output_limit_errors
        )
    candidate_errors = max(0, int(candidate_errors))
    score = passed / total if total else 0.0
    if status is None:
        if judge_errors:
            status = "judge_error"
        elif not format_ok:
            status = "format_error"
        elif candidate_errors:
            status = "candidate_error"
        else:
            status = "ok"
    if isinstance(diagnostics, str):
        diagnostics = [diagnostics]
    if isinstance(results, (str, bytes, bytearray)):
        results = [results]
    return {
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "score": float(score),
        "acc": float(score),
        "all_pass": float(total > 0 and passed == total),
        "format_ok": float(format_ok),
        "passed": passed,
        "total": total,
        "timeouts": timeouts,
        "runtime_errors": runtime_errors,
        "compile_errors": compile_errors,
        "output_limit_errors": output_limit_errors,
        "candidate_errors": candidate_errors,
        "judge_errors": judge_errors,
        "status": str(status),
        "diagnostics": [str(item) for item in (diagnostics or [])],
        "results": list(results or []),
    }
