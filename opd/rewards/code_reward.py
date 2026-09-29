#!/usr/bin/env python3
"""Local CPU execution reward for direct-code GRPO."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from judge_contract import (  # noqa: E402
    JudgeContractError,
    extract_fenced_code,
    make_result,
    parse_stdio_tests,
)


def empty_result(
    *,
    format_ok: float,
    total: int = 0,
    judge_errors: int = 0,
    candidate_errors: int = 0,
    status: str | None = None,
    diagnostics: list[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible helper that emits the shared result schema."""
    return make_result(
        format_ok=format_ok,
        total=total,
        judge_errors=judge_errors,
        candidate_errors=candidate_errors,
        status=status,
        diagnostics=diagnostics,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    timeout: float = 3.0,
    max_tests: int = 10,
    memory_limit_mb: int = 1024,
    file_limit_mb: int = 16,
    output_limit_bytes: int = 1024 * 1024,
    process_limit: int = 8,
) -> dict[str, Any]:
    del data_source, extra_info
    code = extract_fenced_code(solution_str)
    if code is None:
        return empty_result(
            format_ok=0.0,
            candidate_errors=1,
            status="format_error",
            diagnostics=["missing_fenced_code"],
        )
    try:
        inputs, outputs = parse_stdio_tests(ground_truth, max_tests=max_tests)
    except JudgeContractError as error:
        return empty_result(
            format_ok=1.0,
            judge_errors=1,
            status="judge_error",
            diagnostics=[f"{error.code}: {error.detail}"],
        )

    total = len(inputs)
    try:
        payload = json.dumps(
            {
                "code": code,
                "inputs": inputs,
                "outputs": outputs,
                "timeout": timeout,
                "memory_limit_mb": memory_limit_mb,
                "file_limit_mb": file_limit_mb,
                "output_limit": output_limit_bytes,
                "process_limit": process_limit,
            }
        )
    except (TypeError, ValueError):
        return empty_result(
            format_ok=1.0,
            total=total,
            judge_errors=1,
            status="judge_error",
            diagnostics=["ground_truth_not_json_serializable"],
        )
    judge = Path(__file__).with_name("local_code_judge.py")
    with tempfile.TemporaryDirectory(prefix="opd-grpo-") as directory:
        stdout_path = Path(directory) / "judge.stdout"
        stderr_path = Path(directory) / "judge.stderr"
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-I", str(judge)],
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                cwd=directory,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONHASHSEED": "0",
                },
                start_new_session=True,
            )
            try:
                process.communicate(payload, timeout=timeout * total + 5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    process.kill()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    # SIGKILL normally makes wait return immediately. A second
                    # bounded attempt handles races without introducing another
                    # unbounded validation wait (for example, a D-state child).
                    process.kill()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
                return empty_result(
                    format_ok=1.0,
                    total=total,
                    judge_errors=1,
                    status="judge_error",
                    diagnostics=["judge_process_timeout"],
                )
        if process.returncode != 0:
            return empty_result(
                format_ok=1.0,
                total=total,
                judge_errors=1,
                status="judge_error",
                diagnostics=[f"judge_process_exit_{process.returncode}"],
            )
        try:
            summary = json.loads(stdout_path.read_text()[-65536:])
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return empty_result(
                format_ok=1.0,
                total=total,
                judge_errors=1,
                status="judge_error",
                diagnostics=["invalid_judge_response"],
            )

    if not isinstance(summary, dict) or int(summary.get("total", total)) != total:
        return empty_result(
            format_ok=1.0,
            total=total,
            judge_errors=1,
            status="judge_error",
            diagnostics=["judge_total_mismatch"],
        )
    return make_result(
        format_ok=1.0,
        passed=int(summary.get("passed", 0)),
        total=total,
        timeouts=int(summary.get("timeouts", 0)),
        runtime_errors=int(summary.get("runtime_errors", 0)),
        compile_errors=int(summary.get("compile_errors", 0)),
        output_limit_errors=int(summary.get("output_limit_errors", 0)),
        candidate_errors=summary.get("candidate_errors"),
        judge_errors=int(summary.get("judge_errors", 0)),
        status=str(summary.get("status") or "ok"),
        diagnostics=summary.get("diagnostics") or [],
        results=summary.get("results") or [],
    )


def compute_score_compact(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return scalar-only metadata that Ray can concatenate across samples."""

    result = compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    return {
        key: value
        for key, value in result.items()
        if value is None or isinstance(value, (bool, int, float, str))
    }
