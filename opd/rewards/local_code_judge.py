#!/usr/bin/env python3
"""Single-process judge for up to ten stdin/stdout Python test cases."""
from __future__ import annotations

import builtins
import io
import json
import os
import resource
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from judge_contract import (  # noqa: E402
    JudgeContractError,
    make_result,
    outputs_equivalent,
    parse_stdio_tests,
)


class CaseTimeout(BaseException):
    pass


class OutputLimitExceeded(Exception):
    pass


class CappedWriter(io.TextIOBase):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0

    @property
    def buffer(self):
        return self

    def write(self, value) -> int:
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        encoded_size = len(text.encode("utf-8"))
        if self.size + encoded_size > self.limit:
            raise OutputLimitExceeded
        self.parts.append(text)
        self.size += encoded_size
        return len(value)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return "".join(self.parts)


def set_limits(memory_limit_mb: int, file_limit_mb: int, process_limit: int) -> None:
    memory = memory_limit_mb * 1024 * 1024
    file_size = file_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (process_limit, process_limit))
    except (ValueError, OSError):
        pass


def install_reliability_guard() -> None:
    def forbidden(*_args, **_kwargs):
        raise PermissionError("operation disabled by local judge")

    for name in (
        "system",
        "popen",
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "renames",
        "replace",
        "chdir",
        "chmod",
        "chown",
        "write",
    ):
        if hasattr(os, name):
            setattr(os, name, forbidden)
    builtins.open = forbidden
    for module_name in ("subprocess", "ctypes", "socket", "signal"):
        sys.modules[module_name] = None


def timeout_handler(_signum, _frame):
    raise CaseTimeout


def run_case(compiled, stdin: str, timeout: float, output_limit: int) -> tuple[str, str]:
    old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
    old_dunder_stdout, old_dunder_stderr = sys.__stdout__, sys.__stderr__
    stdout = CappedWriter(output_limit)
    stderr = CappedWriter(output_limit)
    sys.stdin = io.StringIO(stdin)
    sys.stdout = stdout
    sys.stderr = stderr
    sys.__stdout__ = stdout
    sys.__stderr__ = stderr
    # Candidate code may have changed the handler in an earlier test case.
    # Reinstall it before every execution so each case starts from the same
    # timeout contract.
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    status = "ok"
    try:
        namespace: dict[str, Any] = {"__name__": "__main__"}
        exec(compiled, namespace, namespace)
    except SystemExit as error:
        if error.code not in (None, 0):
            status = "runtime_error"
    except CaseTimeout:
        status = "timeout"
    except OutputLimitExceeded:
        status = "output_limit"
    except BaseException:
        status = "runtime_error"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        sys.stdin, sys.stdout, sys.stderr = old_stdin, old_stdout, old_stderr
        sys.__stdout__, sys.__stderr__ = old_dunder_stdout, old_dunder_stderr
    return status, stdout.getvalue()


def main() -> None:
    payload = json.loads(sys.stdin.read())
    code = str(payload["code"])
    try:
        parsed_inputs, parsed_outputs = parse_stdio_tests(
            {"inputs": payload["inputs"], "outputs": payload["outputs"]},
            max_tests=max(1, len(payload["inputs"])),
        )
    except (JudgeContractError, KeyError, TypeError) as error:
        detail = error.code if isinstance(error, JudgeContractError) else "invalid_payload"
        print(
            json.dumps(
                make_result(
                    judge_errors=1,
                    status="judge_error",
                    diagnostics=[detail],
                )
            )
        )
        return
    inputs = [str(value) for value in parsed_inputs]
    outputs = [str(value) for value in parsed_outputs]
    timeout = float(payload.get("timeout", 3.0))
    output_limit = int(payload.get("output_limit", 1024 * 1024))
    set_limits(
        int(payload.get("memory_limit_mb", 1024)),
        int(payload.get("file_limit_mb", 16)),
        int(payload.get("process_limit", 8)),
    )
    try:
        compiled = compile(code, "<candidate>", "exec")
    except BaseException:
        print(
            json.dumps(
                make_result(
                    total=len(inputs),
                    compile_errors=1,
                    candidate_errors=1,
                    status="candidate_error",
                    diagnostics=["compile_error"],
                    results=["compile_error"] * len(inputs),
                )
            )
        )
        return
    install_reliability_guard()
    signal.signal(signal.SIGALRM, timeout_handler)
    counts = Counter()
    passed = 0
    results: list[bool | str] = []
    for stdin, expected in zip(inputs, outputs):
        status, actual = run_case(compiled, stdin, timeout, output_limit)
        counts[status] += 1
        correct = status == "ok" and outputs_equivalent(actual, expected)
        if correct:
            passed += 1
        results.append(correct if status == "ok" else status)
    print(
        json.dumps(
            make_result(
                passed=passed,
                total=len(inputs),
                timeouts=counts["timeout"],
                runtime_errors=counts["runtime_error"],
                output_limit_errors=counts["output_limit"],
                diagnostics=[
                    name
                    for name in ("timeout", "runtime_error", "output_limit")
                    if counts[name]
                ],
                results=results,
            )
        )
    )


if __name__ == "__main__":
    main()
