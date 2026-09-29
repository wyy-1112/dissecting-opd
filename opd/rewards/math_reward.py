#!/usr/bin/env python3
"""Boxed-answer math reward for the G-OPD replication GRPO run.

Scoring mirrors ``verl/verl/utils/reward_score/math_verify.py`` in the G-OPD
tree: take the last ``\\boxed{...}`` span in the response, truncate it to 300
characters, and compare against the ground truth with math-verify.  Reward is
1.0 for a match and 0.0 otherwise, matching the paper's binary math reward.

math-verify bounds sympy with ``signal.alarm``, which only works on the main
thread, whereas verl's reward loop calls ``compute_score`` from a shared
ThreadPoolExecutor.  Off the main thread the bound silently disappears and an
answer such as a power tower spins sympy forever, permanently consuming one of
the executor's threads; once enough threads are wedged every rollout sample
stalls, because the same executor also runs the tokenizer decode.  Verification
therefore happens in a pool of persistent child processes: the child works on
its main thread so math-verify's own timeout applies, and the parent enforces a
hard wall-clock bound by killing a child that overruns it.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import threading
from collections import deque
from fractions import Fraction
from typing import Any

# Imported eagerly so forked children inherit the module and never have to take
# the import lock themselves.
import math_verify  # noqa: F401

_ANSWER_CHARACTER_LIMIT = 300
_PARSING_TIMEOUT_SECONDS = 5
_VERIFY_TIMEOUT_SECONDS = 5
_HARD_TIMEOUT_SECONDS = float(os.getenv("MATH_REWARD_VERIFY_TIMEOUT_SECONDS", "15"))
_VERIFY_WORKERS = int(os.getenv("MATH_REWARD_VERIFY_WORKERS", "8"))


def last_boxed_only_string(string: str) -> str | None:
    index = string.rfind("\\boxed")
    if index < 0:
        index = string.rfind("\\fbox")
        if index < 0:
            return None

    position = index
    right_brace_index = None
    open_braces = 0
    while position < len(string):
        if string[position] == "{":
            open_braces += 1
        if string[position] == "}":
            open_braces -= 1
            if open_braces == 0:
                right_brace_index = position
                break
        position += 1

    if right_brace_index is None:
        return None
    return string[index : right_brace_index + 1]


def remove_boxed(span: str | None) -> str | None:
    if span is None:
        return None
    left = "\\boxed{"
    try:
        assert span[: len(left)] == left
        assert span[-1] == "}"
        return span[len(left) : -1]
    except Exception:
        return None


def _normalize(text: str) -> str:
    text = text.replace("\\left", "").replace("\\right", "")
    for spacing in ("\\!", "\\,", "\\;", "\\quad", "\\qquad", "~"):
        text = text.replace(spacing, "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    if text.startswith("+"):
        text = text[1:]
    return text


_INTEGER_OR_DECIMAL = re.compile(r"[-+]?\d+(?:\.\d+)?")
_PLAIN_FRACTION = re.compile(r"([-+]?\d+)/([-+]?\d+)")
_LATEX_FRACTION = re.compile(r"[-+]?\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}")


def _as_fraction(text: str) -> Fraction | None:
    """Exact rational value of a plain numeric answer, else None.

    Only shapes whose value is unambiguous are handled, so agreeing with
    math-verify on them is guaranteed; anything else is left to sympy.
    """
    candidate = text.replace(",", "")
    if not candidate or candidate.endswith("%") or candidate.endswith("\\%"):
        return None

    match = _LATEX_FRACTION.fullmatch(candidate)
    if match:
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if denominator == 0:
            return None
        value = Fraction(numerator, denominator)
        return -value if candidate.startswith("-") else value

    match = _PLAIN_FRACTION.fullmatch(candidate)
    if match:
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if denominator == 0:
            return None
        return Fraction(numerator, denominator)

    if _INTEGER_OR_DECIMAL.fullmatch(candidate):
        return Fraction(candidate)
    return None


def _verify_child(connection) -> None:
    from math_verify import parse, verify

    while True:
        try:
            job = connection.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        ground_truth, answer = job
        try:
            gold = parse("\\boxed{" + ground_truth + "}", parsing_timeout=_PARSING_TIMEOUT_SECONDS)
            prediction = parse("\\boxed{" + answer + "}", parsing_timeout=_PARSING_TIMEOUT_SECONDS)
            payload = ("ok", bool(verify(gold, prediction, timeout_seconds=_VERIFY_TIMEOUT_SECONDS)))
        except Exception:
            payload = ("error", False)
        try:
            connection.send(payload)
        except (BrokenPipeError, OSError):
            return


class _VerifierPool:
    """Fixed set of child processes, each killable on overrun.

    A caller owns a slot for the duration of its request, so only the idle queue
    needs locking.  Waiting for a slot is bounded by the hard timeout, which is
    what keeps a pathological answer from stalling the rollout.
    """

    def __init__(self, size: int, timeout: float) -> None:
        self._context = mp.get_context("fork")
        self._timeout = timeout
        self._slots: list[tuple[Any, Any] | None] = [None] * size
        self._available = threading.Condition()
        self._idle: deque[int] = deque(range(size))

    def _acquire(self) -> int:
        with self._available:
            while not self._idle:
                self._available.wait()
            return self._idle.popleft()

    def _release(self, index: int) -> None:
        with self._available:
            self._idle.append(index)
            self._available.notify()

    def _kill(self, index: int) -> None:
        slot = self._slots[index]
        self._slots[index] = None
        if slot is None:
            return
        process, parent_end = slot
        try:
            parent_end.close()
        except OSError:
            pass
        process.kill()
        process.join(timeout=5)

    def _ensure(self, index: int) -> tuple[Any, Any]:
        slot = self._slots[index]
        if slot is not None and slot[0].is_alive():
            return slot
        self._kill(index)
        parent_end, child_end = self._context.Pipe()
        process = self._context.Process(target=_verify_child, args=(child_end,), daemon=True)
        process.start()
        child_end.close()
        self._slots[index] = (process, parent_end)
        return self._slots[index]

    def verify(self, ground_truth: str, answer: str) -> tuple[bool, str]:
        """Return (matched, status) with status in {"ok", "error", "timeout"}."""
        index = self._acquire()
        try:
            _, parent_end = self._ensure(index)
            try:
                parent_end.send((ground_truth, answer))
                if not parent_end.poll(self._timeout):
                    self._kill(index)
                    return False, "timeout"
                status, matched = parent_end.recv()
            except (EOFError, OSError, BrokenPipeError):
                self._kill(index)
                return False, "error"
            return bool(matched), "ok" if status == "ok" else "error"
        finally:
            self._release(index)


_pool: _VerifierPool | None = None
_pool_lock = threading.Lock()


def _verifier() -> _VerifierPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _VerifierPool(_VERIFY_WORKERS, _HARD_TIMEOUT_SECONDS)
        return _pool


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: str | dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    del data_source, extra_info, kwargs

    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth", "")
    ground_truth = "" if ground_truth is None else str(ground_truth)

    answer = remove_boxed(last_boxed_only_string(solution_str or ""))
    if answer is None:
        return {
            "score": 0.0,
            "boxed_ok": 0.0,
            "fast_path": 0.0,
            "verify_error": 0.0,
            "verify_timeout": 0.0,
        }

    if len(answer) > _ANSWER_CHARACTER_LIMIT:
        answer = answer[:_ANSWER_CHARACTER_LIMIT]

    normalized_answer = _normalize(answer)
    normalized_truth = _normalize(ground_truth)
    if normalized_answer and normalized_answer == normalized_truth:
        return {
            "score": 1.0,
            "boxed_ok": 1.0,
            "fast_path": 1.0,
            "verify_error": 0.0,
            "verify_timeout": 0.0,
        }

    answer_value = _as_fraction(normalized_answer)
    truth_value = _as_fraction(normalized_truth)
    if answer_value is not None and truth_value is not None:
        return {
            "score": 1.0 if answer_value == truth_value else 0.0,
            "boxed_ok": 1.0,
            "fast_path": 1.0,
            "verify_error": 0.0,
            "verify_timeout": 0.0,
        }

    matched, status = _verifier().verify(ground_truth, answer)
    return {
        "score": 1.0 if matched else 0.0,
        "boxed_ok": 1.0,
        "fast_path": 0.0,
        "verify_error": 1.0 if status == "error" else 0.0,
        "verify_timeout": 1.0 if status == "timeout" else 0.0,
    }
