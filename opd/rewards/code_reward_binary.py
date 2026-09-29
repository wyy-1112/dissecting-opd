#!/usr/bin/env python3
"""All-or-nothing view of the local code judge, for parity with G-OPD's code teacher.

G-OPD scores code with verl's ``prime_code`` at ``continuous=False``: a rollout earns 1.0
only when every test passes. Our judge already reports ``all_pass`` alongside the partial
``passed/total``, so this module reuses it verbatim and only swaps which of the two becomes
the reward. The fractional value survives as ``tests_fraction`` so training curves can still
show partial progress.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_FRACTIONAL_PATH = Path(__file__).with_name("code_reward.py")
_spec = importlib.util.spec_from_file_location("opd_code_reward_fractional", _FRACTIONAL_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - packaging guard
    raise ImportError(f"cannot load {_FRACTIONAL_PATH}")
_fractional = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fractional)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    result = _fractional.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    result["tests_fraction"] = result["score"]
    result["score"] = result["all_pass"]
    result["acc"] = result["all_pass"]
    return result


def compute_score_compact(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Scalar-only metadata, which is all Ray can concatenate across samples."""
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
