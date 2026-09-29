#!/usr/bin/env python3
"""Constant task reward for pure OPD.

verl still invokes a reward function in the synchronous OPD trainer even when
``use_task_rewards=False``. This implementation intentionally does not parse,
compile, or execute the sampled response.
"""
from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    del data_source, solution_str, ground_truth, extra_info
    return {
        "score": 0.0,
        "acc": 0.0,
        "all_pass": 0.0,
        "task_reward_enabled": 0.0,
        "reward_contract": "opd_no_task_reward_v1",
    }
