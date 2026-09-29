"""Expose the paper's IF/Agent extensions to the production verl checkout."""
from __future__ import annotations

import importlib
import os
from pathlib import Path


def extend_package(name: str, path: Path) -> None:
    package = importlib.import_module(name)
    value = str(path)
    if value not in package.__path__:
        package.__path__.append(value)


root_value = os.getenv("LLM_FUSION_EXTENSION_ROOT")
if root_value:
    root = Path(root_value).resolve() / "verl"
    extend_package("verl", root)
    extend_package("verl.tools", root / "tools")
    extend_package("verl.utils", root / "utils")
    extend_package("verl.utils.reward_score", root / "utils/reward_score")
