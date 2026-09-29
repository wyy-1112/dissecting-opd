"""The one helper the data-prep scripts import from the original project package."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def atomic_write_parquet(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    dataset_factory: Any = None,
) -> None:
    """Write parquet through a sibling temporary file and atomically replace."""
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if dataset_factory is None:
        import datasets

        dataset_factory = datasets.Dataset.from_list
    try:
        dataset_factory(rows).to_parquet(str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
