#!/usr/bin/env python3
"""Create a content fingerprint for a local Hugging Face checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MODEL_PATTERNS = ("*.safetensors", "*.bin")
METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path) -> dict:
    root = path.resolve()
    model_files = sorted(
        {
            candidate
            for pattern in MODEL_PATTERNS
            for candidate in root.glob(pattern)
            if candidate.is_file()
        }
    )
    if not model_files:
        raise ValueError(f"no model weights found under {root}")
    files = [
        *model_files,
        *(root / name for name in METADATA_FILES if (root / name).is_file()),
    ]
    records = []
    aggregate = hashlib.sha256()
    for file_path in sorted(set(files), key=lambda item: item.name):
        digest = sha256(file_path)
        record = {
            "name": file_path.name,
            "bytes": file_path.stat().st_size,
            "sha256": digest,
        }
        records.append(record)
        aggregate.update(
            f"{record['name']}\0{record['bytes']}\0{digest}\n".encode("utf-8")
        )
    return {
        "schema_version": "hf_checkpoint_fingerprint_v1",
        "path": str(root),
        "fingerprint": aggregate.hexdigest(),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = checkpoint_fingerprint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(document["fingerprint"])


if __name__ == "__main__":
    main()
