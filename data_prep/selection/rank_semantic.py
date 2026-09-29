#!/usr/bin/env python3
"""Rank a frozen candidate pool by deterministic semantic farthest-first traversal."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "opd_semantic_diversity_ranking_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_id(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping) or "index" not in extra:
        raise ValueError("every source row must have extra_info.index")
    return str(extra["index"])


def prompt_text(row: Mapping[str, Any]) -> str:
    messages = row.get("prompt")
    if not isinstance(messages, list):
        raise ValueError(f"prompt is not a message list: {prompt_id(row)}")
    parts = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"invalid prompt message: {prompt_id(row)}")
        parts.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
    return "\n".join(parts)


def tie_break(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:semantic:{identifier}".encode()).hexdigest()


def last_token_pool(hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    if bool((attention_mask[:, -1] == 1).all()):
        return hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device), lengths
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen ranking: {output}")

    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    rows = pq.read_table(source).to_pylist()
    identifiers = [prompt_id(row) for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("source contains duplicate prompt IDs")
    texts = [prompt_text(row) for row in rows]

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            pooled = last_token_pool(hidden, encoded["attention_mask"])
            batches.append(functional.normalize(pooled.float(), p=2, dim=1).cpu())
    embeddings = torch.cat(batches)

    centroid = functional.normalize(embeddings.mean(dim=0), p=2, dim=0)
    centroid_similarity = embeddings @ centroid
    seed_index = min(
        range(len(rows)),
        key=lambda index: (
            -float(centroid_similarity[index]),
            tie_break(args.selection_seed, identifiers[index]),
        ),
    )
    selected = [seed_index]
    available = torch.ones(len(rows), dtype=torch.bool)
    available[seed_index] = False
    max_similarity = embeddings @ embeddings[seed_index]
    nearest_selected_distance = [float("nan")] * len(rows)
    nearest_selected_distance[seed_index] = 0.0
    while len(selected) < len(rows):
        candidates = available.nonzero(as_tuple=False).flatten().tolist()
        next_index = min(
            candidates,
            key=lambda index: (
                float(max_similarity[index]),
                tie_break(args.selection_seed, identifiers[index]),
            ),
        )
        nearest_selected_distance[next_index] = 1.0 - float(max_similarity[next_index])
        selected.append(next_index)
        available[next_index] = False
        max_similarity = torch.maximum(
            max_similarity, embeddings @ embeddings[next_index]
        )

    elapsed = time.monotonic() - started
    payload = {
        "schema_version": SCHEMA,
        "status": "frozen_before_training",
        "selection_method": "qwen3_embedding_centroid_seeded_farthest_first",
        "selection_seed": args.selection_seed,
        "ordered_prompt_ids": [identifiers[index] for index in selected],
        "ranking_rows": [
            {
                "rank": rank,
                "prompt_id": identifiers[index],
                "cosine_to_global_centroid": float(centroid_similarity[index]),
                "distance_to_nearest_previously_selected": nearest_selected_distance[
                    index
                ],
            }
            for rank, index in enumerate(selected, start=1)
        ],
        "embedding_definition": {
            "model": args.model,
            "prompt_serialization": "newline-joined '<role>: <content>' messages",
            "pooling": "last non-padding token",
            "normalization": "L2",
            "max_length": args.max_length,
            "truncation": True,
        },
        "traversal": {
            "initial_point": "candidate with maximum cosine to normalized global centroid",
            "subsequent_points": "maximum minimum cosine distance to selected set",
            "tie_break": "SHA256(selection_seed, 'semantic', prompt_id)",
        },
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "rows": len(rows),
        },
        "selection_cost": {
            "gpu_count": int(torch.cuda.is_available()),
            "wall_seconds": elapsed,
            "measured_gpu_hours": elapsed / 3600 if torch.cuda.is_available() else 0,
            "excludes_model_download": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "selection_method": payload["selection_method"],
                "candidates": len(rows),
                "first_8": payload["ordered_prompt_ids"][:8],
                "selection_cost": payload["selection_cost"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
