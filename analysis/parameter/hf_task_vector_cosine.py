#!/usr/bin/env python3
"""Exact all-parameter task-vector cosines for Hugging Face checkpoints.

Unlike native optimizer-master geometry, this reads the exact published HF
weights that are passed to the frozen-prefix functional forward.  It is the
strict implementation of "same checkpoint on both axes" for a parameter versus
function scatter.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, root: Path, stack: ExitStack) -> None:
        self.root = root.resolve()
        single = self.root / "model.safetensors"
        index_path = self.root / "model.safetensors.index.json"
        if single.is_file():
            handle = stack.enter_context(
                safe_open(single, framework="pt", device="cpu")
            )
            self.weight_map = {key: single.name for key in handle.keys()}
            self.handles = {single.name: handle}
        elif index_path.is_file():
            self.weight_map = json.loads(index_path.read_text())["weight_map"]
            self.handles: dict[str, Any] = {}
        else:
            raise FileNotFoundError(f"{root}: no safetensors weights")
        self.stack = stack

    def slice(self, name: str):
        filename = self.weight_map[name]
        if filename not in self.handles:
            self.handles[filename] = self.stack.enter_context(
                safe_open(
                    self.root / filename,
                    framework="pt",
                    device="cpu",
                )
            )
        return self.handles[filename].get_slice(name)


def parse_endpoint(value: str) -> tuple[str, Path]:
    label, separator, raw = value.partition("=")
    if not separator or not label:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    return label, Path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="LEFT,RIGHT; defaults to every pair",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-chunk-elements", type=int, default=4_194_304)
    args = parser.parse_args()

    entries = [parse_endpoint(value) for value in args.endpoint]
    endpoints = dict(entries)
    if len(endpoints) != len(entries):
        raise ValueError("duplicate endpoint label")
    labels = list(endpoints)
    pairs = (
        [tuple(value.split(",", 1)) for value in args.pair]
        if args.pair
        else list(itertools.combinations(labels, 2))
    )
    if any(len(pair) != 2 or any(label not in endpoints for label in pair) for pair in pairs):
        raise ValueError("invalid pair")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False

    norms = {label: 0.0 for label in labels}
    dots = {pair: 0.0 for pair in pairs}
    parameter_count = 0
    with ExitStack() as stack:
        base = Checkpoint(args.base, stack)
        readers = {
            label: Checkpoint(path, stack) for label, path in endpoints.items()
        }
        expected = set(base.weight_map)
        for label, reader in readers.items():
            if set(reader.weight_map) != expected:
                raise ValueError(
                    f"{label}: tensor keys differ: "
                    f"missing={sorted(expected-set(reader.weight_map))[:5]} "
                    f"extra={sorted(set(reader.weight_map)-expected)[:5]}"
                )

        for key_index, key in enumerate(base.weight_map, start=1):
            base_slice = base.slice(key)
            shape = tuple(base_slice.get_shape()) or (1,)
            parameter_count += math.prod(shape)
            row_elements = math.prod(shape[1:]) if len(shape) > 1 else 1
            chunk_rows = max(1, args.max_chunk_elements // row_elements)
            endpoint_slices = {
                label: reader.slice(key) for label, reader in readers.items()
            }
            for start in range(0, shape[0], chunk_rows):
                stop = min(shape[0], start + chunk_rows)
                origin = base_slice[start:stop].to(
                    device=device, dtype=torch.float32
                )
                deltas = torch.stack(
                    [
                        (
                            endpoint_slices[label][start:stop]
                            .to(device=device, dtype=torch.float32)
                            - origin
                        ).reshape(-1)
                        for label in labels
                    ]
                )
                gram = (deltas @ deltas.T).double().cpu().numpy()
                for index, label in enumerate(labels):
                    norms[label] += float(gram[index, index])
                for pair in pairs:
                    dots[pair] += float(
                        gram[labels.index(pair[0]), labels.index(pair[1])]
                    )
                del origin, deltas, gram
            if key_index % 25 == 0 or key_index == len(base.weight_map):
                print(f"[compare] {key_index}/{len(base.weight_map)} {key}", flush=True)

    cosine = {
        f"{left}_vs_{right}": dots[(left, right)]
        / math.sqrt(norms[left] * norms[right])
        for left, right in pairs
    }
    payload = {
        "schema_version": "opd_hf_task_vector_cosine_v1",
        "definition": "endpoint HF weights minus the same Initial HF weights",
        "base": str(args.base.resolve()),
        "endpoints": {
            label: str(path.resolve()) for label, path in endpoints.items()
        },
        "parameter_count": parameter_count,
        "computation": {
            "all_parameters": True,
            "device": str(device),
            "product_dtype": "float32",
            "scalar_accumulation": "float64",
            "tf32_allowed": False,
            "same_weights_as_functional_forward": True,
        },
        "norms": {label: math.sqrt(value) for label, value in norms.items()},
        "cosines": cosine,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(cosine, indent=2, sort_keys=True))
    print(f"[write] {args.output}")


if __name__ == "__main__":
    main()
