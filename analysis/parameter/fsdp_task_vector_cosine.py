#!/usr/bin/env python3
"""Compare task vectors directly from native FP32 FSDP checkpoints.

The standard VERL FSDP merger casts model shards to bfloat16. That output is
appropriate for inference but not for measuring the optimizer's FP32 master
weight displacement. This script reconstructs one parameter at a time from the
native DTensor shards and never applies that merger cast.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor


EXPECTED_PARAMETER_COUNT = 1_720_574_976


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        action="append",
        required=True,
        metavar="LABEL=FSDP_ACTOR_DIR",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-workers", type=int, default=8)
    parser.add_argument(
        "--load-mode",
        choices=("mmap", "eager"),
        default="mmap",
        help=(
            "mmap minimizes resident memory but can cause slow random reads on "
            "network filesystems; eager reads checkpoint shards sequentially"
        ),
    )
    parser.add_argument(
        "--dot-product-mode",
        choices=("pairwise-fp64-reduction", "gram-fp32"),
        default="pairwise-fp64-reduction",
        help=(
            "gram-fp32 batches all full-parameter dot products into FP32 "
            "matrix multiplies with FP64 scalar accumulation"
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LEFT,RIGHT",
        help="Restrict dot products to selected label pairs.",
    )
    parser.add_argument(
        "--decompose",
        action="append",
        default=[],
        metavar="LONG,SHORT",
        help=(
            "Decompose LONG into its global projection on SHORT and the "
            "orthogonal residual. May be repeated."
        ),
    )
    parser.add_argument(
        "--decomposition-target",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Endpoint against which projected/residual components are compared. "
            "Defaults to every endpoint other than LONG and SHORT."
        ),
    )
    parser.add_argument("--dot-chunk-size", type=int, default=4_194_304)
    parser.add_argument(
        "--expected-parameter-count",
        type=int,
        default=EXPECTED_PARAMETER_COUNT,
        help="Set to 0 to accept the checkpoint-derived parameter count.",
    )
    return parser.parse_args()


def parse_endpoints(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"endpoint must be LABEL=PATH: {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or label in result:
            raise ValueError(f"invalid or duplicate endpoint label: {label!r}")
        result[label] = Path(raw_path)
    if len(result) < 2:
        raise ValueError("at least two endpoints are required")
    return result


class SafeTensorCheckpoint:
    def __init__(self, root: Path) -> None:
        self.root = root
        single = root / "model.safetensors"
        index_path = root / "model.safetensors.index.json"
        self._handles: dict[str, Any] = {}
        if single.is_file():
            handle = safe_open(single, framework="pt", device="cpu")
            self.weight_map = {name: single.name for name in handle.keys()}
            self._handles[single.name] = handle
        elif index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(index["weight_map"])
        else:
            raise FileNotFoundError(f"{root}: safetensors weights are missing")

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(
                self.root / filename,
                framework="pt",
                device="cpu",
            )
            self._handles[filename] = handle
        return handle.get_tensor(name)


class NativeFsdpCheckpoint:
    def __init__(
        self,
        root: Path,
        load_workers: int,
        *,
        mmap: bool,
    ) -> None:
        self.root = root
        config_path = root / "fsdp_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.world_size = int(config["world_size"])
        paths = [
            root / f"model_world_size_{self.world_size}_rank_{rank}.pt"
            for rank in range(self.world_size)
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing FSDP shards: {missing[:3]}")

        def load(path: Path) -> dict[str, Any]:
            # These are trusted, locally produced experiment checkpoints.
            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=mmap,
            )

        with ThreadPoolExecutor(
            max_workers=min(load_workers, self.world_size)
        ) as executor:
            self.states = list(executor.map(load, paths))
        reference_names = set(self.states[0])
        for rank, state in enumerate(self.states[1:], start=1):
            if set(state) != reference_names:
                raise RuntimeError(f"{root}: parameter keys differ at rank {rank}")
        self.names = reference_names
        self.local_dtype_counts = Counter(
            str(
                value._local_tensor.dtype
                if isinstance(value, DTensor)
                else value.dtype
            )
            for state in self.states
            for value in state.values()
        )

    def pop_tensor(self, name: str) -> torch.Tensor:
        local_tensors: list[torch.Tensor] = []
        placements = None
        global_shape = None
        for state in self.states:
            value = state.pop(name)
            if isinstance(value, DTensor):
                local = value._local_tensor
                current_placements = tuple(value.placements)
                # Older serialized DeviceMesh objects may predate the
                # ``_mesh_dim_names`` field; these checkpoints are still valid
                # one-dimensional FSDP meshes.
                mesh_names = getattr(
                    value.device_mesh,
                    "_mesh_dim_names",
                    None,
                )
                if mesh_names and mesh_names[0] in ("dp", "ddp"):
                    current_placements = current_placements[1:]
                if placements is None:
                    placements = current_placements
                    global_shape = tuple(value.shape)
                elif placements != current_placements:
                    raise RuntimeError(f"{self.root}: {name} placement mismatch")
            else:
                local = value
            local_tensors.append(local.detach())

        if placements is None:
            first_shape = tuple(local_tensors[0].shape)
            if all(tuple(tensor.shape) == first_shape for tensor in local_tensors):
                return local_tensors[0].contiguous()
            return torch.cat(local_tensors, dim=0).contiguous()
        if len(placements) != 1:
            raise NotImplementedError(
                f"{self.root}: unsupported placements for {name}: {placements}"
            )
        placement = placements[0]
        if placement.is_replicate():
            result = local_tensors[0]
        elif placement.is_shard():
            result = torch.cat(local_tensors, dim=placement.dim)
            assert global_shape is not None
            if tuple(result.shape) != global_shape:
                slices = tuple(
                    slice(0, expected)
                    for expected in global_shape
                )
                result = result[slices]
        else:
            raise NotImplementedError(
                f"{self.root}: unsupported placement for {name}: {placement}"
            )
        if tuple(result.shape) != global_shape:
            raise RuntimeError(
                f"{self.root}: reconstructed {name} shape {tuple(result.shape)} "
                f"!= {global_shape}"
            )
        return result.contiguous()


def group_name(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embedding"
    if name == "model.norm.weight":
        return "final_norm"
    if name.startswith("model.layers."):
        return ".".join(name.split(".")[:3])
    return "other"


def cosine(dot: float, left_sq: float, right_sq: float) -> float:
    denominator = math.sqrt(left_sq * right_sq)
    return dot / denominator if denominator else float("nan")


def stable_dot(
    left: torch.Tensor,
    right: torch.Tensor,
    chunk_size: int,
) -> float:
    total = 0.0
    for start in range(0, left.numel(), chunk_size):
        stop = min(start + chunk_size, left.numel())
        # Products remain FP32, but FP64 reduction removes size-dependent
        # summation drift from billion-parameter vectors.
        total += torch.sum(
            left[start:stop] * right[start:stop],
            dtype=torch.float64,
        ).item()
    return total


def add_pair(
    pairs: list[tuple[str, str]],
    left: str,
    right: str,
) -> None:
    if left == right:
        return
    if (left, right) not in pairs and (right, left) not in pairs:
        pairs.append((left, right))


def dot_for(
    dots: dict[tuple[str, str], float],
    left: str,
    right: str,
) -> float:
    if (left, right) in dots:
        return dots[(left, right)]
    if (right, left) in dots:
        return dots[(right, left)]
    raise KeyError(f"dot product was not accumulated for {left!r}, {right!r}")


def layer_dot_for(
    dots: dict[tuple[str, str, str], float],
    group: str,
    left: str,
    right: str,
) -> float:
    if (group, left, right) in dots:
        return dots[(group, left, right)]
    if (group, right, left) in dots:
        return dots[(group, right, left)]
    raise KeyError(
        f"layer dot product was not accumulated for {group}: "
        f"{left!r}, {right!r}"
    )


def nonnegative_squared(value: float, *scale_terms: float) -> float:
    scale = max((abs(term) for term in scale_terms), default=1.0)
    tolerance = 1e-7 * max(scale, 1.0)
    if value < -tolerance:
        raise RuntimeError(
            f"derived a negative squared norm {value} at scale {scale}"
        )
    return max(value, 0.0)


def component_geometry(
    *,
    short_squared_norm: float,
    long_squared_norm: float,
    long_short_dot: float,
    targets: dict[str, dict[str, float]],
    coefficient: float | None = None,
) -> dict[str, Any]:
    if short_squared_norm == 0 and coefficient is None:
        raise RuntimeError("cannot project onto a zero-norm short task vector")
    alpha = (
        long_short_dot / short_squared_norm
        if coefficient is None
        else coefficient
    )
    projection_squared_norm = alpha * alpha * short_squared_norm
    projection_residual_dot = alpha * (
        long_short_dot - alpha * short_squared_norm
    )
    residual_squared_norm = nonnegative_squared(
        long_squared_norm
        - 2.0 * alpha * long_short_dot
        + projection_squared_norm,
        long_squared_norm,
        projection_squared_norm,
        2.0 * alpha * long_short_dot,
    )
    result: dict[str, Any] = {
        "coefficient": alpha,
        "exact_norms": {
            "long": math.sqrt(long_squared_norm),
            "short": math.sqrt(short_squared_norm),
            "projection": math.sqrt(projection_squared_norm),
            "residual": math.sqrt(residual_squared_norm),
        },
        "exact_squared_norms": {
            "long": long_squared_norm,
            "short": short_squared_norm,
            "projection": projection_squared_norm,
            "residual": residual_squared_norm,
        },
        "projection_vs_residual": {
            "dot": projection_residual_dot,
            "cosine": cosine(
                projection_residual_dot,
                projection_squared_norm,
                residual_squared_norm,
            ),
        },
        "reconstruction_squared_norm_error": (
            projection_squared_norm
            + residual_squared_norm
            + 2.0 * projection_residual_dot
            - long_squared_norm
        ),
        "energy_fractions_of_long": {
            "projection": (
                projection_squared_norm / long_squared_norm
                if long_squared_norm
                else float("nan")
            ),
            "residual": (
                residual_squared_norm / long_squared_norm
                if long_squared_norm
                else float("nan")
            ),
        },
        "against_targets": {},
    }
    for target, values in targets.items():
        target_squared_norm = values["squared_norm"]
        projection_target_dot = alpha * values["short_dot"]
        residual_target_dot = (
            values["long_dot"] - projection_target_dot
        )
        result["against_targets"][target] = {
            "target_norm": math.sqrt(target_squared_norm),
            "long_dot": values["long_dot"],
            "short_dot": values["short_dot"],
            "projection_dot": projection_target_dot,
            "residual_dot": residual_target_dot,
            "long_cosine": cosine(
                values["long_dot"],
                long_squared_norm,
                target_squared_norm,
            ),
            "short_cosine": cosine(
                values["short_dot"],
                short_squared_norm,
                target_squared_norm,
            ),
            "projection_cosine": cosine(
                projection_target_dot,
                projection_squared_norm,
                target_squared_norm,
            ),
            "residual_cosine": cosine(
                residual_target_dot,
                residual_squared_norm,
                target_squared_norm,
            ),
        }
    return result


def build_decomposition_geometry(
    *,
    long_label: str,
    short_label: str,
    targets: tuple[str, ...],
    squared_norms: dict[str, float],
    dots: dict[tuple[str, str], float],
    layer_squared_norms: dict[str, dict[str, float]],
    layer_dots: dict[tuple[str, str, str], float],
) -> dict[str, Any]:
    global_targets = {
        target: {
            "squared_norm": squared_norms[target],
            "long_dot": dot_for(dots, long_label, target),
            "short_dot": dot_for(dots, short_label, target),
        }
        for target in targets
    }
    long_short_dot = dot_for(dots, long_label, short_label)
    global_geometry = component_geometry(
        short_squared_norm=squared_norms[short_label],
        long_squared_norm=squared_norms[long_label],
        long_short_dot=long_short_dot,
        targets=global_targets,
    )
    global_coefficient = global_geometry["coefficient"]

    layer_geometry = []
    for group in sorted(layer_squared_norms[long_label]):
        layer_targets = {
            target: {
                "squared_norm": layer_squared_norms[target][group],
                "long_dot": layer_dot_for(
                    layer_dots,
                    group,
                    long_label,
                    target,
                ),
                "short_dot": layer_dot_for(
                    layer_dots,
                    group,
                    short_label,
                    target,
                ),
            }
            for target in targets
        }
        layer_long_short_dot = layer_dot_for(
            layer_dots,
            group,
            long_label,
            short_label,
        )
        layer_short_squared_norm = layer_squared_norms[short_label][group]
        layer_long_squared_norm = layer_squared_norms[long_label][group]
        row: dict[str, Any] = {
            "group": group,
            "global_projection_components": component_geometry(
                short_squared_norm=layer_short_squared_norm,
                long_squared_norm=layer_long_squared_norm,
                long_short_dot=layer_long_short_dot,
                targets=layer_targets,
                coefficient=global_coefficient,
            ),
        }
        if layer_short_squared_norm:
            row["locally_orthogonal_components"] = component_geometry(
                short_squared_norm=layer_short_squared_norm,
                long_squared_norm=layer_long_squared_norm,
                long_short_dot=layer_long_short_dot,
                targets=layer_targets,
            )
        else:
            row["locally_orthogonal_components"] = None
        layer_geometry.append(row)
    return {
        "definition": (
            f"delta_{long_label} = alpha * delta_{short_label} + residual"
        ),
        "long": long_label,
        "short": short_label,
        "targets": list(targets),
        "global": global_geometry,
        "layer_geometry": layer_geometry,
        "layer_component_policy": (
            "global_projection_components uses the one all-parameter alpha; "
            "locally_orthogonal_components additionally reports each layer's "
            "own least-squares alpha"
        ),
    }


def main() -> None:
    args = parse_args()
    endpoint_paths = parse_endpoints(args.endpoint)
    labels = tuple(endpoint_paths)
    decomposition_specs: list[tuple[str, str, tuple[str, ...]]] = []
    for value in args.decompose:
        labels_in_spec = tuple(value.split(","))
        if (
            len(labels_in_spec) != 2
            or labels_in_spec[0] == labels_in_spec[1]
            or any(label not in endpoint_paths for label in labels_in_spec)
        ):
            raise ValueError(f"invalid --decompose {value!r}")
        long_label, short_label = labels_in_spec
        targets = tuple(
            args.decomposition_target
            or [
                label
                for label in labels
                if label not in (long_label, short_label)
            ]
        )
        if (
            not targets
            or len(set(targets)) != len(targets)
            or any(
                target not in endpoint_paths
                or target in (long_label, short_label)
                for target in targets
            )
        ):
            raise ValueError(
                f"invalid decomposition targets for {value!r}: {targets}"
            )
        decomposition_specs.append((long_label, short_label, targets))
    if args.pair:
        requested_pairs: list[tuple[str, str]] = []
        for value in args.pair:
            pair = tuple(value.split(","))
            if len(pair) != 2 or any(label not in endpoint_paths for label in pair):
                raise ValueError(f"invalid --pair {value!r}")
            add_pair(requested_pairs, pair[0], pair[1])
    else:
        requested_pairs = list(itertools.combinations(labels, 2))
    for long_label, short_label, targets in decomposition_specs:
        add_pair(requested_pairs, long_label, short_label)
        for target in targets:
            add_pair(requested_pairs, long_label, target)
            add_pair(requested_pairs, short_label, target)
    pairs = tuple(requested_pairs)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    print(f"[load base] {args.base}", flush=True)
    base = SafeTensorCheckpoint(args.base)
    names = tuple(base.weight_map)
    print(f"[base parameters] {len(names)}", flush=True)
    checkpoints: dict[str, NativeFsdpCheckpoint] = {}
    for label, path in endpoint_paths.items():
        print(f"[load {label}] {path}", flush=True)
        checkpoint = NativeFsdpCheckpoint(
            path,
            args.load_workers,
            mmap=args.load_mode == "mmap",
        )
        missing = set(names) - checkpoint.names
        extras = checkpoint.names - set(names)
        if missing:
            raise RuntimeError(f"{label}: missing parameters {sorted(missing)[:5]}")
        if extras != {"lm_head.weight"}:
            raise RuntimeError(
                f"{label}: expected only tied lm_head extra, got "
                f"{sorted(extras)[:5]}"
            )
        if set(checkpoint.local_dtype_counts) != {"torch.float32"}:
            raise RuntimeError(
                f"{label}: native shard dtypes are "
                f"{dict(checkpoint.local_dtype_counts)}"
            )
        checkpoints[label] = checkpoint

    squared_norms: dict[str, float] = defaultdict(float)
    dots: dict[tuple[str, str], float] = defaultdict(float)
    layer_squared_norms: dict[str, dict[str, float]] = {
        label: defaultdict(float) for label in labels
    }
    layer_dots: dict[tuple[str, str, str], float] = defaultdict(float)
    parameter_count = 0
    for index, name in enumerate(names, start=1):
        base_tensor = base.tensor(name)
        parameter_count += base_tensor.numel()
        base_device = base_tensor.to(device=device, dtype=torch.float32)
        deltas: dict[str, torch.Tensor] = {}
        for label in labels:
            endpoint = checkpoints[label].pop_tensor(name)
            if endpoint.dtype != torch.float32:
                raise RuntimeError(f"{label}: {name} is {endpoint.dtype}")
            if endpoint.shape != base_tensor.shape:
                raise RuntimeError(
                    f"{label}: {name} shape {tuple(endpoint.shape)} != "
                    f"{tuple(base_tensor.shape)}"
                )
            deltas[label] = (
                endpoint.to(device=device)
                .sub(base_device)
                .reshape(-1)
            )
        group = group_name(name)
        parameter_norms: dict[str, float]
        parameter_dots: dict[tuple[str, str], float]
        if args.dot_product_mode == "pairwise-fp64-reduction":
            parameter_norms = {
                label: stable_dot(
                    delta,
                    delta,
                    args.dot_chunk_size,
                )
                for label, delta in deltas.items()
            }
            parameter_dots = {
                (left, right): stable_dot(
                    deltas[left],
                    deltas[right],
                    args.dot_chunk_size,
                )
                for left, right in pairs
            }
        else:
            parameter_norms = {label: 0.0 for label in labels}
            parameter_dots = {pair: 0.0 for pair in pairs}
            flat_numel = next(iter(deltas.values())).numel()
            for start in range(0, flat_numel, args.dot_chunk_size):
                stop = min(start + args.dot_chunk_size, flat_numel)
                matrix = torch.stack(
                    [deltas[label][start:stop] for label in labels]
                )
                gram = (matrix @ matrix.T).double().cpu()
                for label_index, label in enumerate(labels):
                    parameter_norms[label] += gram[
                        label_index, label_index
                    ].item()
                for left, right in pairs:
                    parameter_dots[(left, right)] += gram[
                        labels.index(left), labels.index(right)
                    ].item()
                del matrix, gram
        for label, norm_sq in parameter_norms.items():
            squared_norms[label] += norm_sq
            layer_squared_norms[label][group] += norm_sq
        for (left, right), dot in parameter_dots.items():
            dots[(left, right)] += dot
            layer_dots[(group, left, right)] += dot
        del base_device, deltas
        if index % 25 == 0 or index == len(names):
            print(f"[compare] {index}/{len(names)} {name}", flush=True)

    if (
        args.expected_parameter_count
        and parameter_count != args.expected_parameter_count
    ):
        raise RuntimeError(
            f"parameter count {parameter_count} != "
            f"{args.expected_parameter_count}"
        )
    for label, checkpoint in checkpoints.items():
        leftovers = set(checkpoint.states[0])
        if leftovers != {"lm_head.weight"}:
            raise RuntimeError(f"{label}: unexpected leftovers {leftovers}")

    exact_cosines = {
        f"{left}_vs_{right}": cosine(
            dot_for(dots, left, right),
            squared_norms[left],
            squared_norms[right],
        )
        for left, right in pairs
    }
    layer_geometry = []
    groups = sorted(layer_squared_norms[labels[0]])
    for group in groups:
        row: dict[str, Any] = {"group": group}
        for label in labels:
            row[f"{label}_energy_fraction"] = (
                layer_squared_norms[label][group] / squared_norms[label]
            )
        for left, right in pairs:
            row[f"{left}_vs_{right}_cosine"] = cosine(
                layer_dot_for(layer_dots, group, left, right),
                layer_squared_norms[left][group],
                layer_squared_norms[right][group],
            )
        layer_geometry.append(row)

    decompositions = {
        f"{long_label}_on_{short_label}": build_decomposition_geometry(
            long_label=long_label,
            short_label=short_label,
            targets=targets,
            squared_norms=squared_norms,
            dots=dots,
            layer_squared_norms=layer_squared_norms,
            layer_dots=layer_dots,
        )
        for long_label, short_label, targets in decomposition_specs
    }
    report = {
        "schema_version": 2,
        "kind": "native_fp32_fsdp_task_vector_geometry_v2",
        "definition": "native_fp32_endpoint_minus_fp32_sft_base",
        "base": str(args.base.resolve()),
        "endpoints": {
            label: {
                "path": str(endpoint_paths[label].resolve()),
                "world_size": checkpoints[label].world_size,
                "local_dtype_counts": dict(
                    checkpoints[label].local_dtype_counts
                ),
            }
            for label in labels
        },
        "tied_parameter_policy": "exclude_native_lm_head_alias",
        "parameter_count": parameter_count,
        "computation": {
            "device": str(device),
            "load_mode": args.load_mode,
            "load_workers": args.load_workers,
            "dot_product_mode": args.dot_product_mode,
            "dot_chunk_size": args.dot_chunk_size,
            "all_parameters": True,
            "scalar_accumulation": "python_float64",
            "tf32_allowed": (
                torch.backends.cuda.matmul.allow_tf32
                if device.type == "cuda"
                else None
            ),
        },
        "exact_norms": {
            label: math.sqrt(squared_norms[label]) for label in labels
        },
        "exact_cosines": exact_cosines,
        "layer_geometry": layer_geometry,
        "decompositions": decompositions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(exact_cosines, indent=2), flush=True)
    print(f"[complete] {args.output}", flush=True)


if __name__ == "__main__":
    main()
