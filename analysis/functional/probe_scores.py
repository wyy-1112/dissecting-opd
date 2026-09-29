#!/usr/bin/env python3
"""Build, score, and compare exact-token function-space policy deltas.

The pipeline deliberately separates the immutable probe cohort, per-checkpoint
chosen-token log-probabilities, and the final geometry report.  A checkpoint is
therefore loaded only once, every model sees byte-identical token IDs, and all
derived policy deltas can be reproduced from small CPU artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
COHORT_SCHEMA = "opd_function_space_probe_cohort_v1"
COHORT_MANIFEST_SCHEMA = "opd_function_space_probe_manifest_v1"
SCORE_SCHEMA = "opd_function_space_chosen_logprobs_v1"
REPORT_SCHEMA = "opd_function_space_policy_delta_geometry_v1"
FISHER_REFERENCE_SCHEMA = "opd_function_space_fisher_reference_v1"
FISHER_SCORE_SCHEMA = "opd_function_space_frozen_topk_logprobs_v1"
FISHER_REPORT_SCHEMA = "opd_function_space_fisher_geometry_v1"
ATTESTATION_SCHEMA = "opd_gradient_spectrum_base_attestation_v1"
REQUIRED_DOMAINS = ("math", "science", "code", "ood")
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON") from error


def checkpoint_fingerprint_document(
    checkpoint: Path,
    fingerprint_path: Path | None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    analysis_dir = str(Path(__file__).resolve().parent)
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from fingerprint_hf_checkpoint import checkpoint_fingerprint

    document = checkpoint_fingerprint(checkpoint)
    expected = (
        json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if fingerprint_path is not None
        else document
    )
    if expected.get("schema_version") != "hf_checkpoint_fingerprint_v1":
        raise ValueError("checkpoint fingerprint is not hf_checkpoint_fingerprint_v1")
    if Path(expected["path"]).resolve() != checkpoint:
        raise ValueError(
            f"fingerprint path {expected['path']} does not match {checkpoint}"
        )
    if expected.get("fingerprint") != document["fingerprint"]:
        raise ValueError(
            f"{fingerprint_path}: saved fingerprint does not match current "
            f"checkpoint content"
        )
    if document.get("schema_version") != "hf_checkpoint_fingerprint_v1":
        raise ValueError("checkpoint fingerprint is not hf_checkpoint_fingerprint_v1")
    if Path(document["path"]).resolve() != checkpoint:
        raise ValueError(
            f"fingerprint path {document['path']} does not match {checkpoint}"
        )
    fingerprint = document.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("checkpoint fingerprint is missing or malformed")
    return document


def checkpoint_identity(document: dict[str, Any]) -> dict[str, str]:
    return {
        "path": str(Path(document["path"]).resolve()),
        "fingerprint": document["fingerprint"],
    }


def tokenizer_identity(checkpoint: Path) -> dict[str, Any]:
    tokenizer_json = checkpoint.resolve() / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(
            f"tokenizer.json is required for exact-token identity: {checkpoint}"
        )
    return {
        "definition": "sha256(tokenizer.json)",
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
    }


def source_fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        candidate
        for candidate in path.glob("*.jsonl")
        if candidate.is_file()
    )
    if not files:
        raise ValueError(f"{path}: no JSONL source files")
    records = [
        {
            "name": candidate.name,
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
        for candidate in files
    ]
    return {
        "path": str(path),
        "kind": "jsonl_directory",
        "files": records,
        "fingerprint": canonical_sha256(records),
    }


def validate_token_ids(
    values: Any,
    *,
    location: str,
    field: str,
) -> list[int]:
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in values
        )
    ):
        raise ValueError(f"{location}: {field} must be non-empty token IDs")
    return values


def gradient_candidates(
    path: Path,
    *,
    domain: str,
) -> list[dict[str, Any]]:
    candidates = []
    for line_number, row in read_jsonl(path):
        location = f"{path}:{line_number}"
        prompt_ids = validate_token_ids(
            row.get("prompt_token_ids"),
            location=location,
            field="prompt_token_ids",
        )
        samples = row.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{location}: samples must be non-empty")
        prompt_key = str(row.get("problem_id", line_number))
        for sample_index, sample in enumerate(samples):
            response_ids = validate_token_ids(
                sample.get("response_token_ids"),
                location=f"{location}:sample{sample_index}",
                field="response_token_ids",
            )
            candidates.append(
                {
                    "domain": domain,
                    "source_id": f"{prompt_key}:{sample_index}",
                    "prompt_key": prompt_key,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": response_ids,
                    "trajectory_origin": "stored_exact_prompt_and_response_token_ids",
                    "source_metadata": {
                        "problem_id": prompt_key,
                        "sample_index": sample_index,
                    },
                }
            )
    return candidates


def rollout_candidates(
    path: Path,
    *,
    domain: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    candidates = []
    for line_number, row in read_jsonl(path):
        location = f"{path}:{line_number}"
        prompt_text = row.get("input")
        if not isinstance(prompt_text, str) or not prompt_text:
            raise ValueError(f"{location}: input prompt text is missing")
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        expected_prompt_count = row.get("prompt_token_count")
        if (
            expected_prompt_count is not None
            and len(prompt_ids) != expected_prompt_count
        ):
            prefix = "user\n"
            suffix = "\nassistant\n<think>\n\n</think>\n\n"
            if not prompt_text.startswith(prefix) or not prompt_text.endswith(
                suffix
            ):
                raise ValueError(
                    f"{location}: prompt text cannot be mapped back to the "
                    "canonical chat template"
                )
            user_content = prompt_text[
                len(prefix) : len(prompt_text) - len(suffix)
            ]
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer(
                rendered_prompt,
                add_special_tokens=False,
            )["input_ids"]
            if len(prompt_ids) != expected_prompt_count:
                raise ValueError(
                    f"{location}: reconstructed prompt token count "
                    f"{len(prompt_ids)} != recorded {expected_prompt_count}"
                )
        response_ids = validate_token_ids(
            row.get("response_token_ids"),
            location=location,
            field="response_token_ids",
        )
        expected_response_count = row.get("response_token_count")
        if (
            expected_response_count is not None
            and len(response_ids) != expected_response_count
        ):
            raise ValueError(
                f"{location}: response token count {len(response_ids)} "
                f"!= recorded {expected_response_count}"
            )
        extra = row.get("extra_info") or {}
        signal = extra.get("domain_signal") or {}
        prompt_key = str(
            signal.get("sample_id")
            or signal.get("prompt_sha256")
            or hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        )
        candidates.append(
            {
                "domain": domain,
                "source_id": (
                    f"{prompt_key}:step{row.get('step', 'unknown')}:"
                    f"line{line_number}"
                ),
                "prompt_key": prompt_key,
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "trajectory_origin": (
                    "stored_exact_response_ids_plus_canonical_chat_template_"
                    "prompt_ids_validated_by_recorded_token_count"
                ),
                "source_metadata": {
                    "step": row.get("step"),
                    "data_source": row.get("data_source"),
                    "response_truncated": row.get("response_truncated"),
                },
            }
        )
    return candidates


def profile_rollout_candidates(
    path: Path,
    *,
    domain: str,
) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not files:
        raise ValueError(f"{path}: no profile rollout JSONL files")
    candidates = []
    for file_path in files:
        for line_number, row in read_jsonl(file_path):
            row_domain = str(
                row.get("ability") or row.get("family_id") or ""
            ).lower()
            if row_domain != domain:
                continue
            location = f"{file_path}:{line_number}"
            prompt_ids = validate_token_ids(
                row.get("prompt_token_ids"),
                location=location,
                field="prompt_token_ids",
            )
            samples = row.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError(f"{location}: samples must be non-empty")
            prompt_key = str(row.get("problem_id") or row.get("question_id"))
            for fallback_index, sample in enumerate(samples):
                sample_index = int(
                    sample.get("sample_index", fallback_index)
                )
                response_ids = validate_token_ids(
                    sample.get("token_ids"),
                    location=f"{location}:sample{sample_index}",
                    field="token_ids",
                )
                expected_count = sample.get("num_tokens")
                if (
                    expected_count is not None
                    and len(response_ids) != expected_count
                ):
                    raise ValueError(
                        f"{location}:sample{sample_index}: token count "
                        f"{len(response_ids)} != recorded {expected_count}"
                    )
                candidates.append(
                    {
                        "domain": domain,
                        "source_id": f"{prompt_key}:{sample_index}",
                        "prompt_key": prompt_key,
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids,
                        "trajectory_origin": (
                            "stored_exact_profile_prompt_and_response_token_ids"
                        ),
                        "source_metadata": {
                            "problem_id": prompt_key,
                            "sample_index": sample_index,
                            "data_source": row.get("data_source"),
                            "finish_reason": sample.get("finish_reason"),
                        },
                    }
                )
    return candidates


def ood_eval_candidates(
    path: Path,
    *,
    domain: str,
    tokenizer: Any,
    benchmark: str | None = None,
) -> list[dict[str, Any]]:
    """Freeze probe trajectories from saved OOD evaluation shards.

    Every benchmark writes the same shard schema and every Task already carries the user content
    its own recipe prescribes, so the only benchmark-specific input is which task table to load.
    When the caller does not name it, take it from the shard directory, which the evaluation
    layout names after the benchmark.  A wrong guess cannot slip through: the prompt fingerprint
    check below rebuilds the prompt from the task table and would reject every row.
    """

    eval_dir = str(Path(__file__).resolve().parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from ood_common import BENCHMARKS, build_prompt, load_tasks, sha256_text

    if benchmark is None:
        benchmark = (path.parent if path.is_file() else path).name
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f"{path}: cannot resolve benchmark {benchmark!r}; expected one of {BENCHMARKS}"
        )
    tasks = {task.task_id: task for task in load_tasks(benchmark)}
    files = [path] if path.is_file() else sorted(path.glob("shard_*.jsonl"))
    if not files:
        raise ValueError(f"{path}: no {benchmark} evaluation shards")
    candidates = []
    seen_tasks: set[str] = set()
    for file_path in files:
        for line_number, row in read_jsonl(file_path):
            location = f"{file_path}:{line_number}"
            task_id = str(row.get("task_id"))
            if task_id in seen_tasks:
                raise ValueError(f"{location}: duplicate task {task_id}")
            seen_tasks.add(task_id)
            if task_id not in tasks:
                raise ValueError(f"{location}: unknown {benchmark} task {task_id}")
            prompt_text = build_prompt(tokenizer, tasks[task_id].user_content)
            if row.get("prompt_sha256") != sha256_text(prompt_text):
                raise ValueError(f"{location}: rendered prompt fingerprint mismatch")
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
            outputs = row.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                raise ValueError(f"{location}: outputs must be non-empty")
            for sample_index, output_text in enumerate(outputs):
                if not isinstance(output_text, str) or not output_text:
                    continue
                response_ids = tokenizer(
                    output_text,
                    add_special_tokens=False,
                )["input_ids"]
                if not response_ids:
                    continue
                candidates.append(
                    {
                        "domain": domain,
                        "source_id": f"{task_id}:{sample_index}",
                        "prompt_key": task_id,
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids,
                        "trajectory_origin": (
                            "canonical_exact_ids_frozen_from_saved_eval_text"
                        ),
                        "source_metadata": {
                            "benchmark": benchmark,
                            "task_id": task_id,
                            "sample_index": sample_index,
                            "original_output_token_count": (
                                row.get("output_tokens") or [None] * len(outputs)
                            )[sample_index],
                        },
                    }
                )
    return candidates


# Verbatim from G-OPD math_eval/eval_math.py, which appends it to every problem
# before applying the chat template.  Reconstructing the prompt requires
# reproducing it exactly, because the stored generations continue from it.
GOPD_MATH_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def math_eval_candidates(
    path: Path,
    *,
    domain: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Freeze probe trajectories from saved competition-math evaluation outputs.

    Unlike the OOD shards these dumps store no prompt fingerprint, so there is
    nothing to check a rebuilt prompt against.  What makes the probe sound is not
    fidelity to the bytes vLLM originally saw but that every checkpoint is later
    scored on one frozen sequence, so the prompt is rebuilt once here, by the same
    template and instruction the harness used, and then never rebuilt again.  The
    same reasoning already licenses ``ood_eval_candidates`` to retokenize saved
    output text.

    The benchmark name comes from the file stem, which the harness names after
    the benchmark, and it is carried into the prompt key so that problem 1 of
    AIME24 and problem 1 of HMMT25 cannot collide.
    """
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    if not files:
        raise ValueError(f"{path}: no math evaluation output files")
    candidates = []
    for file_path in files:
        benchmark = file_path.stem
        for line_number, row in read_jsonl(file_path):
            location = f"{file_path}:{line_number}"
            problem = row.get("problem")
            if not isinstance(problem, str) or not problem:
                raise ValueError(f"{location}: problem text is missing")
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": problem + GOPD_MATH_INSTRUCTION}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            responses = row.get("responses")
            if not isinstance(responses, list) or not responses:
                raise ValueError(f"{location}: responses must be non-empty")
            prompt_key = f"{benchmark}:{row.get('id', line_number)}"
            for sample_index, response_text in enumerate(responses):
                if not isinstance(response_text, str) or not response_text:
                    continue
                response_ids = tokenizer(
                    response_text,
                    add_special_tokens=False,
                )["input_ids"]
                if not response_ids:
                    continue
                candidates.append(
                    {
                        "domain": domain,
                        "source_id": f"{prompt_key}:{sample_index}",
                        "prompt_key": prompt_key,
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids,
                        "trajectory_origin": (
                            "canonical_exact_ids_frozen_from_saved_math_eval_text"
                        ),
                        "source_metadata": {
                            "benchmark": benchmark,
                            "problem_id": row.get("id"),
                            "sample_index": sample_index,
                            "correct": (row.get("acc_list") or [None] * len(responses))[
                                sample_index
                            ],
                        },
                    }
                )
    return candidates


def parse_source(value: str) -> tuple[str, str, Path]:
    left, separator, raw_path = value.partition("=")
    domain, colon, adapter = left.partition(":")
    domain = domain.lower()
    if (
        not separator
        or not colon
        or not raw_path
        # gpqa_eval predates the generalization and stays accepted so the frozen cohort
        # manifest still describes a runnable command.
        or adapter
        not in {
            "gradient",
            "rollout",
            "profile_rollout",
            "gpqa_eval",
            "ood_eval",
            "math_eval",
        }
    ):
        raise ValueError(
            "--source expects DOMAIN:{gradient,rollout,profile_rollout,"
            "gpqa_eval,ood_eval,math_eval}=PATH"
        )
    if not LABEL.fullmatch(domain):
        raise ValueError(f"invalid source domain {domain!r}")
    return domain, adapter, Path(raw_path)


def stable_candidate_key(
    candidate: dict[str, Any],
    *,
    seed: int,
) -> tuple[str, str]:
    identity = (
        f"{seed}\0{candidate['domain']}\0{candidate['source_id']}"
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest(), candidate["source_id"]


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: stable_candidate_key(candidate, seed=seed),
    )
    selected = []
    selected_ids: set[str] = set()
    prompt_keys: set[str] = set()
    for unique_prompts_only in (True, False):
        for candidate in ordered:
            source_id = candidate["source_id"]
            if source_id in selected_ids:
                continue
            if (
                unique_prompts_only
                and candidate["prompt_key"] in prompt_keys
            ):
                continue
            selected.append(candidate)
            selected_ids.add(source_id)
            prompt_keys.add(candidate["prompt_key"])
            if len(selected) == count:
                return selected
    raise ValueError(
        f"only {len(selected)} usable candidates, but {count} were requested"
    )


def build_cohort(args: argparse.Namespace) -> None:
    if args.per_domain <= 0:
        raise ValueError("--per-domain must be positive")
    if args.max_response_tokens <= 0 or args.max_sequence_tokens <= 1:
        raise ValueError("token limits must be positive")
    base_document = checkpoint_fingerprint_document(
        args.tokenizer,
        args.base_fingerprint,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    required_domains = tuple(
        domain.lower()
        for domain in (args.required_domain or REQUIRED_DOMAINS)
    )
    if len(set(required_domains)) != len(required_domains):
        raise ValueError("required domains must be unique")

    by_domain: dict[str, list[dict[str, Any]]] = {}
    source_records = []
    fingerprint_cache: dict[Path, dict[str, Any]] = {}
    for source_value in args.source:
        domain, adapter, path = parse_source(source_value)
        resolved_path = path.resolve()
        if resolved_path not in fingerprint_cache:
            fingerprint_cache[resolved_path] = source_fingerprint(path)
        fingerprint = fingerprint_cache[resolved_path]
        if adapter == "gradient":
            candidates = gradient_candidates(path, domain=domain)
        elif adapter == "rollout":
            candidates = rollout_candidates(
                path,
                domain=domain,
                tokenizer=tokenizer,
            )
        elif adapter == "profile_rollout":
            candidates = profile_rollout_candidates(
                path,
                domain=domain,
            )
        elif adapter == "math_eval":
            candidates = math_eval_candidates(
                path,
                domain=domain,
                tokenizer=tokenizer,
            )
        else:
            candidates = ood_eval_candidates(
                path,
                domain=domain,
                tokenizer=tokenizer,
                benchmark="gpqa_diamond" if adapter == "gpqa_eval" else None,
            )
        by_domain.setdefault(domain, []).extend(candidates)
        source_records.append(
            {
                "domain": domain,
                "adapter": adapter,
                "candidates": len(candidates),
                "source": fingerprint,
            }
        )
    if set(by_domain) != set(required_domains):
        raise ValueError(
            f"source domains {sorted(by_domain)} != required "
            f"{sorted(required_domains)}"
        )

    rows = []
    selected_prompt_counts = {}
    for domain in required_domains:
        usable = []
        for candidate in by_domain[domain]:
            prompt_ids = candidate["prompt_token_ids"]
            available = args.max_sequence_tokens - len(prompt_ids)
            response_limit = min(args.max_response_tokens, available)
            if response_limit <= 0:
                continue
            response_ids = candidate["response_token_ids"][:response_limit]
            if not response_ids:
                continue
            usable.append(
                {
                    **candidate,
                    "response_token_ids": response_ids,
                    "original_response_token_count": len(
                        candidate["response_token_ids"]
                    ),
                    "response_truncated_for_probe": (
                        len(response_ids)
                        < len(candidate["response_token_ids"])
                    ),
                }
            )
        selected = select_candidates(
            usable,
            count=args.per_domain,
            seed=args.seed,
        )
        selected_prompt_counts[domain] = len(
            {candidate["prompt_key"] for candidate in selected}
        )
        for candidate in selected:
            content_fingerprint = canonical_sha256(
                {
                    "domain": domain,
                    "prompt_token_ids": candidate["prompt_token_ids"],
                    "response_token_ids": candidate["response_token_ids"],
                }
            )
            rows.append(
                {
                    "schema_version": COHORT_SCHEMA,
                    "probe_id": (
                        f"{domain}:{len(rows):04d}:"
                        f"{content_fingerprint[:16]}"
                    ),
                    "domain": domain,
                    "source_id": candidate["source_id"],
                    "prompt_token_ids": candidate["prompt_token_ids"],
                    "response_token_ids": candidate["response_token_ids"],
                    "prompt_token_count": len(
                        candidate["prompt_token_ids"]
                    ),
                    "response_token_count": len(
                        candidate["response_token_ids"]
                    ),
                    "original_response_token_count": candidate[
                        "original_response_token_count"
                    ],
                    "response_truncated_for_probe": candidate[
                        "response_truncated_for_probe"
                    ],
                    "trajectory_origin": candidate["trajectory_origin"],
                    "source_metadata": candidate["source_metadata"],
                    "content_fingerprint": content_fingerprint,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, args.output)

    domain_counts = Counter(row["domain"] for row in rows)
    domain_tokens = Counter()
    for row in rows:
        domain_tokens[row["domain"]] += row["response_token_count"]
    manifest_path = args.manifest or args.output.with_name("manifest.json")
    manifest = {
        "schema_version": COHORT_MANIFEST_SCHEMA,
        "cohort": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "records": len(rows),
            "chosen_tokens": sum(
                row["response_token_count"] for row in rows
            ),
        },
        "selection": {
            "seed": args.seed,
            "per_domain": args.per_domain,
            "required_domains": list(required_domains),
            "max_response_tokens": args.max_response_tokens,
            "max_sequence_tokens": args.max_sequence_tokens,
            "domain_records": dict(domain_counts),
            "domain_chosen_tokens": dict(domain_tokens),
            "domain_distinct_prompts": selected_prompt_counts,
            "policy": (
                "sha256-ranked; maximize distinct prompts before selecting "
                "additional trajectories from a prompt"
            ),
        },
        "base_checkpoint": checkpoint_identity(base_document),
        "tokenizer_identity": tokenizer_identity(args.tokenizer),
        "sources": source_records,
        "exact_token_contract": (
            "Every checkpoint is scored on the stored prompt_token_ids and "
            "response_token_ids; no scoring-time tokenization is permitted."
        ),
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "cohort": str(args.output),
                "manifest": str(manifest_path),
                "records": len(rows),
                "domain_records": dict(domain_counts),
                "chosen_tokens": manifest["cohort"]["chosen_tokens"],
            },
            sort_keys=True,
        )
    )


def load_and_validate_cohort(
    cohort_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != COHORT_MANIFEST_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong cohort manifest schema")
    claimed_manifest_hash = manifest.get("manifest_payload_sha256")
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_payload_sha256", None)
    if claimed_manifest_hash != canonical_sha256(manifest_payload):
        raise ValueError(f"{manifest_path}: manifest payload fingerprint mismatch")
    expected_hash = manifest.get("cohort", {}).get("sha256")
    actual_hash = sha256_file(cohort_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"{cohort_path}: sha256 {actual_hash} != manifest {expected_hash}"
        )
    rows = [row for _, row in read_jsonl(cohort_path)]
    if len(rows) != manifest["cohort"]["records"]:
        raise ValueError("cohort row count differs from manifest")
    probe_ids: set[str] = set()
    for index, row in enumerate(rows):
        location = f"{cohort_path}:record{index}"
        if row.get("schema_version") != COHORT_SCHEMA:
            raise ValueError(f"{location}: wrong cohort schema")
        probe_id = row.get("probe_id")
        if not isinstance(probe_id, str) or probe_id in probe_ids:
            raise ValueError(f"{location}: missing or duplicate probe_id")
        probe_ids.add(probe_id)
        prompt_ids = validate_token_ids(
            row.get("prompt_token_ids"),
            location=location,
            field="prompt_token_ids",
        )
        response_ids = validate_token_ids(
            row.get("response_token_ids"),
            location=location,
            field="response_token_ids",
        )
        if len(prompt_ids) != row.get("prompt_token_count"):
            raise ValueError(f"{location}: prompt token count mismatch")
        if len(response_ids) != row.get("response_token_count"):
            raise ValueError(f"{location}: response token count mismatch")
        expected_content = canonical_sha256(
            {
                "domain": row["domain"],
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
            }
        )
        if row.get("content_fingerprint") != expected_content:
            raise ValueError(f"{location}: content fingerprint mismatch")
    return rows, manifest


def chosen_logprobs_from_logits(
    prediction_logits: Any,
    chosen_token_ids: Any,
    *,
    chunk_tokens: int,
) -> Any:
    import torch

    if prediction_logits.ndim != 2:
        raise ValueError("prediction_logits must have shape [tokens, vocab]")
    if chosen_token_ids.ndim != 1:
        raise ValueError("chosen_token_ids must have shape [tokens]")
    if prediction_logits.shape[0] != chosen_token_ids.numel():
        raise ValueError("logit and chosen-token lengths differ")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    values = []
    for start in range(0, chosen_token_ids.numel(), chunk_tokens):
        stop = min(start + chunk_tokens, chosen_token_ids.numel())
        logits = prediction_logits[start:stop].float()
        chosen = chosen_token_ids[start:stop]
        selected = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)
        values.append(selected - torch.logsumexp(logits, dim=1))
    return torch.cat(values)


def parse_torch_dtype(value: str) -> Any:
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[value]


def score_cohort(args: argparse.Namespace) -> None:
    if not LABEL.fullmatch(args.label):
        raise ValueError(f"invalid label {args.label!r}")
    rows, cohort_manifest = load_and_validate_cohort(
        args.cohort,
        args.manifest,
    )
    checkpoint_document = checkpoint_fingerprint_document(
        args.checkpoint,
        args.checkpoint_fingerprint,
    )
    # Scoring consumes frozen token IDs directly, so the relevant tokenizer is
    # the authority that defines their ID namespace.  A merged checkpoint may
    # contain a byte-different reserialization of the same tokenizer; requiring
    # that incidental file to hash identically would reject valid descendants.
    tokenizer_reference = (
        args.tokenizer_reference
        if args.tokenizer_reference is not None
        else args.checkpoint
    )
    score_tokenizer = tokenizer_identity(tokenizer_reference)
    if score_tokenizer != cohort_manifest["tokenizer_identity"]:
        raise ValueError(
            f"{args.label}: tokenizer-reference identity differs from probe "
            "cohort"
        )

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = parse_torch_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU scoring requires --dtype float32")
    print(f"[load] {args.label}: {args.checkpoint}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    model.to(device)
    vocab_size = int(model.config.vocab_size)

    vectors = []
    offsets = [0]
    probe_ids = []
    domains = []
    means = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt_ids = row["prompt_token_ids"]
            response_ids = row["response_token_ids"]
            sequence = torch.tensor(
                [*prompt_ids, *response_ids],
                dtype=torch.long,
                device=device,
            )
            if int(sequence.max()) >= vocab_size:
                raise ValueError(
                    f"{row['probe_id']}: token ID exceeds vocab {vocab_size}"
                )
            output = model(
                input_ids=sequence.unsqueeze(0),
                attention_mask=torch.ones_like(sequence).unsqueeze(0),
                use_cache=False,
            )
            start = len(prompt_ids) - 1
            stop = start + len(response_ids)
            chosen = torch.tensor(
                response_ids,
                dtype=torch.long,
                device=device,
            )
            logprobs = chosen_logprobs_from_logits(
                output.logits[0, start:stop, :],
                chosen,
                chunk_tokens=args.logit_chunk_tokens,
            )
            if not torch.isfinite(logprobs).all():
                raise RuntimeError(
                    f"{row['probe_id']}: non-finite chosen log-probability"
                )
            array = logprobs.detach().cpu().numpy().astype(np.float32)
            vectors.append(array)
            offsets.append(offsets[-1] + len(array))
            probe_ids.append(row["probe_id"])
            domains.append(row["domain"])
            means.append(float(array.mean()))
            del output, sequence, chosen, logprobs
            print(
                f"[score {args.label}] {index}/{len(rows)} "
                f"{row['probe_id']} tokens={len(array)}",
                flush=True,
            )

    flat = np.concatenate(vectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probe_ids=np.asarray(probe_ids),
            domains=np.asarray(domains),
            offsets=np.asarray(offsets, dtype=np.int64),
            chosen_logprobs=flat,
        )
    os.replace(temporary, args.output)
    score_manifest = {
        "schema_version": SCORE_SCHEMA,
        "label": args.label,
        "definition": (
            "log p_checkpoint(response_token_t | exact prompt and prior "
            "stored response tokens)"
        ),
        "checkpoint": checkpoint_identity(checkpoint_document),
        "tokenizer_identity": score_tokenizer,
        "tokenizer_reference": str(tokenizer_reference.resolve()),
        "cohort": {
            "path": str(args.cohort.resolve()),
            "sha256": sha256_file(args.cohort),
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "scoring": {
            "device": str(device),
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "records": len(rows),
            "chosen_tokens": len(flat),
            "chosen_logprob_mean": float(flat.mean()),
            "per_probe_logprob_mean": means,
        },
        "data": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
    }
    score_manifest_path = args.output.with_suffix(".json")
    write_json_atomic(score_manifest_path, score_manifest)
    print(
        json.dumps(
            {
                "score": str(args.output),
                "manifest": str(score_manifest_path),
                "tokens": len(flat),
            },
            sort_keys=True,
        )
    )


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not LABEL.fullmatch(label) or not raw_path:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    return label, Path(raw_path)


def load_score_artifact(
    label: str,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCORE_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong score schema")
    if manifest.get("label") != label:
        raise ValueError(
            f"{manifest_path}: label {manifest.get('label')!r} != {label!r}"
        )
    if manifest.get("data", {}).get("sha256") != sha256_file(path):
        raise ValueError(f"{path}: data fingerprint mismatch")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            "probe_ids": payload["probe_ids"].astype(str),
            "domains": payload["domains"].astype(str),
            "offsets": payload["offsets"].astype(np.int64),
            "chosen_logprobs": payload["chosen_logprobs"].astype(np.float64),
        }
    if arrays["offsets"][0] != 0:
        raise ValueError(f"{path}: offsets must start at zero")
    if arrays["offsets"][-1] != len(arrays["chosen_logprobs"]):
        raise ValueError(f"{path}: final offset differs from vector length")
    if len(arrays["probe_ids"]) + 1 != len(arrays["offsets"]):
        raise ValueError(f"{path}: probe/offset lengths differ")
    if np.any(np.diff(arrays["offsets"]) <= 0):
        raise ValueError(f"{path}: every probe must contain chosen tokens")
    if not np.isfinite(arrays["chosen_logprobs"]).all():
        raise ValueError(f"{path}: chosen log-probabilities are not finite")
    return manifest, arrays


def exact_cosine(left: Any, right: Any) -> float | None:
    import numpy as np

    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = math.sqrt(
        float(np.dot(left64, left64))
        * float(np.dot(right64, right64))
    )
    if denominator == 0:
        return None
    return float(np.dot(left64, right64) / denominator)


def geometry_for_vectors(
    vectors: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    squared_norms = {
        label: float(np.dot(vector, vector))
        for label, vector in vectors.items()
    }
    pairs = {}
    for left, right in itertools.combinations(vectors, 2):
        dot = float(np.dot(vectors[left], vectors[right]))
        pairs[f"{left}_vs_{right}"] = {
            "dot": dot,
            "cosine": exact_cosine(vectors[left], vectors[right]),
        }
    return {
        "exact_norms": {
            label: math.sqrt(value)
            for label, value in squared_norms.items()
        },
        "pairs": pairs,
    }


def spectrum_summary(report: dict[str, Any]) -> dict[str, Any]:
    arms = {}
    for label, arm in report.get("arms", {}).items():
        arms[label] = {
            "variance_decomposition": arm.get("variance_decomposition"),
            "effective_rank": arm.get("spectrum"),
            "disjoint_support_cosine": arm.get(
                "disjoint_subset_cosine"
            ),
        }
    return {
        "schema_version": report.get("schema_version"),
        "responses": report.get("responses"),
        "prompts": report.get("prompts"),
        "samples_per_prompt": report.get("samples_per_prompt"),
        "arms": arms,
    }


def assess_gradient_spectrum_reuse(
    *,
    current_base_fingerprint: str,
    spectrum_report: Path | None,
    spectrum_cohort_manifest: Path | None,
    spectrum_attestation: Path | None,
) -> dict[str, Any]:
    requested = any(
        path is not None
        for path in (
            spectrum_report,
            spectrum_cohort_manifest,
            spectrum_attestation,
        )
    )
    if not requested:
        return {"status": "not_requested", "reused": False}
    if spectrum_report is None or spectrum_cohort_manifest is None:
        return {
            "status": "not_reused_missing_artifact_paths",
            "reused": False,
        }
    if spectrum_attestation is None or not spectrum_attestation.is_file():
        return {
            "status": "not_reused_missing_base_attestation",
            "reused": False,
            "reason": (
                "A path alone cannot prove which base produced an existing "
                "gradient-spectrum artifact."
            ),
        }
    attestation = json.loads(
        spectrum_attestation.read_text(encoding="utf-8")
    )
    if attestation.get("schema_version") != ATTESTATION_SCHEMA:
        return {
            "status": "not_reused_bad_attestation_schema",
            "reused": False,
        }
    checks = {
        "base_fingerprint": (
            attestation.get("base_checkpoint", {}).get("fingerprint")
            == current_base_fingerprint
        ),
        "spectrum_report_sha256": (
            attestation.get("gradient_spectrum_report", {}).get("sha256")
            == sha256_file(spectrum_report)
        ),
        "cohort_manifest_sha256": (
            attestation.get("cohort_manifest", {}).get("sha256")
            == sha256_file(spectrum_cohort_manifest)
        ),
    }
    if not all(checks.values()):
        return {
            "status": "not_reused_fingerprint_mismatch",
            "reused": False,
            "checks": checks,
            "attested_base_fingerprint": attestation.get(
                "base_checkpoint",
                {},
            ).get("fingerprint"),
            "current_base_fingerprint": current_base_fingerprint,
        }
    report = json.loads(spectrum_report.read_text(encoding="utf-8"))
    return {
        "status": "reused_verified_current_base",
        "reused": True,
        "checks": checks,
        "attestation": str(spectrum_attestation.resolve()),
        "summary": spectrum_summary(report),
    }


def analyze_policy_deltas(args: argparse.Namespace) -> None:
    import numpy as np

    entries = [parse_labeled_path(value) for value in args.score]
    if len({label for label, _ in entries}) != len(entries):
        raise ValueError("score labels must be unique")
    if args.base_label not in {label for label, _ in entries}:
        raise ValueError(f"base label {args.base_label!r} is missing")
    manifests = {}
    arrays = {}
    for label, path in entries:
        manifests[label], arrays[label] = load_score_artifact(label, path)
    base_arrays = arrays[args.base_label]
    for label, values in arrays.items():
        for field in ("probe_ids", "domains", "offsets"):
            if not np.array_equal(values[field], base_arrays[field]):
                raise ValueError(
                    f"{label}: {field} differs from base artifact"
                )
        if (
            manifests[label]["cohort"]["sha256"]
            != manifests[args.base_label]["cohort"]["sha256"]
        ):
            raise ValueError(f"{label}: cohort fingerprint differs from base")

    deltas = {
        label: values["chosen_logprobs"]
        - base_arrays["chosen_logprobs"]
        for label, values in arrays.items()
        if label != args.base_label
    }
    if len(deltas) < 2:
        raise ValueError("at least two non-base checkpoints are required")
    offsets = base_arrays["offsets"]
    domains = base_arrays["domains"]
    per_domain_indices: dict[str, list[int]] = {"all": []}
    for record_index, domain in enumerate(domains):
        indices = list(range(offsets[record_index], offsets[record_index + 1]))
        per_domain_indices["all"].extend(indices)
        per_domain_indices.setdefault(str(domain), []).extend(indices)

    domain_geometry = {}
    for domain, indices in per_domain_indices.items():
        index_array = np.asarray(indices, dtype=np.int64)
        domain_vectors = {
            label: vector[index_array]
            for label, vector in deltas.items()
        }
        domain_geometry[domain] = {
            "trajectories": (
                len(domains)
                if domain == "all"
                else int(np.sum(domains == domain))
            ),
            "chosen_tokens": len(indices),
            **geometry_for_vectors(domain_vectors),
        }

    delta_output = args.delta_output or args.output.with_suffix(".npz")
    delta_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = delta_output.with_suffix(delta_output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probe_ids=base_arrays["probe_ids"],
            domains=domains,
            offsets=offsets,
            **{
                f"delta__{label}": vector.astype(np.float32)
                for label, vector in deltas.items()
            },
        )
    os.replace(temporary, delta_output)

    base_fingerprint = manifests[args.base_label]["checkpoint"][
        "fingerprint"
    ]
    spectrum_reuse = assess_gradient_spectrum_reuse(
        current_base_fingerprint=base_fingerprint,
        spectrum_report=args.gradient_spectrum_report,
        spectrum_cohort_manifest=args.gradient_spectrum_cohort_manifest,
        spectrum_attestation=args.gradient_spectrum_attestation,
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "definition": (
            "For every stored chosen token, delta_model = log "
            "p_model(y_t|x,y_<t) - log p_base(y_t|x,y_<t); domain vectors "
            "concatenate exact-token deltas without retokenization."
        ),
        "base_label": args.base_label,
        "cohort": manifests[args.base_label]["cohort"],
        "models": {
            label: manifest["checkpoint"]
            for label, manifest in manifests.items()
        },
        "score_artifacts": {
            label: {
                "path": str(path.resolve()),
                "sha256": manifests[label]["data"]["sha256"],
                "manifest": str(path.with_suffix(".json").resolve()),
            }
            for label, path in entries
        },
        "delta_artifact": {
            "path": str(delta_output.resolve()),
            "sha256": sha256_file(delta_output),
        },
        "domain_geometry": domain_geometry,
        "gradient_spectrum_reuse": spectrum_reuse,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "delta_artifact": str(delta_output),
                "domains": {
                    domain: values["chosen_tokens"]
                    for domain, values in domain_geometry.items()
                },
                "gradient_spectrum": spectrum_reuse["status"],
            },
            sort_keys=True,
        )
    )


def frozen_topk_from_logits(
    prediction_logits: Any,
    *,
    top_k: int,
    chunk_tokens: int,
) -> tuple[Any, Any]:
    import torch

    if prediction_logits.ndim != 2:
        raise ValueError("prediction_logits must have shape [tokens, vocab]")
    if not 0 < top_k <= prediction_logits.shape[1]:
        raise ValueError("top_k must be in [1, vocab_size]")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    token_ids = []
    logprobs = []
    for start in range(0, prediction_logits.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, prediction_logits.shape[0])
        logits = prediction_logits[start:stop].float()
        values, indices = torch.topk(
            logits,
            k=top_k,
            dim=1,
            sorted=True,
        )
        token_ids.append(indices)
        logprobs.append(values - torch.logsumexp(logits, dim=1, keepdim=True))
    return torch.cat(token_ids), torch.cat(logprobs)


def frozen_ids_logprobs_from_logits(
    prediction_logits: Any,
    frozen_token_ids: Any,
    *,
    chunk_tokens: int,
) -> Any:
    import torch

    if prediction_logits.ndim != 2 or frozen_token_ids.ndim != 2:
        raise ValueError("logits and frozen IDs must both be matrices")
    if prediction_logits.shape[0] != frozen_token_ids.shape[0]:
        raise ValueError("logit and frozen-ID position counts differ")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    values = []
    for start in range(0, prediction_logits.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, prediction_logits.shape[0])
        logits = prediction_logits[start:stop].float()
        ids = frozen_token_ids[start:stop]
        selected = logits.gather(1, ids)
        values.append(
            selected - torch.logsumexp(logits, dim=1, keepdim=True)
        )
    return torch.cat(values)


def scoring_tokenizer_identity(
    *,
    checkpoint: Path,
    tokenizer_reference: Path | None,
    cohort_manifest: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], Path]:
    reference = (
        tokenizer_reference
        if tokenizer_reference is not None
        else checkpoint
    )
    identity = tokenizer_identity(reference)
    if identity != cohort_manifest["tokenizer_identity"]:
        raise ValueError(
            f"{label}: tokenizer-reference identity differs from probe cohort"
        )
    return identity, reference


def score_fisher_reference(args: argparse.Namespace) -> None:
    if args.top_k <= 1:
        raise ValueError("--top-k must be at least two")
    rows, cohort_manifest = load_and_validate_cohort(
        args.cohort,
        args.manifest,
    )
    checkpoint_document = checkpoint_fingerprint_document(
        args.checkpoint,
        args.checkpoint_fingerprint,
    )
    checkpoint = checkpoint_identity(checkpoint_document)
    expected_base = cohort_manifest["base_checkpoint"]["fingerprint"]
    if checkpoint["fingerprint"] != expected_base:
        raise ValueError(
            "Fisher reference checkpoint is not the cohort's fingerprinted base"
        )
    score_tokenizer, tokenizer_reference = scoring_tokenizer_identity(
        checkpoint=args.checkpoint,
        tokenizer_reference=args.tokenizer_reference,
        cohort_manifest=cohort_manifest,
        label="fisher-base",
    )

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = parse_torch_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU scoring requires --dtype float32")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    model.to(device)
    if args.top_k > int(model.config.vocab_size):
        raise ValueError("--top-k exceeds model vocabulary")

    topk_ids = []
    base_logprobs = []
    position_offsets = [0]
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt_ids = row["prompt_token_ids"]
            response_ids = row["response_token_ids"]
            sequence = torch.tensor(
                [*prompt_ids, *response_ids],
                dtype=torch.long,
                device=device,
            )
            output = model(
                input_ids=sequence.unsqueeze(0),
                attention_mask=torch.ones_like(sequence).unsqueeze(0),
                use_cache=False,
            )
            start = len(prompt_ids) - 1
            stop = start + len(response_ids)
            ids, logprobs = frozen_topk_from_logits(
                output.logits[0, start:stop, :],
                top_k=args.top_k,
                chunk_tokens=args.logit_chunk_tokens,
            )
            topk_ids.append(ids.cpu().numpy().astype(np.int32))
            base_logprobs.append(
                logprobs.cpu().numpy().astype(np.float32)
            )
            position_offsets.append(position_offsets[-1] + len(response_ids))
            del output, sequence, ids, logprobs
            print(
                f"[fisher-base] {index}/{len(rows)} {row['probe_id']} "
                f"positions={len(response_ids)}",
                flush=True,
            )

    ids_array = np.concatenate(topk_ids)
    logprobs_array = np.concatenate(base_logprobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probe_ids=np.asarray([row["probe_id"] for row in rows]),
            domains=np.asarray([row["domain"] for row in rows]),
            position_offsets=np.asarray(position_offsets, dtype=np.int64),
            topk_token_ids=ids_array,
            base_logprobs=logprobs_array,
        )
    os.replace(temporary, args.output)
    retained_mass = np.exp(logprobs_array.astype(np.float64)).sum(axis=1)
    score_manifest = {
        "schema_version": FISHER_REFERENCE_SCHEMA,
        "definition": (
            "At each exact probe response position, freeze the base model's "
            "top-K token IDs and their full-vocabulary-normalized log "
            "probabilities."
        ),
        "base_checkpoint": checkpoint,
        "tokenizer_identity": score_tokenizer,
        "tokenizer_reference": str(tokenizer_reference.resolve()),
        "cohort": {
            "path": str(args.cohort.resolve()),
            "sha256": sha256_file(args.cohort),
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "scoring": {
            "device": str(device),
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "top_k": args.top_k,
            "records": len(rows),
            "positions": len(ids_array),
            "retained_base_probability_mass_mean": float(
                retained_mass.mean()
            ),
            "retained_base_probability_mass_min": float(
                retained_mass.min()
            ),
        },
        "data": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
    }
    write_json_atomic(args.output.with_suffix(".json"), score_manifest)
    print(
        json.dumps(
            {
                "reference": str(args.output),
                "positions": len(ids_array),
                "top_k": args.top_k,
                "retained_mass_mean": float(retained_mass.mean()),
            },
            sort_keys=True,
        )
    )


def load_fisher_reference(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FISHER_REFERENCE_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong Fisher reference schema")
    if manifest.get("data", {}).get("sha256") != sha256_file(path):
        raise ValueError(f"{path}: Fisher reference fingerprint mismatch")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            "probe_ids": payload["probe_ids"].astype(str),
            "domains": payload["domains"].astype(str),
            "position_offsets": payload["position_offsets"].astype(np.int64),
            "topk_token_ids": payload["topk_token_ids"].astype(np.int64),
            "base_logprobs": payload["base_logprobs"].astype(np.float64),
        }
    offsets = arrays["position_offsets"]
    ids = arrays["topk_token_ids"]
    logprobs = arrays["base_logprobs"]
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"{path}: invalid Fisher position offsets")
    if offsets[-1] != len(ids) or ids.shape != logprobs.shape:
        raise ValueError(f"{path}: Fisher reference array shapes differ")
    if len(arrays["probe_ids"]) + 1 != len(offsets):
        raise ValueError(f"{path}: Fisher probe/offset lengths differ")
    if ids.shape[1] != manifest["scoring"]["top_k"]:
        raise ValueError(f"{path}: top-K width differs from manifest")
    if np.any(ids < 0) or not np.isfinite(logprobs).all():
        raise ValueError(f"{path}: invalid frozen IDs or log-probabilities")
    if np.any(logprobs > 1e-6):
        raise ValueError(f"{path}: positive base log-probability detected")
    return manifest, arrays


def score_frozen_fisher_ids(args: argparse.Namespace) -> None:
    if not LABEL.fullmatch(args.label) or args.label == "base":
        raise ValueError("endpoint label must be valid and non-base")
    rows, cohort_manifest = load_and_validate_cohort(
        args.cohort,
        args.manifest,
    )
    reference_manifest, reference = load_fisher_reference(args.reference)
    if (
        reference_manifest["cohort"]["sha256"]
        != cohort_manifest["cohort"]["sha256"]
    ):
        raise ValueError("Fisher reference cohort differs from requested cohort")
    if not all(
        left == right
        for left, right in zip(
            reference["probe_ids"],
            (row["probe_id"] for row in rows),
            strict=True,
        )
    ):
        raise ValueError("Fisher reference probe order differs from cohort")
    expected_base = cohort_manifest["base_checkpoint"]["fingerprint"]
    if (
        reference_manifest["base_checkpoint"]["fingerprint"]
        != expected_base
    ):
        raise ValueError("Fisher reference was not produced by current base")

    checkpoint_document = checkpoint_fingerprint_document(
        args.checkpoint,
        args.checkpoint_fingerprint,
    )
    score_tokenizer, tokenizer_reference = scoring_tokenizer_identity(
        checkpoint=args.checkpoint,
        tokenizer_reference=args.tokenizer_reference,
        cohort_manifest=cohort_manifest,
        label=args.label,
    )

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = parse_torch_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU scoring requires --dtype float32")
    device_map = getattr(args, "device_map", None)
    # A checkpoint whose weights do not fit on one card would otherwise force a
    # lower dtype than the arms it is compared against, and that dtype shift
    # would be charged to the comparison rather than to the checkpoint.
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        **({"device_map": device_map} if device_map else {}),
    ).eval()
    if device_map:
        device = model.get_input_embeddings().weight.device
    else:
        model.to(device)
    if int(reference["topk_token_ids"].max()) >= int(model.config.vocab_size):
        raise ValueError("frozen base token ID exceeds endpoint vocabulary")

    endpoint_logprobs = []
    offsets = reference["position_offsets"]
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt_ids = row["prompt_token_ids"]
            response_ids = row["response_token_ids"]
            sequence = torch.tensor(
                [*prompt_ids, *response_ids],
                dtype=torch.long,
                device=device,
            )
            output = model(
                input_ids=sequence.unsqueeze(0),
                attention_mask=torch.ones_like(sequence).unsqueeze(0),
                use_cache=False,
            )
            start = len(prompt_ids) - 1
            stop = start + len(response_ids)
            prediction_logits = output.logits[0, start:stop, :]
            frozen_ids = torch.as_tensor(
                reference["topk_token_ids"][
                    offsets[index - 1] : offsets[index]
                ],
                dtype=torch.long,
                device=prediction_logits.device,
            )
            logprobs = frozen_ids_logprobs_from_logits(
                prediction_logits,
                frozen_ids,
                chunk_tokens=args.logit_chunk_tokens,
            )
            endpoint_logprobs.append(
                logprobs.cpu().numpy().astype(np.float32)
            )
            del output, sequence, frozen_ids, logprobs, prediction_logits
            print(
                f"[fisher-score {args.label}] {index}/{len(rows)} "
                f"{row['probe_id']} positions={len(response_ids)}",
                flush=True,
            )

    logprobs_array = np.concatenate(endpoint_logprobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probe_ids=reference["probe_ids"],
            domains=reference["domains"],
            position_offsets=offsets,
            scored_logprobs=logprobs_array,
        )
    os.replace(temporary, args.output)
    score_manifest = {
        "schema_version": FISHER_SCORE_SCHEMA,
        "label": args.label,
        "definition": (
            "Endpoint full-vocabulary-normalized log probabilities evaluated "
            "only at token IDs frozen by the fingerprinted base reference."
        ),
        "checkpoint": checkpoint_identity(checkpoint_document),
        "tokenizer_identity": score_tokenizer,
        "tokenizer_reference": str(tokenizer_reference.resolve()),
        "cohort": reference_manifest["cohort"],
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
            "manifest_path": str(
                args.reference.with_suffix(".json").resolve()
            ),
            "manifest_sha256": sha256_file(
                args.reference.with_suffix(".json")
            ),
            "base_checkpoint": reference_manifest["base_checkpoint"],
            "top_k": reference_manifest["scoring"]["top_k"],
        },
        "scoring": {
            "device": str(device),
            "device_map": device_map,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "records": len(rows),
            "positions": len(logprobs_array),
        },
        "data": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
    }
    write_json_atomic(args.output.with_suffix(".json"), score_manifest)
    print(
        json.dumps(
            {
                "score": str(args.output),
                "positions": len(logprobs_array),
                "top_k": logprobs_array.shape[1],
            },
            sort_keys=True,
        )
    )


def load_fisher_score(
    label: str,
    path: Path,
    *,
    reference_path: Path,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    import numpy as np

    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FISHER_SCORE_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong Fisher score schema")
    if manifest.get("label") != label:
        raise ValueError(f"{manifest_path}: endpoint label mismatch")
    if manifest.get("data", {}).get("sha256") != sha256_file(path):
        raise ValueError(f"{path}: Fisher endpoint fingerprint mismatch")
    if (
        manifest.get("reference", {}).get("sha256")
        != sha256_file(reference_path)
    ):
        raise ValueError(f"{path}: frozen-base reference mismatch")
    with np.load(path, allow_pickle=False) as payload:
        probe_ids = payload["probe_ids"].astype(str)
        domains = payload["domains"].astype(str)
        offsets = payload["position_offsets"].astype(np.int64)
        logprobs = payload["scored_logprobs"].astype(np.float64)
    for name, value in (
        ("probe_ids", probe_ids),
        ("domains", domains),
        ("position_offsets", offsets),
    ):
        if not np.array_equal(value, reference[name]):
            raise ValueError(f"{path}: {name} differs from Fisher reference")
    if logprobs.shape != reference["base_logprobs"].shape:
        raise ValueError(f"{path}: endpoint/reference score shapes differ")
    if not np.isfinite(logprobs).all() or np.any(logprobs > 1e-6):
        raise ValueError(f"{path}: invalid endpoint log-probabilities")
    return manifest, logprobs


def fisher_centered_vector(
    base_logprobs: Any,
    endpoint_logprobs: Any,
) -> tuple[Any, Any]:
    import numpy as np

    base = np.asarray(base_logprobs, dtype=np.float64)
    endpoint = np.asarray(endpoint_logprobs, dtype=np.float64)
    if base.shape != endpoint.shape or base.ndim != 2:
        raise ValueError("base and endpoint top-K matrices must match")
    probabilities = np.exp(base)
    retained_mass = probabilities.sum(axis=1)
    if np.any(retained_mass <= 0) or not np.isfinite(retained_mass).all():
        raise ValueError("invalid retained base probability mass")
    delta_logprob = endpoint - base
    conditional_weights = probabilities / retained_mass[:, None]
    center = (conditional_weights * delta_logprob).sum(axis=1)
    centered = delta_logprob - center[:, None]
    vector = np.sqrt(probabilities) * centered
    return vector, retained_mass


def bootstrap_trajectory_cosine(
    left: Any,
    right: Any,
    *,
    position_offsets: Any,
    trajectory_indices: list[int],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    if not trajectory_indices:
        raise ValueError("bootstrap trajectory set is empty")
    sufficient = []
    for index in trajectory_indices:
        block_left = left[
            position_offsets[index] : position_offsets[index + 1]
        ].reshape(-1)
        block_right = right[
            position_offsets[index] : position_offsets[index + 1]
        ].reshape(-1)
        sufficient.append(
            (
                float(np.dot(block_left, block_left)),
                float(np.dot(block_right, block_right)),
                float(np.dot(block_left, block_right)),
            )
        )
    stats = np.asarray(sufficient, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(
        0,
        len(trajectory_indices),
        size=(draws, len(trajectory_indices)),
    )
    totals = stats[sampled].sum(axis=1)
    denominator = np.sqrt(totals[:, 0] * totals[:, 1])
    valid = denominator > 0
    if not np.all(valid):
        raise ValueError("bootstrap encountered a zero-norm resample")
    values = totals[:, 2] / denominator
    return {
        "draws": draws,
        "seed": seed,
        "sampling_unit": "whole_probe_trajectory",
        "cosine_mean": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def fisher_domain_geometry(
    vectors: dict[str, Any],
    retained_mass: Any,
    *,
    position_offsets: Any,
    trajectory_indices: list[int],
) -> dict[str, Any]:
    import numpy as np

    rows = np.concatenate(
        [
            np.arange(
                position_offsets[index],
                position_offsets[index + 1],
                dtype=np.int64,
            )
            for index in trajectory_indices
        ]
    )
    selected_mass = retained_mass[rows]
    total_mass = float(selected_mass.sum())
    selected = {label: vector[rows] for label, vector in vectors.items()}
    squared_norms = {
        label: float(np.sum(value * value, dtype=np.float64))
        for label, value in selected.items()
    }
    endpoint_rms = {
        label: math.sqrt(value / total_mass)
        for label, value in squared_norms.items()
    }
    pairs = {}
    for left, right in itertools.combinations(selected, 2):
        left_vector = selected[left].reshape(-1)
        right_vector = selected[right].reshape(-1)
        difference = left_vector - right_vector
        pairs[f"{left}_vs_{right}"] = {
            "dot": float(np.dot(left_vector, right_vector)),
            "cosine": exact_cosine(left_vector, right_vector),
            "weighted_rms_difference": math.sqrt(
                float(np.dot(difference, difference)) / total_mass
            ),
        }
    ratios = {}
    if {"short", "long"}.issubset(endpoint_rms):
        short_long = pairs.get("short_vs_long") or pairs["long_vs_short"]
        ratios.update(
            {
                "long_over_short_fisher_norm": (
                    math.sqrt(squared_norms["long"])
                    / math.sqrt(squared_norms["short"])
                ),
                "short_long_difference_over_short_rms": (
                    short_long["weighted_rms_difference"]
                    / endpoint_rms["short"]
                ),
                "short_long_difference_over_long_rms": (
                    short_long["weighted_rms_difference"]
                    / endpoint_rms["long"]
                ),
            }
        )
    if {"short", "long", "m48"}.issubset(endpoint_rms):
        ratios.update(
            {
                "m48_over_short_fisher_norm": (
                    math.sqrt(squared_norms["m48"])
                    / math.sqrt(squared_norms["short"])
                ),
                "m48_over_long_fisher_norm": (
                    math.sqrt(squared_norms["m48"])
                    / math.sqrt(squared_norms["long"])
                ),
            }
        )
    return {
        "trajectories": len(trajectory_indices),
        "positions": len(rows),
        "retained_base_probability_mass": {
            "mean": float(selected_mass.mean()),
            "min": float(selected_mass.min()),
            "p05": float(np.quantile(selected_mass, 0.05)),
            "median": float(np.quantile(selected_mass, 0.5)),
            "p95": float(np.quantile(selected_mass, 0.95)),
            "max": float(selected_mass.max()),
            "sum": total_mass,
        },
        "fisher_weighted_norms": {
            label: math.sqrt(value)
            for label, value in squared_norms.items()
        },
        "fisher_weighted_rms_from_base": endpoint_rms,
        "pairs": pairs,
        "short_long_m48_ratios": ratios,
    }


def analyze_fisher_geometry(args: argparse.Namespace) -> None:
    import numpy as np

    reference_manifest, reference = load_fisher_reference(args.reference)
    entries = [parse_labeled_path(value) for value in args.score]
    if len({label for label, _ in entries}) != len(entries):
        raise ValueError("Fisher score labels must be unique")
    if len(entries) < 2:
        raise ValueError("at least two Fisher endpoint scores are required")
    manifests = {}
    endpoint_logprobs = {}
    for label, path in entries:
        manifests[label], endpoint_logprobs[label] = load_fisher_score(
            label,
            path,
            reference_path=args.reference,
            reference=reference,
        )
        if (
            manifests[label]["reference"]["base_checkpoint"]["fingerprint"]
            != reference_manifest["base_checkpoint"]["fingerprint"]
        ):
            raise ValueError(f"{label}: base fingerprint differs")

    vectors = {}
    retained_mass = None
    for label, values in endpoint_logprobs.items():
        vector, current_mass = fisher_centered_vector(
            reference["base_logprobs"],
            values,
        )
        vectors[label] = vector
        if retained_mass is None:
            retained_mass = current_mass
        elif not np.array_equal(retained_mass, current_mass):
            raise RuntimeError("base retained mass changed across endpoints")
    assert retained_mass is not None

    domains = reference["domains"]
    domain_trajectories = {
        "all": list(range(len(domains))),
        **{
            str(domain): [
                index
                for index, value in enumerate(domains)
                if value == domain
            ]
            for domain in dict.fromkeys(domains)
        },
    }
    domain_geometry = {}
    for domain_index, (domain, trajectory_indices) in enumerate(
        domain_trajectories.items()
    ):
        geometry = fisher_domain_geometry(
            vectors,
            retained_mass,
            position_offsets=reference["position_offsets"],
            trajectory_indices=trajectory_indices,
        )
        if (
            args.bootstrap_left in vectors
            and args.bootstrap_right in vectors
        ):
            geometry["trajectory_bootstrap"] = {
                f"{args.bootstrap_left}_vs_{args.bootstrap_right}": (
                    bootstrap_trajectory_cosine(
                        vectors[args.bootstrap_left],
                        vectors[args.bootstrap_right],
                        position_offsets=reference["position_offsets"],
                        trajectory_indices=trajectory_indices,
                        draws=args.bootstrap_draws,
                        seed=args.bootstrap_seed + domain_index,
                    )
                )
            }
        domain_geometry[domain] = geometry

    report = {
        "schema_version": FISHER_REPORT_SCHEMA,
        "representation": {
            "frozen_support": (
                "At every exact context, token IDs are selected once by BASE "
                "and reused unchanged for all endpoints."
            ),
            "raw_delta": (
                "delta_i = log p_endpoint(token_i|context) - "
                "log p_base(token_i|context)"
            ),
            "centering": (
                "For each context independently, subtract the base-probability "
                "weighted mean delta over the frozen top-K, with weights "
                "renormalized only for computing that mean."
            ),
            "vector": (
                "v_i = sqrt(p_base(token_i|context)) * centered_delta_i; "
                "the original unnormalized top-K base probability weights are "
                "retained in the metric."
            ),
            "invariance": (
                "Per-context centering makes the representation invariant to "
                "any additive constant in endpoint-minus-base logits; using "
                "normalized log probabilities already removes each model's "
                "own log-partition shift."
            ),
            "approximation": (
                "This is a frozen top-K approximation to the base Fisher "
                "metric. Retained base probability mass is reported for every "
                "domain and must be inspected before interpretation."
            ),
        },
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
            "manifest": str(args.reference.with_suffix(".json").resolve()),
            "base_checkpoint": reference_manifest["base_checkpoint"],
            "cohort": reference_manifest["cohort"],
            "top_k": reference_manifest["scoring"]["top_k"],
        },
        "models": {
            label: manifest["checkpoint"]
            for label, manifest in manifests.items()
        },
        "score_artifacts": {
            label: {
                "path": str(path.resolve()),
                "sha256": manifests[label]["data"]["sha256"],
                "manifest": str(path.with_suffix(".json").resolve()),
            }
            for label, path in entries
        },
        "bootstrap": {
            "left": args.bootstrap_left,
            "right": args.bootstrap_right,
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
        },
        "domain_geometry": domain_geometry,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "top_k": reference_manifest["scoring"]["top_k"],
                "domains": list(domain_geometry),
            },
            sort_keys=True,
        )
    )


def attest_gradient_spectrum(args: argparse.Namespace) -> None:
    if not args.confirm_produced_from_base:
        raise ValueError(
            "--confirm-produced-from-base is required; do not retroactively "
            "attest an artifact whose producing checkpoint is unknown"
        )
    base_document = checkpoint_fingerprint_document(
        args.base,
        args.base_fingerprint,
    )
    cohort_manifest = json.loads(
        args.cohort_manifest.read_text(encoding="utf-8")
    )
    cohort_path = (
        args.cohort
        or args.cohort_manifest.with_name("cohort.jsonl")
    )
    expected_cohort_hash = cohort_manifest.get("cohort_sha256")
    if expected_cohort_hash != sha256_file(cohort_path):
        raise ValueError("gradient-spectrum cohort hash differs from manifest")
    attestation = {
        "schema_version": ATTESTATION_SCHEMA,
        "base_checkpoint": checkpoint_identity(base_document),
        "gradient_spectrum_report": {
            "path": str(args.report.resolve()),
            "sha256": sha256_file(args.report),
        },
        "cohort_manifest": {
            "path": str(args.cohort_manifest.resolve()),
            "sha256": sha256_file(args.cohort_manifest),
        },
        "cohort": {
            "path": str(cohort_path.resolve()),
            "sha256": sha256_file(cohort_path),
        },
        "assertion": (
            "The operator confirmed these gradient-spectrum artifacts were "
            "produced from the fingerprinted base checkpoint."
        ),
    }
    write_json_atomic(args.output, attestation)
    print(json.dumps({"attestation": str(args.output)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Freeze a balanced mixed-domain exact-token probe cohort.",
    )
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--base-fingerprint", type=Path)
    build.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="DOMAIN:ADAPTER=PATH",
    )
    build.add_argument("--required-domain", action="append")
    build.add_argument("--per-domain", type=int, default=16)
    build.add_argument("--max-response-tokens", type=int, default=1024)
    build.add_argument("--max-sequence-tokens", type=int, default=2048)
    build.add_argument("--seed", type=int, default=20260821)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest", type=Path)
    build.set_defaults(handler=build_cohort)

    score = subparsers.add_parser(
        "score",
        help="Score stored chosen tokens under one checkpoint.",
    )
    score.add_argument("--cohort", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--checkpoint-fingerprint", type=Path)
    score.add_argument(
        "--tokenizer-reference",
        type=Path,
        help=(
            "Checkpoint whose tokenizer.json defines the frozen token-ID "
            "namespace. Defaults to --checkpoint."
        ),
    )
    score.add_argument("--label", required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--device", default="cuda:0")
    score.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    score.add_argument("--attn-implementation", default="sdpa")
    score.add_argument("--logit-chunk-tokens", type=int, default=64)
    score.set_defaults(handler=score_cohort)

    analyze = subparsers.add_parser(
        "analyze",
        help="Compute model-minus-base policy-delta cosines by domain.",
    )
    analyze.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="LABEL=NPZ",
    )
    analyze.add_argument("--base-label", default="base")
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--delta-output", type=Path)
    analyze.add_argument("--gradient-spectrum-report", type=Path)
    analyze.add_argument("--gradient-spectrum-cohort-manifest", type=Path)
    analyze.add_argument("--gradient-spectrum-attestation", type=Path)
    analyze.set_defaults(handler=analyze_policy_deltas)

    fisher_base = subparsers.add_parser(
        "fisher-base",
        help="Freeze base top-K IDs and log probabilities at every position.",
    )
    fisher_base.add_argument("--cohort", type=Path, required=True)
    fisher_base.add_argument("--manifest", type=Path, required=True)
    fisher_base.add_argument("--checkpoint", type=Path, required=True)
    fisher_base.add_argument("--checkpoint-fingerprint", type=Path)
    fisher_base.add_argument("--tokenizer-reference", type=Path)
    fisher_base.add_argument("--top-k", type=int, default=128)
    fisher_base.add_argument("--output", type=Path, required=True)
    fisher_base.add_argument("--device", default="cuda:0")
    fisher_base.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    fisher_base.add_argument("--attn-implementation", default="sdpa")
    fisher_base.add_argument("--logit-chunk-tokens", type=int, default=64)
    fisher_base.set_defaults(handler=score_fisher_reference)

    fisher_score = subparsers.add_parser(
        "fisher-score",
        help="Score exactly the token IDs frozen by a Fisher base artifact.",
    )
    fisher_score.add_argument("--cohort", type=Path, required=True)
    fisher_score.add_argument("--manifest", type=Path, required=True)
    fisher_score.add_argument("--reference", type=Path, required=True)
    fisher_score.add_argument("--checkpoint", type=Path, required=True)
    fisher_score.add_argument("--checkpoint-fingerprint", type=Path)
    fisher_score.add_argument("--tokenizer-reference", type=Path)
    fisher_score.add_argument("--label", required=True)
    fisher_score.add_argument("--output", type=Path, required=True)
    fisher_score.add_argument("--device", default="cuda:0")
    fisher_score.add_argument(
        "--device-map",
        help="shard the checkpoint over the visible devices, e.g. auto; use it "
        "when the checkpoint cannot hold the dtype the other arms are scored in",
    )
    fisher_score.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    fisher_score.add_argument("--attn-implementation", default="sdpa")
    fisher_score.add_argument("--logit-chunk-tokens", type=int, default=64)
    fisher_score.set_defaults(handler=score_frozen_fisher_ids)

    fisher_analyze = subparsers.add_parser(
        "fisher-analyze",
        help="Analyze centered frozen-top-K Fisher policy deltas.",
    )
    fisher_analyze.add_argument("--reference", type=Path, required=True)
    fisher_analyze.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="LABEL=NPZ",
    )
    fisher_analyze.add_argument("--output", type=Path, required=True)
    fisher_analyze.add_argument("--bootstrap-left", default="short")
    fisher_analyze.add_argument("--bootstrap-right", default="long")
    fisher_analyze.add_argument("--bootstrap-draws", type=int, default=2000)
    fisher_analyze.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260821,
    )
    fisher_analyze.set_defaults(handler=analyze_fisher_geometry)

    attest = subparsers.add_parser(
        "attest-spectrum",
        help="Bind a freshly produced gradient spectrum to its exact base.",
    )
    attest.add_argument("--base", type=Path, required=True)
    attest.add_argument("--base-fingerprint", type=Path)
    attest.add_argument("--report", type=Path, required=True)
    attest.add_argument("--cohort-manifest", type=Path, required=True)
    attest.add_argument("--cohort", type=Path)
    attest.add_argument("--output", type=Path, required=True)
    attest.add_argument("--confirm-produced-from-base", action="store_true")
    attest.set_defaults(handler=attest_gradient_spectrum)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
