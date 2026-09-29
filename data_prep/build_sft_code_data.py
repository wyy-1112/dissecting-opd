#!/usr/bin/env python3
"""Build the direct-code SFT data of the Qwen3-1.7B-Base student (SFT-554).

Reloads the pinned source revisions in data_prep/sft_code_data.yaml and emits one user
turn followed by one assistant turn containing exactly one Python code block and no
thinking tags, after near-duplicate removal and decontamination against HumanEval+,
MBPP+ and LiveCodeBench v6 (downloaded at pinned versions).

    python data_prep/build_sft_code_data.py            # writes $OUTPUT_ROOT/sft_code_data
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
DECONTAM_CACHE = Path(os.environ.get("DECONTAM_CACHE", Path.home() / ".cache" / "opd_decontam"))


def load_data_profile(path: Path, expected_schema: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema_version") != expected_schema:
        raise SystemExit(f"{path}: expected schema_version {expected_schema!r}")
    return data


def expand_path(value: str) -> str:
    variables = {"REPO_ROOT": str(REPO_ROOT), **os.environ}
    variables.setdefault("OUTPUT_ROOT", str(REPO_ROOT / "outputs"))
    expanded = re.sub(r"\$\{(\w+)\}", lambda m: variables.get(m.group(1), m.group(0)), value)
    if re.search(r"\$\{\w+\}", expanded):
        raise SystemExit(f"unresolved variable in {value!r}")
    return expanded


def resolve_target(target: dict[str, Any]) -> Path:
    """Local path, pinned Hugging Face dataset file, or (gzipped) URL download."""
    if "path" in target:
        return Path(expand_path(target["path"]))
    if "hf_repo" in target:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(target["hf_repo"], target["hf_file"], repo_type="dataset",
                                    revision=target["hf_revision"]))
    import gzip
    import shutil
    import urllib.request

    destination = DECONTAM_CACHE / target["file"]
    if not destination.is_file():
        DECONTAM_CACHE.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(target["url"]) as response, open(partial, "wb") as handle:
            source = gzip.GzipFile(fileobj=response) if target["url"].endswith(".gz") else response
            shutil.copyfileobj(source, handle)
        partial.replace(destination)
    return destination


PYTHON_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+\-.]*)[ \t]*\n(?P<code>.*?)```",
    re.DOTALL,
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
THINK_MARKERS = ("<think>", "</think>", "/think", "/no_think")


@dataclass
class Candidate:
    source: str
    source_revision: str
    source_row_id: str
    native_id: str
    family_id: str
    license: str
    problem: str
    code: str
    tests_json: str
    style: str
    difficulty: str
    source_platform: str
    verification_level: str
    lineage: dict[str, Any]
    rendered: dict[str, Any] | None = None

    @property
    def problem_hash(self) -> str:
        return stable_hex(normalize_text(self.problem))


@dataclass
class EvalDocument:
    target: str
    eval_id: str
    tokens: tuple[str, ...]
    ngrams: set[str]


class Decontaminator:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.enabled = bool(cfg.get("enable", True))
        self.n = int(cfg.get("ngram_size", 13))
        self.threshold = float(cfg.get("containment_threshold", 0.5))
        self.documents: list[EvalDocument] = []
        self.index: dict[str, list[int]] = defaultdict(list)
        self.exact_documents: set[str] = set()
        if not self.enabled:
            return
        for target in cfg.get("targets", []):
            self._load_target(target["name"], resolve_target(target))
        for idx, doc in enumerate(self.documents):
            for gram in doc.ngrams:
                self.index[gram].append(idx)

    def _load_target(self, target: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing decontamination target: {path}")
        for row_index, row in enumerate(read_json_records(path)):
            eval_id = str(
                first_present(
                    row,
                    ["task_id", "question_id", "id", "problem_id", "title"],
                    f"{target}:{row_index}",
                )
            )
            for part_index, text in enumerate(flatten_strings(row)):
                tokens = tuple(normalized_tokens(text))
                if len(tokens) >= 8:
                    self.exact_documents.add(" ".join(tokens))
                if len(tokens) < self.n * 2:
                    continue
                grams = make_ngrams(tokens, self.n)
                self.documents.append(
                    EvalDocument(
                        target=target,
                        eval_id=f"{eval_id}#{part_index}",
                        tokens=tokens,
                        ngrams=grams,
                    )
                )

    def match(self, candidate: Candidate) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        for field, text in (("problem", candidate.problem), ("code", candidate.code)):
            tokens = tuple(normalized_tokens(text))
            if " ".join(tokens) in self.exact_documents:
                return {"reason": f"{field}_exact"}
            if len(tokens) < self.n:
                continue
            grams = make_ngrams(tokens, self.n)
            counts: Counter[int] = Counter()
            for gram in grams:
                counts.update(self.index.get(gram, ()))
            for doc_idx, overlap in counts.most_common():
                doc = self.documents[doc_idx]
                containment = overlap / max(1, min(len(grams), len(doc.ngrams)))
                if containment >= self.threshold:
                    return {
                        "reason": f"{field}_{self.n}gram",
                        "target": doc.target,
                        "eval_id": doc.eval_id,
                        "containment": round(containment, 4),
                    }
        return None


class StableNearDeduper:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.enabled = bool(cfg.get("enable", True))
        self.shingle_size = int(cfg.get("shingle_size", 5))
        self.bands = int(cfg.get("simhash_bands", 4))
        if 64 % self.bands:
            raise ValueError("dedup.simhash_bands must divide 64")
        self.hamming_threshold = int(cfg.get("simhash_hamming_threshold", 3))
        self.exact: dict[str, str] = {}
        self.families: dict[str, str] = {}
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.items: list[tuple[str, int]] = []

    def add(self, candidate: Candidate) -> tuple[bool, str | None]:
        normalized = normalize_text(candidate.problem)
        exact_key = stable_hex(normalized)
        if exact_key in self.exact:
            return False, self.exact[exact_key]
        family_key = f"{candidate.source}:{candidate.family_id}"
        if candidate.family_id and family_key in self.families:
            return False, self.families[family_key]
        if not self.enabled:
            self.exact[exact_key] = candidate.source_row_id
            self.families[family_key] = candidate.source_row_id
            return True, None
        fingerprint = simhash64(normalized.split())
        duplicate_of = None
        for bucket_key in simhash_band_keys(fingerprint, self.bands):
            for item_idx in self.buckets.get(bucket_key, ()):
                other_id, other_fingerprint = self.items[item_idx]
                if hamming_distance(fingerprint, other_fingerprint) <= self.hamming_threshold:
                    duplicate_of = other_id
                    break
            if duplicate_of:
                break
        if duplicate_of:
            return False, duplicate_of
        item_idx = len(self.items)
        self.items.append((candidate.source_row_id, fingerprint))
        self.exact[exact_key] = candidate.source_row_id
        self.families[family_key] = candidate.source_row_id
        for bucket_key in simhash_band_keys(fingerprint, self.bands):
            self.buckets[bucket_key].append(item_idx)
        return True, None


def stable_hex(text: str, size: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def stable_u64(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
    )


def normalize_text(text: str) -> str:
    return " ".join(normalized_tokens(text))


def normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def make_ngrams(tokens: tuple[str, ...] | list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {"\x1f".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def text_shingles(normalized: str, n: int) -> set[str]:
    tokens = normalized.split()
    if len(tokens) < n:
        return {normalized} if normalized else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def bottom_k_signature(shingles: set[str], k: int) -> tuple[int, ...]:
    values = sorted(stable_u64(value) for value in shingles)
    if not values:
        values = [0]
    if len(values) < k:
        values.extend([values[-1]] * (k - len(values)))
    return tuple(values[:k])


def band_keys(signature: tuple[int, ...], bands: int) -> list[tuple[int, tuple[int, ...]]]:
    width = len(signature) // bands
    return [(band, signature[band * width : (band + 1) * width]) for band in range(bands)]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def simhash64(tokens: list[str]) -> int:
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        value = stable_u64(token)
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def simhash_band_keys(fingerprint: int, bands: int) -> list[tuple[int, int]]:
    width = 64 // bands
    mask = (1 << width) - 1
    return [(band, (fingerprint >> (band * width)) & mask) for band in range(bands)]


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def first_present(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return default


def flatten_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if len(value.strip()) >= 24:
            yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from flatten_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from flatten_strings(nested)


def read_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    data = json.loads(path.read_text())
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ("data", "rows", "problems"):
            if isinstance(data.get(key), list):
                yield from data[key]
                return
        yield data


def parse_jsonish(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def extract_python_code(text: str) -> str | None:
    text = (text or "").strip()
    candidates: list[tuple[int, int, str]] = []
    for order, match in enumerate(PYTHON_FENCE_RE.finditer(text)):
        lang = match.group("lang").strip().lower()
        if lang not in ("", "py", "python", "python3"):
            continue
        code = clean_code(match.group("code"))
        score = code_quality_score(code)
        if score is not None:
            candidates.append((score, -order, code))
    if not candidates:
        code = clean_code(text)
        if code_quality_score(code) is not None:
            return code
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def clean_code(code: str) -> str:
    code = (code or "").strip()
    code = re.sub(r"^(?:python|python3|py)\s*\n", "", code, flags=re.IGNORECASE)
    return code.strip()


def code_quality_score(code: str) -> int | None:
    if len(code) < 20 or any(marker in code for marker in THINK_MARKERS):
        return None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    if not tree.body:
        return None
    functions = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
    classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    imports = sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
    return 10 + min(functions, 5) * 4 + min(classes, 3) * 3 + min(imports, 3)


def infer_style(code: str, tests_json: str, problem: str) -> str:
    tests_lower = tests_json.lower()
    if "assert " in tests_lower:
        return "function"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "unknown"
    has_input = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
        for node in ast.walk(tree)
    )
    has_stdio = has_input or "stdin" in code or "stdout" in code
    if has_stdio or re.search(r"\binput format\b|\boutput format\b", problem, re.IGNORECASE):
        return "competitive"
    if any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body):
        return "function"
    return "competitive"


def has_repeated_code(code: str) -> bool:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in code.splitlines() if line.strip()]
    if len(lines) < 12:
        return False
    counts = Counter(line for line in lines if len(line) >= 12)
    return bool(counts and max(counts.values()) >= 8)


def has_function_test_scaffolding(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in tree.body:
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            text = ast.unparse(node.test)
            if "__name__" in text and "__main__" in text:
                return True
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in {"print", "unittest.main"}
        ):
            return True
    return False


def normalize_difficulty(value: Any) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ("hard", "advanced", "expert")):
        return "hard"
    if any(token in text for token in ("easy", "intro", "basic")):
        return "easy"
    return "medium"


def adapt_row(
    source: str,
    spec: dict[str, Any],
    row: dict[str, Any],
    row_index: int,
) -> tuple[Candidate | None, str | None]:
    revision = spec["revision"]
    source_row_id = f"{source}:{row_index}"
    license_name = spec.get("license", "")

    if source == "open_code_instruct":
        score = parse_float(row.get("average_test_score"))
        statuses = parse_jsonish(row.get("tests_execution_status"), [])
        if score != 1.0 or not statuses or any(str(v).lower() != "pass" for v in statuses):
            return None, "not_all_tests_pass"
        problem = str(row.get("input") or "").strip()
        code = extract_python_code(str(row.get("output") or ""))
        tests = parse_jsonish(row.get("unit_tests"), [])
        tests_json = json.dumps(tests, ensure_ascii=False, sort_keys=True)
        native_id = str(row.get("id") or source_row_id)
        verification = "upstream_all_tests_pass"
        platform = str(row.get("domain") or "generic")
        family_id = native_id
        lineage = {
            "generation_algorithm": row.get("generation_algorithm"),
            "llm_judgement": row.get("llm_judgement"),
        }
        difficulty = "medium"
    elif source == "self_oss_sc2":
        problem = str(row.get("instruction") or "").strip()
        code = extract_python_code(str(row.get("response") or ""))
        tests_json = ""
        native_id = str(row.get("id") or source_row_id)
        family_id = str(row.get("sha1") or native_id)
        verification = "upstream_execution_filtered"
        platform = "oss_python"
        lineage = {"seed_sha1": row.get("sha1"), "concepts": row.get("concepts")}
        difficulty = "medium"
    elif source == "magicoder_oss_python":
        if str(row.get("lang") or "").lower() not in ("python", "py"):
            return None, "non_python"
        problem = str(row.get("problem") or "").strip()
        code = extract_python_code(str(row.get("solution") or ""))
        tests_json = ""
        native_id = str(row.get("index") or source_row_id)
        family_id = f"magicoder-seed:{row.get('raw_index', native_id)}"
        verification = "static_only"
        platform = "oss_instruct"
        lineage = {
            "raw_index": row.get("raw_index"),
            "openai_fingerprint": row.get("openai_fingerprint"),
        }
        difficulty = "medium"
    elif source == "rstar_seed_verified":
        if not bool(row.get("verified")) or not bool(row.get("is_passed")):
            return None, "not_verified_passed"
        question = str(row.get("question") or "").strip()
        starter = str(row.get("starter_code") or "").strip()
        problem = question
        if starter:
            problem += f"\n\nStarter code:\n```python\n{starter}\n```"
        code = extract_python_code(str(row.get("code") or ""))
        tests_json = ""
        native_id = str(row.get("question_id") or source_row_id)
        family_id = native_id
        verification = "upstream_verified_passed"
        platform = "competitive_programming"
        lineage = {"starter_code": bool(starter)}
        difficulty = normalize_difficulty(row.get("difficulty"))
    elif source == "rstar_synthetic_code":
        question = str(row.get("question") or "").strip()
        seed_question = str(row.get("seed_question") or "").strip()
        problem = question
        code = extract_python_code(str(row.get("code") or ""))
        tests_json = ""
        native_id = stable_hex(question)
        family_id = f"rstar-seed:{stable_hex(seed_question or question)}"
        verification = "upstream_rstar_verified_dataset"
        platform = str(row.get("seed_source") or "competitive_programming")
        lineage = {
            "seed_question_hash": stable_hex(seed_question) if seed_question else "",
            "seed_source": row.get("seed_source"),
            "ignored_response": True,
        }
        difficulty = "medium"
    else:
        return None, "unknown_source"

    if not problem:
        return None, "empty_problem"
    if code is None:
        return None, "no_valid_python"
    if has_repeated_code(code):
        return None, "repeated_code"
    style = (
        "competitive"
        if source in {"rstar_seed_verified", "rstar_synthetic_code"}
        else infer_style(code, tests_json, problem)
    )
    if style == "function" and has_function_test_scaffolding(code):
        return None, "function_test_scaffolding"
    return (
        Candidate(
            source=source,
            source_revision=revision,
            source_row_id=source_row_id,
            native_id=native_id,
            family_id=family_id,
            license=license_name,
            problem=problem,
            code=code,
            tests_json=tests_json,
            style=style,
            difficulty=difficulty,
            source_platform=platform,
            verification_level=verification,
            lineage=lineage,
        ),
        None,
    )


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_candidate(candidate: Candidate, cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    instruction = (
        cfg["function_instruction"]
        if candidate.style == "function"
        else cfg["competitive_instruction"]
    )
    prompt = f"{candidate.problem.strip()}\n\n{instruction.strip()}"
    response = f"```python\n{candidate.code.strip()}\n```"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    assistant_prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_segment = tokenizer.encode(
        f"<|im_start|>assistant\n{response}<|im_end|>\n",
        add_special_tokens=False,
    )
    assistant_tokens = len(assistant_segment) - len(assistant_prefix)
    return {
        "messages": messages,
        "prompt": prompt,
        "response": response,
        "text": text,
        "n_tokens": len(input_ids),
        "assistant_tokens": assistant_tokens,
    }


def load_stream(spec: dict[str, Any], seed: int) -> Iterable[dict[str, Any]]:
    import datasets

    dataset = datasets.load_dataset(
        spec["hf_id"],
        spec.get("config"),
        split=spec.get("split", "train"),
        streaming=True,
        revision=spec["revision"],
        trust_remote_code=False,
    )
    buffer_size = int(spec.get("shuffle_buffer", 0))
    if buffer_size > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return dataset


def collect_source(
    source: str,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    tokenizer: Any,
    decontaminator: Decontaminator,
    target_scale: float,
) -> tuple[list[Candidate], dict[str, Any]]:
    target = max(1, round(int(spec["target_count"]) * target_scale))
    wanted = max(target, round(target * float(spec.get("oversample", 1.4))))
    scan_limit = int(spec.get("scan_limit", target * 10))
    kept: list[Candidate] = []
    rejects: Counter[str] = Counter()
    contamination_examples: list[dict[str, Any]] = []
    seen = 0
    stream = load_stream(spec, int(cfg["seed"]) + int(spec.get("priority", 0)))
    for row_index, row in enumerate(stream):
        seen += 1
        candidate, reject_reason = adapt_row(source, spec, row, row_index)
        if candidate is None:
            rejects[reject_reason or "adapter_reject"] += 1
        else:
            match = decontaminator.match(candidate)
            if match:
                rejects["contaminated"] += 1
                if len(contamination_examples) < 20:
                    contamination_examples.append(
                        {
                            "source_row_id": candidate.source_row_id,
                            "native_id": candidate.native_id,
                            **match,
                        }
                    )
            else:
                rendered = render_candidate(candidate, cfg, tokenizer)
                if rendered["n_tokens"] > int(cfg["max_seq_len"]):
                    rejects["too_long"] += 1
                elif rendered["assistant_tokens"] <= 0:
                    rejects["empty_assistant"] += 1
                else:
                    candidate.rendered = rendered
                    kept.append(candidate)
        if len(kept) >= wanted or seen >= scan_limit:
            break
    stats = {
        "target": target,
        "wanted_with_oversample": wanted,
        "seen": seen,
        "candidate_count": len(kept),
        "rejects": dict(rejects),
        "contamination_examples": contamination_examples,
    }
    return kept, stats


def collect_source_with_retries(
    source: str,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    tokenizer: Any,
    decontaminator: Decontaminator,
    target_scale: float,
) -> tuple[list[Candidate], dict[str, Any]]:
    retries = 6
    for attempt in range(retries):
        try:
            return collect_source(
                source,
                spec,
                cfg,
                tokenizer,
                decontaminator,
                target_scale,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 >= retries:
                raise
            wait_seconds = min(120, 10 * (2**attempt))
            print(
                f"[retry] {source} attempt={attempt + 1}/{retries} "
                f"error={type(exc).__name__}: {exc}; sleep={wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def source_cache_key(
    source: str,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    target_scale: float,
) -> str:
    payload = {
        "source": source,
        "spec": spec,
        "target_scale": target_scale,
        "max_seq_len": cfg["max_seq_len"],
        "function_instruction": cfg["function_instruction"],
        "competitive_instruction": cfg["competitive_instruction"],
        "decontam": cfg["decontam"],
        "schema_version": cfg["schema_version"],
    }
    return stable_hex(json.dumps(payload, sort_keys=True, ensure_ascii=False), size=12)


def candidate_to_cache_row(candidate: Candidate) -> dict[str, Any]:
    return {
        "source": candidate.source,
        "source_revision": candidate.source_revision,
        "source_row_id": candidate.source_row_id,
        "native_id": candidate.native_id,
        "family_id": candidate.family_id,
        "license": candidate.license,
        "problem": candidate.problem,
        "code": candidate.code,
        "tests_json": candidate.tests_json,
        "style": candidate.style,
        "difficulty": candidate.difficulty,
        "source_platform": candidate.source_platform,
        "verification_level": candidate.verification_level,
        "lineage": candidate.lineage,
        "rendered": candidate.rendered,
    }


def candidate_from_cache_row(row: dict[str, Any]) -> Candidate:
    return Candidate(**row)


def write_source_cache(
    path: Path,
    pool: list[Candidate],
    stats: dict[str, Any],
    cache_key: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(
            json.dumps(
                {"_meta": {"cache_key": cache_key, "stats": stats}},
                ensure_ascii=False,
            )
            + "\n"
        )
        for candidate in pool:
            handle.write(
                json.dumps(candidate_to_cache_row(candidate), ensure_ascii=False) + "\n"
            )
    os.replace(temporary, path)


def load_source_cache(path: Path, cache_key: str) -> tuple[list[Candidate], dict[str, Any]] | None:
    if not path.is_file():
        return None
    with path.open() as handle:
        first_line = handle.readline()
        if not first_line:
            return None
        meta = json.loads(first_line).get("_meta", {})
        if meta.get("cache_key") != cache_key:
            return None
        pool = [candidate_from_cache_row(json.loads(line)) for line in handle if line.strip()]
    return pool, meta.get("stats", {})


def load_source_cache_meta(path: Path, cache_key: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as handle:
        first_line = handle.readline()
    if not first_line:
        return None
    meta = json.loads(first_line).get("_meta", {})
    if meta.get("cache_key") != cache_key:
        return None
    return meta.get("stats", {})


def iter_source_cache(path: Path, cache_key: str) -> Iterator[Candidate]:
    with path.open() as handle:
        first_line = handle.readline()
        meta = json.loads(first_line).get("_meta", {}) if first_line else {}
        if meta.get("cache_key") != cache_key:
            raise RuntimeError(f"Source cache key mismatch: {path}")
        for line in handle:
            if line.strip():
                yield candidate_from_cache_row(json.loads(line))


def deterministic_rank(candidate: Candidate, seed: int) -> str:
    return stable_hex(f"{seed}:{candidate.source}:{candidate.native_id}:{candidate.problem_hash}")


def select_after_dedup(
    pools: dict[str, list[Candidate]],
    specs: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    target_scale: float,
) -> tuple[list[Candidate], dict[str, Any]]:
    deduper = StableNearDeduper(cfg["dedup"])
    kept_by_source: dict[str, list[Candidate]] = defaultdict(list)
    removed: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    ordered_sources = sorted(
        pools,
        key=lambda source: int(specs[source].get("priority", 999)),
    )
    for source in ordered_sources:
        candidates = sorted(
            pools[source],
            key=lambda item: deterministic_rank(item, int(cfg["seed"])),
        )
        for candidate in candidates:
            accepted, duplicate_of = deduper.add(candidate)
            if accepted:
                kept_by_source[source].append(candidate)
            else:
                removed[source] += 1
                if len(examples) < 30:
                    examples.append(
                        {
                            "source_row_id": candidate.source_row_id,
                            "duplicate_of": duplicate_of or "",
                        }
                    )
    final: list[Candidate] = []
    source_selection: dict[str, Any] = {}
    for source in ordered_sources:
        target = max(1, round(int(specs[source]["target_count"]) * target_scale))
        selected = kept_by_source[source][:target]
        final.extend(selected)
        source_selection[source] = {
            "target": target,
            "after_dedup": len(kept_by_source[source]),
            "selected": len(selected),
            "gap": max(0, target - len(selected)),
            "removed_as_duplicate": removed[source],
        }
    return final, {"sources": source_selection, "examples": examples}


def select_cached_sources_to_jsonl(
    cache_info: dict[str, tuple[Path, str]],
    specs: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    target_scale: float,
    selected_path: Path,
) -> tuple[int, dict[str, Any]]:
    deduper = StableNearDeduper(cfg["dedup"])
    source_selection: dict[str, Any] = {}
    examples: list[dict[str, str]] = []
    total_selected = 0
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected_path.with_suffix(selected_path.suffix + ".tmp")
    ordered_sources = sorted(
        cache_info,
        key=lambda source: int(specs[source].get("priority", 999)),
    )
    with temporary.open("w") as output:
        for source in ordered_sources:
            target = max(1, round(int(specs[source]["target_count"]) * target_scale))
            selected = 0
            removed = 0
            cache_path, cache_key = cache_info[source]
            for candidate in iter_source_cache(cache_path, cache_key):
                accepted, duplicate_of = deduper.add(candidate)
                if not accepted:
                    removed += 1
                    if len(examples) < 30:
                        examples.append(
                            {
                                "source_row_id": candidate.source_row_id,
                                "duplicate_of": duplicate_of or "",
                            }
                        )
                    continue
                output.write(
                    json.dumps(to_output_row(candidate, cfg), ensure_ascii=False) + "\n"
                )
                selected += 1
                total_selected += 1
                if selected >= target:
                    break
            source_selection[source] = {
                "target": target,
                "selected": selected,
                "gap": max(0, target - selected),
                "removed_as_duplicate": removed,
            }
    os.replace(temporary, selected_path)
    return total_selected, {"sources": source_selection, "examples": examples}


def to_output_row(candidate: Candidate, cfg: dict[str, Any]) -> dict[str, Any]:
    assert candidate.rendered is not None
    rendered = candidate.rendered
    sample_id = stable_hex(
        f"{candidate.source_revision}:{candidate.source}:{candidate.native_id}:{candidate.problem_hash}:{stable_hex(candidate.code)}"
    )
    lineage = {
        "hf_id": cfg["sources"][candidate.source]["hf_id"],
        "revision": candidate.source_revision,
        "config": cfg["sources"][candidate.source].get("config"),
        "split": cfg["sources"][candidate.source].get("split", "train"),
        "source_row_id": candidate.source_row_id,
        "family_id": candidate.family_id,
        **candidate.lineage,
    }
    return {
        "schema_version": cfg["schema_version"],
        "sample_id": sample_id,
        "data_source": candidate.source,
        "source_revision": candidate.source_revision,
        "source_row_id": candidate.source_row_id,
        "native_id": candidate.native_id,
        "family_id": candidate.family_id,
        "problem_id": candidate.problem_hash,
        "license": candidate.license,
        "style": candidate.style,
        "mode": "direct",
        "difficulty": candidate.difficulty,
        "source_platform": candidate.source_platform,
        "verification_level": candidate.verification_level,
        "tests_json": candidate.tests_json,
        "has_testcases": bool(candidate.tests_json),
        "lineage_json": json.dumps(lineage, ensure_ascii=False, sort_keys=True),
        "messages": rendered["messages"],
        "prompt": rendered["prompt"],
        "response": rendered["response"],
        "text": rendered["text"],
        "n_tokens": rendered["n_tokens"],
        "assistant_tokens": rendered["assistant_tokens"],
    }


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def token_distribution(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    grouped: dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[str(row[group_key])] += int(row["assistant_tokens"])
    total = sum(grouped.values())
    return {
        key: {
            "assistant_tokens": value,
            "fraction": round(value / max(1, total), 6),
        }
        for key, value in sorted(grouped.items())
    }


def build_manifest(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    collect_stats: dict[str, Any],
    dedup_stats: dict[str, Any],
) -> dict[str, Any]:
    sequence_lengths = [int(row["n_tokens"]) for row in rows]
    assistant_lengths = [int(row["assistant_tokens"]) for row in rows]
    return {
        "schema_version": cfg["schema_version"],
        "seed": cfg["seed"],
        "total_final": len(rows),
        "target_total": cfg["total_samples"],
        "val_samples": cfg["val_samples"],
        "max_seq_len": cfg["max_seq_len"],
        "target_format": {
            "thinking_markers_supervised": False,
            "assistant_blocks": "exactly one fenced Python block",
            "assistant_prefix": "```python",
        },
        "source_counts": distribution(rows, "data_source"),
        "source_assistant_token_distribution": token_distribution(rows, "data_source"),
        "style_counts": distribution(rows, "style"),
        "style_assistant_token_distribution": token_distribution(rows, "style"),
        "verification_counts": distribution(rows, "verification_level"),
        "has_testcases_count": sum(bool(row["has_testcases"]) for row in rows),
        "sequence_token_stats": {
            "total": sum(sequence_lengths),
            "p50": percentile(sequence_lengths, 0.50),
            "p95": percentile(sequence_lengths, 0.95),
            "p99": percentile(sequence_lengths, 0.99),
            "max": max(sequence_lengths, default=0),
        },
        "assistant_token_stats": {
            "total": sum(assistant_lengths),
            "p50": percentile(assistant_lengths, 0.50),
            "p95": percentile(assistant_lengths, 0.95),
            "p99": percentile(assistant_lengths, 0.99),
            "max": max(assistant_lengths, default=0),
        },
        "quality_assertions": quality_assertions(rows),
        "collect_stats": collect_stats,
        "dedup_stats": dedup_stats,
        "decontam": cfg["decontam"],
        "sources": {
            source: {
                key: value
                for key, value in spec.items()
                if key not in {"shuffle_buffer", "scan_limit", "oversample"}
            }
            for source, spec in cfg["sources"].items()
            if spec.get("enable", True)
        },
    }


def quality_assertions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_direct_mode": all(row["mode"] == "direct" for row in rows),
        "all_one_python_block": all(
            len(PYTHON_FENCE_RE.findall(row["response"])) == 1 for row in rows
        ),
        "all_without_think_markers": all(
            not any(marker in row["response"] for marker in THINK_MARKERS) for row in rows
        ),
        "all_ast_parse": all(
            code_quality_score(extract_python_code(row["response"]) or "") is not None
            for row in rows
        ),
        "all_with_lineage": all(bool(row["lineage_json"]) for row in rows),
        "all_with_positive_assistant_tokens": all(row["assistant_tokens"] > 0 for row in rows),
    }


def iter_selected_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_manifest_streaming(
    selected_path: Path,
    cfg: dict[str, Any],
    collect_stats: dict[str, Any],
    dedup_stats: dict[str, Any],
) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    source_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    style_tokens: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    assistant_lengths: list[int] = []
    ranked_ids: list[tuple[int, str]] = []
    previews: list[dict[str, Any]] = []
    has_testcases_count = 0
    assertions = {
        "all_direct_mode": True,
        "all_one_python_block": True,
        "all_without_think_markers": True,
        "all_ast_parse": True,
        "all_with_lineage": True,
        "all_with_positive_assistant_tokens": True,
    }
    total = 0
    for row in iter_selected_rows(selected_path):
        total += 1
        source = str(row["data_source"])
        style = str(row["style"])
        verification = str(row["verification_level"])
        assistant_tokens = int(row["assistant_tokens"])
        source_counts[source] += 1
        style_counts[style] += 1
        verification_counts[verification] += 1
        source_tokens[source] += assistant_tokens
        style_tokens[style] += assistant_tokens
        sequence_lengths.append(int(row["n_tokens"]))
        assistant_lengths.append(assistant_tokens)
        has_testcases_count += bool(row["has_testcases"])
        ranked_ids.append((stable_u64(str(row["sample_id"])), str(row["sample_id"])))
        if len(previews) < 20:
            previews.append(
                {
                    key: row[key]
                    for key in (
                        "sample_id",
                        "data_source",
                        "style",
                        "verification_level",
                        "n_tokens",
                        "assistant_tokens",
                        "messages",
                    )
                }
            )
        assertions["all_direct_mode"] &= row["mode"] == "direct"
        assertions["all_one_python_block"] &= len(PYTHON_FENCE_RE.findall(row["response"])) == 1
        assertions["all_without_think_markers"] &= not any(
            marker in row["response"] for marker in THINK_MARKERS
        )
        assertions["all_ast_parse"] &= (
            code_quality_score(extract_python_code(row["response"]) or "") is not None
        )
        assertions["all_with_lineage"] &= bool(row["lineage_json"])
        assertions["all_with_positive_assistant_tokens"] &= assistant_tokens > 0

    val_count = min(int(cfg["val_samples"]), max(0, total // 20))
    ranked_ids.sort()
    val_ids = {sample_id for _, sample_id in ranked_ids[:val_count]}
    total_assistant_tokens = sum(assistant_lengths)

    def token_report(values: Counter[str]) -> dict[str, Any]:
        return {
            key: {
                "assistant_tokens": value,
                "fraction": round(value / max(1, total_assistant_tokens), 6),
            }
            for key, value in sorted(values.items())
        }

    manifest = {
        "schema_version": cfg["schema_version"],
        "seed": cfg["seed"],
        "total_final": total,
        "target_total": cfg["total_samples"],
        "train_samples": total - val_count,
        "val_samples": val_count,
        "max_seq_len": cfg["max_seq_len"],
        "target_format": {
            "thinking_markers_supervised": False,
            "assistant_blocks": "exactly one fenced Python block",
            "assistant_prefix": "```python",
        },
        "source_counts": dict(sorted(source_counts.items())),
        "source_assistant_token_distribution": token_report(source_tokens),
        "style_counts": dict(sorted(style_counts.items())),
        "style_assistant_token_distribution": token_report(style_tokens),
        "verification_counts": dict(sorted(verification_counts.items())),
        "has_testcases_count": has_testcases_count,
        "sequence_token_stats": {
            "total": sum(sequence_lengths),
            "p50": percentile(sequence_lengths, 0.50),
            "p95": percentile(sequence_lengths, 0.95),
            "p99": percentile(sequence_lengths, 0.99),
            "max": max(sequence_lengths, default=0),
        },
        "assistant_token_stats": {
            "total": total_assistant_tokens,
            "p50": percentile(assistant_lengths, 0.50),
            "p95": percentile(assistant_lengths, 0.95),
            "p99": percentile(assistant_lengths, 0.99),
            "max": max(assistant_lengths, default=0),
        },
        "quality_assertions": assertions,
        "collect_stats": collect_stats,
        "dedup_stats": dedup_stats,
        "decontam": cfg["decontam"],
        "sources": {
            source: {
                key: value
                for key, value in spec.items()
                if key not in {"shuffle_buffer", "scan_limit", "oversample"}
            }
            for source, spec in cfg["sources"].items()
            if spec.get("enable", True)
        },
    }
    return manifest, val_ids, previews


def write_parquet_streaming(
    selected_path: Path,
    output_dir: Path,
    val_ids: set[str],
    batch_size: int = 1000,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, Any] = {"train": None, "val": None}
    buffers: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}

    def flush(split: str) -> None:
        rows = buffers[split]
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writers[split] is None:
            writers[split] = pq.ParquetWriter(
                output_dir / f"{split}.parquet",
                table.schema,
                compression="zstd",
            )
        writers[split].write_table(table)
        rows.clear()

    try:
        for row in iter_selected_rows(selected_path):
            split = "val" if str(row["sample_id"]) in val_ids else "train"
            buffers[split].append(row)
            if len(buffers[split]) >= batch_size:
                flush(split)
        flush("train")
        flush("val")
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()


def write_outputs(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    import datasets

    output_dir = Path(cfg["out_sft_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(rows)
    val_count = min(int(cfg["val_samples"]), max(0, len(rows) // 20))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    datasets.Dataset.from_list(train_rows).to_parquet(output_dir / "train.parquet")
    if val_rows:
        datasets.Dataset.from_list(val_rows).to_parquet(output_dir / "val.parquet")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    with (output_dir / "preview.jsonl").open("w") as handle:
        for row in rows[:20]:
            preview = {
                key: row[key]
                for key in (
                    "sample_id",
                    "data_source",
                    "style",
                    "verification_level",
                    "n_tokens",
                    "assistant_tokens",
                    "messages",
                )
            }
            handle.write(json.dumps(preview, ensure_ascii=False) + "\n")


def run_self_test() -> None:
    good = """Explanation.\n```python\ndef add(a, b):\n    return a + b\n```\n```python\nassert add(1, 2) == 3\n```"""
    assert extract_python_code(good) == "def add(a, b):\n    return a + b"
    assert extract_python_code("<think>bad</think>") is None
    left = text_shingles("a b c d e f g", 3)
    assert jaccard(left, left) == 1.0
    signature = bottom_k_signature(left, 16)
    assert len(signature) == 16
    assert len(band_keys(signature, 4)) == 4
    fingerprint = simhash64("alpha beta gamma".split())
    assert hamming_distance(fingerprint, fingerprint) == 0
    assert len(simhash_band_keys(fingerprint, 4)) == 4
    print("[self-test] ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "data_prep/sft_code_data.yaml"))
    parser.add_argument("--tokenizer", default=None, help="override tokenizer_path")
    parser.add_argument("--out-dir", default=None, help="override out_sft_dir")
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    config_path = Path(args.config).resolve()
    cfg = load_data_profile(config_path, expected_schema="code_sft_v2.0")
    if args.total is not None:
        cfg["total_samples"] = args.total
    cfg["tokenizer_path"] = args.tokenizer or expand_path(cfg["tokenizer_path"])
    cfg["out_sft_dir"] = args.out_dir or expand_path(cfg["out_sft_dir"])
    configured_total = sum(
        int(spec["target_count"])
        for spec in cfg["sources"].values()
        if spec.get("enable", True)
    )
    target_scale = int(cfg["total_samples"]) / configured_total

    cache_dir = cfg.get("hf_cache_dir")
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", str(Path(cache_dir) / "datasets"))
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_path"],
        trust_remote_code=True,
        use_fast=False,
    )
    decontaminator = Decontaminator(cfg["decontam"])
    specs = {
        source: spec
        for source, spec in cfg["sources"].items()
        if spec.get("enable", True)
    }
    collect_stats: dict[str, Any] = {}
    cache_info: dict[str, tuple[Path, str]] = {}
    source_cache_dir = Path(cfg["out_sft_dir"]) / ".source_cache"
    for source, spec in sorted(specs.items(), key=lambda item: item[1].get("priority", 999)):
        print(f"[collect] {source}", flush=True)
        cache_key = source_cache_key(source, spec, cfg, target_scale)
        cache_path = source_cache_dir / f"{source}-{cache_key}.jsonl"
        stats = load_source_cache_meta(cache_path, cache_key)
        if stats is None:
            pool, stats = collect_source_with_retries(
                source,
                spec,
                cfg,
                tokenizer,
                decontaminator,
                target_scale,
            )
            write_source_cache(cache_path, pool, stats, cache_key)
            del pool
        else:
            stats = {**stats, "loaded_from_cache": str(cache_path)}
        cache_info[source] = (cache_path, cache_key)
        collect_stats[source] = stats
        print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

    selected_path = source_cache_dir / "selected-v2.jsonl"
    total_selected, dedup_stats = select_cached_sources_to_jsonl(
        cache_info,
        specs,
        cfg,
        target_scale,
        selected_path,
    )
    manifest, val_ids, previews = build_manifest_streaming(
        selected_path,
        cfg,
        collect_stats,
        dedup_stats,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("[dry-run] outputs were not written")
        return
    output_dir = Path(cfg["out_sft_dir"])
    write_parquet_streaming(selected_path, output_dir, val_ids)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    with (output_dir / "preview.jsonl").open("w") as handle:
        for row in previews:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected_path.unlink()
    print(f"[done] wrote {total_selected} rows to {cfg['out_sft_dir']}")


if __name__ == "__main__":
    main()
