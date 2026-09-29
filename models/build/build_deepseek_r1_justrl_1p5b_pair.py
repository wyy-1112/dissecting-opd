#!/usr/bin/env python3
"""Publish a strict OPD pair for DeepSeek-R1-Distill and JustRL-DeepSeek.

The Hub checkpoints use one monolithic ``tokenizer.json`` and omit
``pad_token_id`` from ``config.json``.  The repository's fail-closed OPD
preflight additionally requires the classic split BPE payload and a standalone
special-token map.  This script derives those files from the shared tokenizer,
fills the token IDs from Transformers, and symlinks the original weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("/path/to/models")
DEFAULT_STUDENT = MODEL_ROOT / "DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_TEACHER = MODEL_ROOT / "JustRL-DeepSeek-1.5B"
DEFAULT_OUTPUT = REPO_ROOT / "data/models/opd_deepseek_r1_justrl_1p5b"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors weights in {directory}")
    return files


def _classic_special_token(value: object) -> object:
    if not isinstance(value, dict):
        return value
    content = value.get("content")
    if not isinstance(content, str) or not content:
        raise SystemExit(f"invalid special-token declaration: {value!r}")
    return {
        "content": content,
        "lstrip": bool(value.get("lstrip", False)),
        "normalized": bool(value.get("normalized", False)),
        "rstrip": bool(value.get("rstrip", False)),
        "single_word": bool(value.get("single_word", False)),
    }


def _shared_tokenizer_payload(source: Path) -> dict[str, object]:
    tokenizer_config = _load_json(source / "tokenizer_config.json")
    tokenizer_json = _load_json(source / "tokenizer.json")
    model = tokenizer_json.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise SystemExit("expected a BPE model in tokenizer.json")
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not vocab:
        raise SystemExit("tokenizer.json has no BPE vocabulary")
    if not isinstance(merges, list) or not merges:
        raise SystemExit("tokenizer.json has no BPE merges")

    special_tokens: dict[str, object] = {}
    for key in (
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "additional_special_tokens",
    ):
        value = tokenizer_config.get(key)
        if value is None:
            continue
        special_tokens[key] = (
            [_classic_special_token(item) for item in value]
            if isinstance(value, list)
            else _classic_special_token(value)
        )
    if not special_tokens:
        raise SystemExit("tokenizer_config.json declares no special tokens")

    merge_lines: list[str] = []
    for merge in merges:
        if isinstance(merge, str):
            merge_lines.append(merge)
        elif (
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(piece, str) for piece in merge)
        ):
            merge_lines.append(" ".join(merge))
        else:
            raise SystemExit(f"unsupported BPE merge entry: {merge!r}")
    return {
        "tokenizer_config": tokenizer_config,
        "tokenizer_json": tokenizer_json,
        "vocab": vocab,
        "merges": "#version: 0.2\n" + "\n".join(merge_lines) + "\n",
        "special_tokens": special_tokens,
    }


def _verify_sources(student: Path, teacher: Path) -> tuple[int, int, int]:
    from transformers import AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(str(student))
    teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher))
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise SystemExit("source vocabularies differ")
    if student_tokenizer.get_added_vocab() != teacher_tokenizer.get_added_vocab():
        raise SystemExit("source added-token vocabularies differ")
    if student_tokenizer.special_tokens_map != teacher_tokenizer.special_tokens_map:
        raise SystemExit("source special-token maps differ")
    if student_tokenizer.chat_template != teacher_tokenizer.chat_template:
        raise SystemExit("source chat templates differ")

    messages = [
        {
            "role": "user",
            "content": "Solve $1+1$ step by step and put the answer in \\boxed{}.",
        }
    ]
    rendered = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for tokenizer in (student_tokenizer, teacher_tokenizer)
    ]
    if rendered[0] != rendered[1]:
        raise SystemExit("rendered prompts differ")
    if student_tokenizer(rendered[0]).input_ids != teacher_tokenizer(rendered[1]).input_ids:
        raise SystemExit("rendered prompt token IDs differ")

    token_ids = (
        student_tokenizer.bos_token_id,
        student_tokenizer.eos_token_id,
        student_tokenizer.pad_token_id,
    )
    if any(token_id is None for token_id in token_ids):
        raise SystemExit(f"source tokenizer has an undefined special-token ID: {token_ids}")
    print("[verify] vocab, added tokens, special tokens, template, and IDs agree")
    return tuple(int(token_id) for token_id in token_ids)


def _publish(
    source: Path,
    destination: Path,
    *,
    shared: dict[str, object],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    config = _load_json(source / "config.json")
    config.update(
        {
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
    )
    _write_json(destination / "config.json", config)
    _write_json(destination / "tokenizer_config.json", shared["tokenizer_config"])
    _write_json(destination / "tokenizer.json", shared["tokenizer_json"])
    _write_json(destination / "vocab.json", shared["vocab"])
    _write_json(destination / "special_tokens_map.json", shared["special_tokens"])
    (destination / "merges.txt").write_text(
        str(shared["merges"]),
        encoding="utf-8",
    )
    if (source / "generation_config.json").is_file():
        shutil.copyfile(
            source / "generation_config.json",
            destination / "generation_config.json",
        )

    for weight in _weight_files(source):
        link = destination / weight.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(weight.resolve())
    print(f"[publish] {destination} <- {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    student = args.student.expanduser().resolve()
    teacher = args.teacher.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for source in (student, teacher):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)

    bos_token_id, eos_token_id, pad_token_id = _verify_sources(student, teacher)
    shared = _shared_tokenizer_payload(student)
    if _sha256(student / "tokenizer.json") != _sha256(teacher / "tokenizer.json"):
        raise SystemExit("source tokenizer.json payloads are not byte-identical")
    if _sha256(student / "tokenizer_config.json") != _sha256(
        teacher / "tokenizer_config.json"
    ):
        raise SystemExit("source tokenizer_config.json payloads are not byte-identical")

    _publish(
        student,
        output_root / "student_deepseek_r1_distill_qwen_1p5b",
        shared=shared,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    _publish(
        teacher,
        output_root / "teacher_justrl_deepseek_1p5b",
        shared=shared,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


if __name__ == "__main__":
    main()
