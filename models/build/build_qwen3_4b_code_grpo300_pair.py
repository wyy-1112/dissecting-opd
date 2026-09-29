#!/usr/bin/env python3
"""Publish the Qwen3-4B non-thinking student / Code GRPO-300 OPD pair."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDENT = (
    ROOT / "data/models/opd_qwen3_4b_t_grpo500/student_qwen3_4b"
)
DEFAULT_TEACHER = ROOT / "results/grpo/qwen3_4b_eurus_code/merged_hf/step_300"
DEFAULT_OUTPUT = ROOT / "data/models/opd_qwen3_4b_code_grpo300"
SHARED_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


def _weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors weights in {directory}")
    return files


def _verify_sources(student: Path, teacher: Path) -> None:
    from transformers import AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(str(student))
    teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher))
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise SystemExit("student and Code GRPO teacher vocabularies differ")
    if student_tokenizer.get_added_vocab() != teacher_tokenizer.get_added_vocab():
        raise SystemExit("student and Code GRPO teacher added tokens differ")

    messages = [{"role": "user", "content": "Write a Python program that prints 42."}]
    rendered = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for tokenizer in (student_tokenizer, teacher_tokenizer)
    ]
    if rendered[0] != rendered[1]:
        raise SystemExit("non-thinking chat templates render different prompts")
    if student_tokenizer(rendered[0]).input_ids != teacher_tokenizer(rendered[1]).input_ids:
        raise SystemExit("non-thinking rendered prompt token IDs differ")
    print("[verify] Code student and GRPO teacher token IDs agree")


def _publish(
    source: Path,
    destination: Path,
    *,
    tokenizer_source: Path,
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
    (destination / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in SHARED_TOKENIZER_FILES:
        shutil.copyfile(tokenizer_source / name, destination / name)
    if (source / "generation_config.json").is_file():
        shutil.copyfile(
            source / "generation_config.json",
            destination / "generation_config.json",
        )
    if (source / "model.safetensors.index.json").is_file():
        shutil.copyfile(
            source / "model.safetensors.index.json",
            destination / "model.safetensors.index.json",
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
        if not (source / "config.json").is_file():
            raise FileNotFoundError(source / "config.json")
    for name in SHARED_TOKENIZER_FILES:
        if not (student / name).is_file():
            raise FileNotFoundError(student / name)

    _verify_sources(student, teacher)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(student))
    student_config = _load_json(student / "config.json")
    ids = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else student_config.get("bos_token_id"),
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else student_config.get("eos_token_id"),
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else student_config.get("pad_token_id"),
    )
    if any(token_id is None for token_id in ids):
        raise SystemExit(f"student tokenizer has undefined special-token IDs: {ids}")
    bos_token_id, eos_token_id, pad_token_id = (int(token_id) for token_id in ids)

    _publish(
        student,
        output_root / "student_qwen3_4b_nonthinking",
        tokenizer_source=student,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    _publish(
        teacher,
        output_root / "teacher_qwen3_4b_code_grpo_step300",
        tokenizer_source=student,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


if __name__ == "__main__":
    main()
