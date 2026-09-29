#!/usr/bin/env python3
"""Publish the Qwen3-30B-A3B teacher and its 4B / 1.7B students for heterogeneous OPD.

Same job as build_qwen3_4b_t_grpo500_pair.py, one size class up: republish stock
checkpoints so scripts/opd/check_opd_models.py --pair heterogeneous accepts them.  Stock
Qwen3 checkpoints are rejected for two reasons that have nothing to do with tokenization:
``config.json`` declares no ``pad_token_id`` (Qwen3 keeps the pad token in
``tokenizer_config.json``), and no checkpoint ships the standalone
``special_tokens_map.json`` that current transformers stopped emitting.

Unlike the same-size script, each destination keeps **its own** tokenizer files rather
than one shared copy.  Qwen3-30B-A3B-Instruct-2507 is a non-thinking release: its chat
template has no ``<think>`` block and no ``enable_thinking`` switch, so overwriting it
with the dense Qwen3 hybrid template would be a lie about what the teacher was trained
on -- and pointless, because the teacher is fed token IDs and never applies a template.
The heterogeneous pair check is built for exactly this: it proves the shared token
payload by encoding probe strings instead of by hashing the files.

``special_tokens_map.json`` is the one file written identically everywhere.  Its content
is genuinely identical across Qwen3 releases; giving all destinations the same bytes lets
the pair check keep enforcing special-token equality, which the teacher does depend on
because it reads the student's IDs literally.

Weights are symlinked, so each directory costs a few megabytes; a launcher that stages
models onto node-local disk must dereference (``cp -aL``).

Run with --verify to prove the sources encode identically before publishing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("/path/to/models")
OUTPUT_ROOT = REPO_ROOT / "data/models/opd_qwen3_30b_teacher"

TEACHER_SOURCE = MODEL_ROOT / "Qwen3-30B-A3B-Instruct-2507"
STUDENT_SOURCES = {
    "student_qwen3_4b": MODEL_ROOT / "Qwen3-4B",
    "student_qwen3_1p7b": MODEL_ROOT / "Qwen3-1.7B",
}
# The special tokens are shared; take the canonical bytes from one dense checkpoint.
SPECIAL_TOKEN_SOURCE = MODEL_ROOT / "Qwen3-4B"

OWN_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
PAD_TOKEN_ID = 151643  # <|endoftext|>, the pad token every Qwen3 tokenizer_config declares


def _weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors weights in {directory}")
    return files


def _special_tokens_map(source: Path) -> str:
    """Rebuild ``special_tokens_map.json`` in the classic HF shape from what is shipped."""
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    attributes = {
        str(entry.get("content")): entry
        for entry in (tokenizer_config.get("added_tokens_decoder") or {}).values()
        if isinstance(entry, dict)
    }

    def encode(token: object) -> object:
        entry = attributes.get(str(token))
        if entry is None:
            return token
        return {
            "content": str(token),
            "lstrip": bool(entry.get("lstrip", False)),
            "normalized": bool(entry.get("normalized", False)),
            "rstrip": bool(entry.get("rstrip", False)),
            "single_word": bool(entry.get("single_word", False)),
        }

    payload: dict[str, object] = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token", "additional_special_tokens"):
        value = tokenizer_config.get(key)
        if value is None:
            continue
        payload[key] = [encode(item) for item in value] if isinstance(value, list) else encode(value)
    if not payload:
        raise SystemExit(f"{source}/tokenizer_config.json declares no special tokens")
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _publish(source: Path, destination: Path, special_tokens_map: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["pad_token_id"] = PAD_TOKEN_ID
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for name in OWN_TOKENIZER_FILES:
        shutil.copyfile(source / name, destination / name)
    (destination / "special_tokens_map.json").write_text(special_tokens_map, encoding="utf-8")
    for name in ("generation_config.json", "model.safetensors.index.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)

    for weight in _weight_files(source):
        link = destination / weight.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(weight.resolve())

    print(f"[publish] {destination} <- {source}")


def _verify_sources() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from opd_code.models import compare_token_encodings

    report = compare_token_encodings(
        SPECIAL_TOKEN_SOURCE,
        {"teacher": TEACHER_SOURCE, **{name: path for name, path in STUDENT_SOURCES.items()}},
    )
    if not report.compatible:
        raise SystemExit("sources do not encode identically: " + "; ".join(report.errors))
    print(f"[verify] token encodings agree across {len(report.matches)} checkpoints")

    from transformers import AutoTokenizer

    messages = [{"role": "user", "content": "What is $1+1$? Answer in \\boxed{}."}]
    teacher = AutoTokenizer.from_pretrained(str(TEACHER_SOURCE))
    student = AutoTokenizer.from_pretrained(str(SPECIAL_TOKEN_SOURCE))
    teacher_prompt = teacher.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    student_prompt = student.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if teacher_prompt == student_prompt:
        raise SystemExit(
            "teacher and student templates now agree; the heterogeneous pair mode assumes "
            "they differ, so re-check whether a plain pair check would do"
        )
    extra = student_prompt[len(teacher_prompt) :] if student_prompt.startswith(teacher_prompt) else None
    print(f"[verify] teacher template ends {teacher_prompt[-40:]!r}")
    print(f"[verify] student non-thinking template adds {extra!r}")
    if extra is None:
        print("[warn] the templates differ by more than a suffix; inspect before training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Check the sources, do not publish.")
    arguments = parser.parse_args()

    if arguments.verify:
        _verify_sources()
        return

    _verify_sources()
    special_tokens_map = _special_tokens_map(SPECIAL_TOKEN_SOURCE)
    _publish(TEACHER_SOURCE, OUTPUT_ROOT / "teacher_qwen3_30b_a3b_instruct_2507", special_tokens_map)
    for name, source in STUDENT_SOURCES.items():
        _publish(source, OUTPUT_ROOT / name, special_tokens_map)


if __name__ == "__main__":
    raise SystemExit(main())
