#!/usr/bin/env python3
"""Assemble the Qwen3-4B OPD pair that scripts/opd/check_opd_models.py accepts.

The two source checkpoints are token-ID identical but not file identical, so the
fail-closed pair check rejects them:

* stock ``Qwen3-4B`` declares its pad token only in ``tokenizer_config.json``,
  never in ``config.json``;
* verl's checkpoint merger drops ``vocab.json``/``merges.txt``, writes
  ``bos_token_id: null``, and re-serializes ``tokenizer.json`` with a newer
  ``tokenizers`` release, which rewrites the ``trim_offsets`` and decoder
  ``ByteLevel`` flags.

None of that changes tokenization -- the vocabularies, added tokens, special
token ids, and the token ids of a rendered non-thinking prompt are all equal --
so this script republishes both checkpoints with one shared set of tokenizer
files and the missing config fields filled in.  Weights are symlinked, so each
directory costs a few megabytes; a launcher that stages models onto node-local
disk must dereference (``cp -aL``).

Run with --verify to re-check that the sources still tokenize identically.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = Path("/path/to/models/Qwen3-4B")
TEACHER_MODEL = REPO_ROOT / "results/grpo/qwen3_4b_gopd_math/merged_hf/step_500"
OUTPUT_ROOT = REPO_ROOT / "data/models/opd_qwen3_4b_t_grpo500"

# Byte-identical in both destinations: the pair check hashes these to prove the
# student and the teacher agree on token ids.
SHARED_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
PAD_TOKEN_ID = 151643  # <|endoftext|>, the pad token both tokenizer_configs declare
BOS_TOKEN_ID = 151643


def _weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors weights in {directory}")
    return files


def _special_tokens_map() -> str:
    """The ``special_tokens_map.json`` neither source checkpoint ships.

    Qwen3 keeps its special tokens in ``tokenizer_config.json``, and current
    transformers no longer emits the standalone file on ``save_pretrained``, but
    the pair check requires it.  Rebuild it in the classic HF shape straight from
    the base ``tokenizer_config.json`` -- the metadata that is actually shipped --
    and give both destinations the same bytes.
    """
    tokenizer_config = json.loads(
        (BASE_MODEL / "tokenizer_config.json").read_text(encoding="utf-8")
    )
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
        payload[key] = (
            [encode(item) for item in value] if isinstance(value, list) else encode(value)
        )
    if not payload:
        raise SystemExit("base tokenizer_config.json declares no special tokens")
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _publish(
    source: Path,
    destination: Path,
    config_overrides: dict[str, object],
    special_tokens_map: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    for key, value in config_overrides.items():
        config[key] = value
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for name in SHARED_TOKENIZER_FILES:
        shutil.copyfile(BASE_MODEL / name, destination / name)
    (destination / "special_tokens_map.json").write_text(
        special_tokens_map, encoding="utf-8"
    )
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
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    teacher = AutoTokenizer.from_pretrained(str(TEACHER_MODEL))
    if student.get_vocab() != teacher.get_vocab():
        raise SystemExit("source vocabularies differ; the pair is not token-ID safe")
    if student.get_added_vocab() != teacher.get_added_vocab():
        raise SystemExit("source added tokens differ; the pair is not token-ID safe")
    if student.special_tokens_map != teacher.special_tokens_map:
        raise SystemExit("source special tokens differ; the pair is not token-ID safe")

    messages = [{"role": "user", "content": "What is $1+1$? Answer in \\boxed{}."}]
    rendered = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for tokenizer in (student, teacher)
    ]
    if rendered[0] != rendered[1]:
        raise SystemExit("non-thinking chat templates disagree")
    if student(rendered[0]).input_ids != teacher(rendered[1]).input_ids:
        raise SystemExit("token ids disagree for an identical prompt")
    print("[verify] vocab, special tokens, non-thinking template, and token ids agree")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Re-check source tokenizers.")
    arguments = parser.parse_args()

    if arguments.verify:
        _verify_sources()

    special_tokens_map = _special_tokens_map()
    _publish(
        BASE_MODEL,
        OUTPUT_ROOT / "student_qwen3_4b",
        {"pad_token_id": PAD_TOKEN_ID},
        special_tokens_map,
    )
    _publish(
        TEACHER_MODEL,
        OUTPUT_ROOT / "teacher_grpo_step500",
        {"bos_token_id": BOS_TOKEN_ID, "pad_token_id": PAD_TOKEN_ID},
        special_tokens_map,
    )


if __name__ == "__main__":
    main()
