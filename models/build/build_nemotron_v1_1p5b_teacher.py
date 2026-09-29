#!/usr/bin/env python3
"""Publish the pinned Nemotron v1 weights with the shared DeepSeek tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDENT = (
    ROOT
    / "data/models/opd_deepseek_r1_justrl_1p5b"
    / "student_deepseek_r1_distill_qwen_1p5b"
)
DEFAULT_TEACHER = Path(
    "/path/to/models/"
    "Nemotron-Research-Reasoning-Qwen-1.5B-v1"
)
DEFAULT_OUTPUT = ROOT / "data/models/opd_nemotron_v1_1p5b_teacher_shared_tokenizer"
EXPECTED_COMMIT = "b89048893f95246c6b5749b287f0049e6df42ee9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-v1", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    student = args.student.expanduser().resolve()
    teacher = args.teacher_v1.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata = teacher / ".cache/huggingface/download/config.json.metadata"
    resolved_commit = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    if resolved_commit != EXPECTED_COMMIT:
        raise SystemExit(
            f"Nemotron v1 commit mismatch: {resolved_commit} != {EXPECTED_COMMIT}"
        )

    student_tokenizer = AutoTokenizer.from_pretrained(student)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise SystemExit("Nemotron v1 and student vocabularies differ")
    if student_tokenizer.get_added_vocab() != teacher_tokenizer.get_added_vocab():
        raise SystemExit("Nemotron v1 and student added vocabularies differ")
    probes = (
        "Solve $1+1$ and put the answer in \\boxed{}.",
        "Write Python code that prints UTF-8: 你好, world!",
        "A chemistry experiment uses 3.5 mol H₂O.",
    )
    for probe in probes:
        student_ids = student_tokenizer(
            probe, add_special_tokens=False
        ).input_ids
        teacher_ids = teacher_tokenizer(
            probe, add_special_tokens=False
        ).input_ids
        if student_ids != teacher_ids:
            raise SystemExit(f"token encoding mismatch for probe: {probe!r}")
    messages = [{"role": "user", "content": probes[0]}]
    student_prompt = student_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    teacher_prompt = teacher_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if student_prompt != teacher_prompt:
        raise SystemExit("Nemotron v1 and student chat templates render differently")

    student_config = json.loads((student / "config.json").read_text())
    teacher_config = json.loads((teacher / "config.json").read_text())
    structural_keys = (
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    )
    mismatches = {
        key: (student_config.get(key), teacher_config.get(key))
        for key in structural_keys
        if student_config.get(key) != teacher_config.get(key)
    }
    if mismatches:
        raise SystemExit(f"model structures differ: {mismatches}")

    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        shutil.copyfile(teacher / name, output / name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        shutil.copyfile(student / name, output / name)

    weights = sorted(teacher.glob("*.safetensors"))
    if not weights:
        raise SystemExit(f"no safetensors weights in {teacher}")
    for weight in weights:
        link = output / weight.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(weight.resolve())
    index = teacher / "model.safetensors.index.json"
    if index.is_file():
        shutil.copyfile(index, output / index.name)

    manifest = {
        "schema_version": 1,
        "teacher_repo": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "requested_revision": "v1",
        "resolved_commit": resolved_commit,
        "teacher_weights_source": str(teacher),
        "teacher_weight_sha256": {path.name: sha256(path) for path in weights},
        "prompt_tokenizer_source": str(student),
        "source_v1_tokenizer_verified_equivalent": True,
        "teacher_weights_unchanged": True,
    }
    (output / "OPD_TEACHER_PACKAGE.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
