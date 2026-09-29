#!/usr/bin/env python3
"""为 Qwen3-1.7B-Base 造一个带极简 chat_template 的模型目录（不复制权重）。

Base 无 chat_template，而 verl MultiTurnSFTDataset 需要它逐轮套模板。
Qwen3 官方 thinking 模板会对 <think> 做特殊处理，与我们预渲染冲突。
这里用极简模板：每轮渲染成 `<|im_start|>{role}\n{content}<|im_end|>\n`，
与 build_sft 的预渲染逐 token 一致 → sanity_check 通过、assistant 轮 loss 正确、
我们放进 content 的 <think>…</think> 原样作为训练目标。

产物：<base>-chat/ ，其中 tokenizer_config.json、config.json、
generation_config.json 为真实文件；权重/tokenizer.json/vocab/merges 等大文件仍软链
到原 Base。这样既不修改原始 Base，也能让 chat 模型在 <|im_end|> 处停止。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

IM_END_TOKEN_ID = 151645
END_OF_TEXT_TOKEN_ID = 151643


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/path/to/models/Qwen3-1.7B-Base")
    ap.add_argument("--out", default="/path/to/models/Qwen3-1.7B-Base-chat")
    args = ap.parse_args()

    base, out = os.path.abspath(args.base), os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    materialized_json = {"tokenizer_config.json", "config.json", "generation_config.json"}
    for fn in sorted(os.listdir(base)):
        src, dst = os.path.join(base, fn), os.path.join(out, fn)
        if fn in materialized_json:
            # Never write through an old symlink: doing so would mutate the pristine Base.
            if os.path.islink(dst) or os.path.isfile(dst):
                os.remove(dst)
            with open(src) as f:
                cfg = json.load(f)
            if fn == "tokenizer_config.json":
                cfg["chat_template"] = SIMPLE_CHAT_TEMPLATE
                cfg["eos_token"] = "<|im_end|>"
            elif fn == "config.json":
                cfg["eos_token_id"] = IM_END_TOKEN_ID
            else:
                cfg["eos_token_id"] = [IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID]
                cfg["pad_token_id"] = END_OF_TEXT_TOKEN_ID
            with open(dst, "w") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[real] {fn} (chat/EOS override)")
        else:
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            print(f"[link] {fn} -> {src}")

    # 校验模板渲染与我们的约定一致
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    assert tok.eos_token == "<|im_end|>", tok.eos_token
    assert tok.eos_token_id == IM_END_TOKEN_ID, tok.eos_token_id
    msgs = [{"role": "user", "content": "PROB"}, {"role": "assistant", "content": "<think>\nR\n</think>\n\n```python\nprint(1)\n```"}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    expected = "<|im_start|>user\nPROB<|im_end|>\n<|im_start|>assistant\n<think>\nR\n</think>\n\n```python\nprint(1)\n```<|im_end|>\n"
    assert rendered == expected, f"template mismatch:\n{rendered!r}\nvs\n{expected!r}"
    print("[ok] chat_template renders exactly as build_sft; model dir ready:", out)


if __name__ == "__main__":
    main()
