# Training the teachers and the 1.7B SFT student

The three teachers trained for the paper and the Qwen3-1.7B-Base SFT-554 student are published on
[Hugging Face](https://huggingface.co/wyy1112/OPD-Models); this folder retrains them. Each model has a
config in `configs/teacher_training/`, launched like the OPD runs:

```bash
python opd/run.py configs/teacher_training/<model>.yaml            # original layout
python opd/run.py configs/teacher_training/<model>.yaml --gpus 8   # one 8-GPU node
```

| Config | Model | Init | Data | Recipe |
|---|---|---|---|---|
| `math_grpo500_qwen3_4b` | Math GRPO-500 | `qwen3_4b` | DeepMath-103K difficulty ≥ 6 (`data/pools/deepmath_l6.parquet`, 57,046) | GRPO, 500 steps × 128 prompts × 8, lr 1e-6, response 16,384 |
| `code_grpo300_qwen3_4b` | Code GRPO-300 | `qwen3_4b` | Eurus-2 code, stdio tests (`$POOL_ROOT/eurus_code.parquet`, 22,618) | GRPO, 300 × 128 × 8, lr 1e-6, response 8,192, all-tests-pass reward |
| `qwen3_1p7b_base_sft554` | SFT-554 student | `qwen3_1p7b_base_chat` | direct-code SFT data (70,951 + 512 val) | SFT, 1 epoch = 554 steps, batch 128, lr 3e-6 cosine |
| `code_grpo400_qwen3_1p7b` | Code GRPO-400 | `qwen3_1p7b_base_sft554` | Open-R1 verifiable Python (`data/pools/openr1_code.parquet`, 6,519) | GRPO, 400 × 128 × 8, lr 5e-7, KL 0.001, T = 0.8, fraction-of-tests reward |

The Qwen3-4B teachers use `enable_thinking=False`, token-level rollout importance sampling (threshold
5.0) and no KL loss. Code GRPO-400 predates that recipe and keeps its own settings.

## Data

- DeepMath, Open-R1 and the validation sets are shipped.
- Eurus code pool: `python data_prep/build_eurus_code_pool.py` (downloads `PRIME-RL/Eurus-2-RL-Data@9776b13`,
  writes `data/pools/eurus_code.parquet`; rebuilt row by row identical to the training file).
- SFT data: `python data_prep/build_sft_code_data.py` (sources and decontamination targets pinned in
  `data_prep/sft_code_data.yaml`, writes `$OUTPUT_ROOT/sft_code_data`). The build is deterministic
  (identical across machines); against the data used for the paper, the validation split is identical
  and 2 of the 71,463 samples differ, both from the Magicoder stream.

## Files

- `train_grpo.sh` — GRPO driver of the two Qwen3-4B teachers.
- `train_grpo_code400.sh` — GRPO driver of Code GRPO-400.
- `train_sft.sh` — SFT driver (torchrun, `verl.trainer.sft_trainer`).
- `reference/` — the verl configuration printed by each original run. The configs above reproduce them key
  by key (apart from paths and keys added to verl later, which are null/disabled), with one exception:
  Code GRPO-400 calls `compute_score_compact` instead of `compute_score`. Both return the same score; the
  compact variant drops the per-test lists that the remote reward manager cannot batch.
  `openr1_code_pool.yaml` records how the Open-R1 pool was built.

`models/qwen3_1p7b_base_chat` is Qwen3-1.7B-Base with a minimal chat template
(`<|im_start|>{role}\n{content}<|im_end|>\n`, eos `<|im_end|>`), built by
`models/build/setup_chat_template_model.py`.
