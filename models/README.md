# Models

Run configs refer to `${MODEL_ROOT}/<dir>`. Each `<dir>` below has a folder here with

the files that define prompting and decoding (`config.json`, `generation_config.json`,
`tokenizer_config.json` with the chat template, `special_tokens_map.json`, `chat_template.jinja`),
where applicable a packaging/provenance json, and for the DeepSeek-family directories the tokenizer files
used in training (`tokenizer.json`, `vocab.json`, `merges.txt`, re-exported from the Hub tokenizer by the
pair builders; files identical to the Hub checkpoint are omitted).

To assemble `$MODEL_ROOT`, put the upstream weights of each row into `$MODEL_ROOT/<dir>/` and copy this
folder's files over them (the files here take precedence). `build/` holds the scripts that built these
directories. For the models trained here, the tokenizer is published with the weights.

## Directories

| `<dir>` | Used as | Weights | Notes |
|---|---|---|---|
| `qwen3_4b` | student (Math GRPO-500, Code GRPO-300, 30B pairs, teacher switch) | `Qwen/Qwen3-4B` | the three original pair directories were identical and are merged here |
| `math_grpo500_qwen3_4b` | teacher | [`wyy1112/OPD-Models/math_grpo500_qwen3_4b`](https://huggingface.co/wyy1112/OPD-Models/tree/main/math_grpo500_qwen3_4b) (trained here, Math GRPO-500) | `configs/teacher_training/math_grpo500_qwen3_4b.yaml` |
| `code_grpo300_qwen3_4b` | teacher | [`wyy1112/OPD-Models/code_grpo300_qwen3_4b`](https://huggingface.co/wyy1112/OPD-Models/tree/main/code_grpo300_qwen3_4b) (trained here, Code GRPO-300) | `configs/teacher_training/code_grpo300_qwen3_4b.yaml` |
| `qwen3_1p7b_base_sft554` | student | [`wyy1112/OPD-Models/qwen3_1p7b_base_sft554`](https://huggingface.co/wyy1112/OPD-Models/tree/main/qwen3_1p7b_base_sft554) (trained here, Qwen3-1.7B-Base + direct-code SFT, step 554) | `configs/teacher_training/qwen3_1p7b_base_sft554.yaml` |
| `qwen3_1p7b_base_chat` | SFT init of `qwen3_1p7b_base_sft554` | `Qwen/Qwen3-1.7B-Base` | minimal chat template, eos `<|im_end|>`; `build/setup_chat_template_model.py` |
| `code_grpo400_qwen3_1p7b` | teacher | [`wyy1112/OPD-Models/code_grpo400_qwen3_1p7b`](https://huggingface.co/wyy1112/OPD-Models/tree/main/code_grpo400_qwen3_1p7b) (trained here, GRPO-400 from the SFT-554 model) | `configs/teacher_training/code_grpo400_qwen3_1p7b.yaml` |
| `qwen3_30b_a3b_instruct_2507` | teacher | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `build/build_opd_30b_teacher_pairs.py` |
| `qwen3_1p7b` | student | `Qwen/Qwen3-1.7B` | same builder |
| `r1_distill_qwen_1p5b` | student (JustRL, DeepScaleR, Nemotron) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Hub config + bos/eos/pad ids from the tokenizer; `build/build_deepseek_r1_justrl_1p5b_pair.py` |
| `justrl_deepseek_1p5b` | teacher | `hbx/JustRL-DeepSeek-1.5B` | same builder |
| `deepscaler_1p5b_preview` | teacher | `agentica-org/DeepScaleR-1.5B-Preview` | weights unchanged; student-side tokenizer (`OPD_TEACHER_PACKAGE.json`) |
| `nemotron_reasoning_1p5b_v1` | teacher | `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`, revision `v1` (commit `b89048893f95246c6b5749b287f0049e6df42ee9`) | weights unchanged; shared tokenizer; `build/build_nemotron_v1_1p5b_teacher.py` |
| `r1_distill_qwen_7b` | student | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | same packaging rule as the 1.5B pair (Hub config + bos 151646 / eos 151643 / pad 151643) |
| `light_r1_7b_ds` | teacher | `qihoo360/Light-R1-7B-DS` | same packaging rule |
| `qwen3_4b_instruct_2507` | student | `Qwen/Qwen3-4B-Instruct-2507` | `warmstart_provenance.json` lists weight sha256 |
| `qwen3_4b_inst_mix` | teacher | [`Siye01/Qwen3-4B-Inst-Mix`](https://huggingface.co/Siye01/Qwen3-4B-Inst-Mix) (LLM-Fusion Mix checkpoint of Qwen3-4B-Instruct-2507, from the [LLM-Fusion collection](https://huggingface.co/collections/Siye01/llm-fusion)) | `warmstart_provenance.json` lists weight sha256; both shards match the Hub LFS sha256 |

The builders in `build/` keep the output paths of the research repository.

## Models trained for this paper

The three teachers and the 1.7B SFT student are published in one Hugging Face repository,
[`wyy1112/OPD-Models`](https://huggingface.co/wyy1112/OPD-Models), with one subfolder per model
directory. Download them straight into `$MODEL_ROOT`:

```bash
hf download wyy1112/OPD-Models --local-dir $MODEL_ROOT \
  --include "math_grpo500_qwen3_4b/*" "code_grpo300_qwen3_4b/*" "qwen3_1p7b_base_sft554/*" "code_grpo400_qwen3_1p7b/*"
```

| Subfolder | Weights | sha256 |
|---|---|---|
| `math_grpo500_qwen3_4b` | `model.safetensors` (BF16, 8.0 GB) | `0e611153504ed93f4ef9d803a3d26b7fb94c8fa7ecdbc2b2e64a7988419dfd90` |
| `code_grpo300_qwen3_4b` | `model.safetensors` (BF16, 8.0 GB) | `699a2eabeafc6a6d2b9c0008c7ea179c5c1cc9a4518761e2048fcc2149ccc39a` |
| `qwen3_1p7b_base_sft554` | `model-00001-of-00002.safetensors` (FP32) | `6671c165ef89e170de8cc412bf378127c655286631491fb1024a20cff4d96742` |
| | `model-00002-of-00002.safetensors` (FP32) | `7e7cbdd843adaa7d4414098d5951aa7d3dd3ef7520dec8b78847fd3a317b7a89` |
| `code_grpo400_qwen3_1p7b` | `model.safetensors` (BF16, 3.4 GB) | `57a968db99192d3da792c9b6a48af16ced1c97458849977cebd4fe0ba2d7f5b1` |

The SFT-554 student is stored in FP32, exactly as loaded by its OPD runs. To retrain them see [`teacher_training/`](teacher_training/).

The full-pool Math GRPO-500 run was launched from an older copy of the same student/teacher pair; its
recorded model identity (architecture, special tokens, tokenizer hashes, weight files) equals
`qwen3_4b` / `math_grpo500_qwen3_4b`.
