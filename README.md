<div align="center">

# 🔬 Dissecting On-Policy Distillation

<a href="https://huggingface.co/wyy1112/OPD-Models"><img src="https://img.shields.io/badge/🤗_HuggingFace-Models-ffbd45" alt="HuggingFace"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>

</div>

---

## ✨ Highlights

- 🧩 **One entry point** — `opd/run.py <config>` launches any of the 114 runs, each matching the original verl config key by key.
- 📦 **Exact data** — every run ships its prompt-ID schedule (order, selection seed, training seed), rebuilt row by row.
- 🔍 **Mechanism analysis** — parameter displacement and functional change on frozen probes.
- 🎯 **Prompt selection** — uniform, stratified, semantic, hard, shortest and cost-aware D-optimal.

## 🚀 Quick start

```bash
# 1. verl + patch
git clone https://github.com/verl-project/verl && cd verl
git checkout 7aed6b2 && git apply ../dissecting-opd/opd/verl_patch/verl-7aed6b2-opd.patch
export VERL_ROOT=$PWD && cd ../dissecting-opd

# 2. models (see models/README.md)
export MODEL_ROOT=/path/to/models OUTPUT_ROOT=/path/to/outputs
hf download wyy1112/OPD-Models --local-dir $MODEL_ROOT

# 3. Ray
bash opd/start_ray.sh --head --num-gpus=8

# 4. train and merge
python opd/run.py configs/support_size/justrl_1p5b_to_r1_distill_1p5b/deepmath_m1_s100.yaml
python opd/run.py <config> --gpus 8     # single 8-GPU node
bash opd/tools/merge_checkpoint.sh $OUTPUT_ROOT/<experiment>/<pair>/<run> <step>
```

Environment details: [`ENVIRONMENT.md`](ENVIRONMENT.md).

## 📁 Structure

```text
opd/          training entry point, verl driver, rewards, verl patch
configs/      114 OPD run configs (<experiment>/<teacher>_to_<student>/<run>.yaml) + teacher_training/
data/         training schedules, prompt pools, probes, validation and test sets
data_prep/    data construction and prompt-selection methods
analysis/     parameter and functional geometry, probe construction
eval/         math and OOD evaluation
models/       model sources; teacher_training/ retrains the teachers and the 1.7B SFT student
```

> 💡 Run names encode the setting: `deepmath_m48_s15` = DeepMath prompts, **M = 48** distinct prompts, **15** steps.

## 🧪 Experiments

Full list with M, steps, batch and seeds: [`configs/INDEX.md`](configs/INDEX.md).

### 📈 Support size

| Teacher → Student | Prompt sources (M) |
|---|---|
| Math GRPO-500 → Qwen3-4B | DeepMath: 1 → full |
| Code GRPO-300 → Qwen3-4B | Eurus code: 1 → full |
| Code GRPO-400 → Qwen3-1.7B (SFT) | Open-R1 code: 1 → full |
| JustRL-1.5B → R1-Distill-1.5B | DeepMath, DAPO: 1 → full |
| DeepScaleR-1.5B → R1-Distill-1.5B | DeepMath, DAPO: 8 |
| Nemotron-1.5B → R1-Distill-1.5B | math / code / science / mix: 48 |
| Qwen3-4B-Inst-Mix → Qwen3-4B-Instruct | math / code / science / IF / agent / mix: 48 |
| Light-R1-7B → R1-Distill-7B | DeepMath, DAPO: 8 · Light stage 2: full |
| Qwen3-30B-A3B-Instruct → Qwen3-4B | DeepMath, DAPO: 1 → full · GSM8K · code · science |
| Qwen3-30B-A3B-Instruct → Qwen3-1.7B | DeepMath, DAPO: 1 → full |

### 🔄 Teacher switch

{Math GRPO-500, Code GRPO-300} × {DeepMath 48, Eurus code 48} on Qwen3-4B, 5 seed blocks (20 runs).

### 🎯 Prompt selection

| Teacher → Student | Pool | C → M | Steps |
|---|---|---|---|
| JustRL-1.5B → R1-Distill-1.5B | DeepMath | 384 → 8 | 65 |
| Code GRPO-300 → Qwen3-4B | Eurus code | 384 → 48 | 100 |
| Qwen3-30B-A3B-Instruct → Qwen3-4B | DeepMath | 2304 → 48 | 25 |

Methods: [`data_prep/selection/`](data_prep/selection/).

## 🎓 Teacher training

The teachers trained for the paper (Math GRPO-500, Code GRPO-300, Code GRPO-400) and the Qwen3-1.7B-Base SFT-554 student are on [Hugging Face](https://huggingface.co/wyy1112/OPD-Models); to retrain them:

```bash
python data_prep/build_eurus_code_pool.py      # Code GRPO-300 data
python data_prep/build_sft_code_data.py        # SFT data

python opd/run.py configs/teacher_training/math_grpo500_qwen3_4b.yaml --gpus 8
python opd/run.py configs/teacher_training/code_grpo300_qwen3_4b.yaml --gpus 8
python opd/run.py configs/teacher_training/qwen3_1p7b_base_sft554.yaml --gpus 8
python opd/run.py configs/teacher_training/code_grpo400_qwen3_1p7b.yaml --gpus 8
```

Recipes and data: [`models/teacher_training/`](models/teacher_training/).

## 🔍 Mechanism analysis

| | Metrics | Code |
|---|---|---|
| ⚙️ **Parameter** | cosine to the teacher's task vector · remaining distance | [`analysis/parameter/`](analysis/parameter/) |
| 🧠 **Functional** | direction similarity · forward KL on frozen probes | [`analysis/functional/`](analysis/functional/) |
| 🧷 **Probes** | fixed prefix cohorts, construction and sampling | [`analysis/probes/`](analysis/probes/) |

Definitions: [`analysis/README.md`](analysis/README.md).

## 📊 Evaluation

| Suite | Benchmarks | Samples |
|---|---|---|
| 🧮 Math | AIME24 · AIME25 · HMMT25 Feb · HMMT25 Nov | 16 |
| 🌐 OOD | GPQA-Diamond · HumanEval+ · LiveCodeBench v6 | 8 |

Scripts and sampling settings: [`eval/`](eval/).

## 📜 License

Code in this repository, including the verl patch, is released under the [Apache License 2.0](LICENSE).
Files under `data/` are derived from public datasets (sources in [`data/README.md`](data/README.md)) and remain under the licenses of those datasets; model weights follow the licenses of their base models.
