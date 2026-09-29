# Environment

All OPD training, analysis and evaluation ran in one conda environment (`verl`) on 8×H20 nodes
(driver 535.247.01). `requirements-verl-env.txt` is its full `pip freeze`.

| Package | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.11.0+cu130 (CUDA 13.0) |
| vLLM | 0.20.3.dev0+gbc150f502 (built from vLLM commit `bc150f502`, 2026-08-13; wheel sha256 `644094df…e859981`) |
| flash-attn | 2.8.3 |
| transformers | 5.15.0 |
| ray | 2.57.0 |
| hydra-core / omegaconf | 1.3.5 / 2.3.1 |
| tensordict | 0.10.0 |
| math-verify | 0.9.0 (with latex2sympy2_extended 1.11.0) |
| numpy / pandas / pyarrow / datasets | 1.26.4 / 3.0.5 / 25.0.1 / 5.0.1 |

The two wheels installed from local files (`vllm`, `flash_attn`) are listed in the freeze file with
their sha256. All nodes of a Ray cluster must have the same package versions (a numpy 1.x / 2.x mix
breaks object transfer between workers). `matplotlib` is only needed for the optional plot in
`data_prep/selection/update_representativeness.py`.

## External code

| Repository | Version | Used for |
|---|---|---|
| verl (`verl-project/verl`) | `7aed6b230776f963fa09509c10d9c3a767d1102c` + `opd/verl_patch` | OPD / GRPO training |
| G-OPD (`RUCBM/G-OPD`) | `37371a4c31ad7947746200d234161769191f4748` + `eval/math/g-opd-37371a4-eval_math.patch` | math eval harness (`eval/math/eval_math.py` is the patched file), evalplus and LiveCodeBench graders (`GOPD_ROOT`, default `external/G-OPD`) |
| LLM-Fusion (`Di-viner/LLM-Fusion`) | `e314dd28cea056e617490065fc08c4aed90204f3` | tool implementations for the Inst-Mix IF/Agent runs (`LLM_FUSION_ROOT`) |

The G-OPD LiveCodeBench loader expects `code_eval/coding/LiveCodeBench/code_generation_lite` (the
`livecodebench/code_generation_lite` dataset, v6 release window, 175 problems).

## Environment variables

| Variable | Meaning |
|---|---|
| `VERL_ROOT` | patched verl checkout |
| `MODEL_ROOT` | directory holding the model directories listed in `models/` |
| `OUTPUT_ROOT` | where OPD run directories are written (default `outputs/`) |
| `POOL_ROOT` | prompt pools not shipped in `data/pools` (Eurus code train pool) |
| `LLM_FUSION_ROOT` | LLM-Fusion checkout (Inst-Mix Agent48 / Mix5 runs only) |
| `GOPD_ROOT` | G-OPD checkout for the code graders |
| `RAY_ADDRESS` | running Ray cluster (default `auto`) |
| `LLM_FUSION_EXTENSION_ROOT` | LLM-Fusion checkout, exported before `opd/start_ray.sh` (Inst-Mix Agent48 / Mix5 runs only) |

Start Ray with `opd/start_ray.sh` on every node: Ray workers inherit the environment of `ray start`, not
the driver's, and need the patched verl on `PYTHONPATH`, the vLLM prompt-logprob patch
(`OPD_PATCH_VLLM_PROMPT_LOGPROBS=1`, `OPD_VLLM_PATCH_PATH`) and the rollout / reward timeouts.
Model weights are read by several vLLM workers at once; if `MODEL_ROOT` is on a network file system and
loading stalls, copy the weights to local disk.
