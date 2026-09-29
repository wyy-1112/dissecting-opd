# Evaluation

## Math — `math/`

```bash
VERL_ROOT=... bash eval/math/run_math_eval.sh <merged_hf_checkpoint_dir> 16
# or a step number of an OPD run (merges the FSDP checkpoint first):
RUN_DIR=$OUTPUT_ROOT/<group>/<pair>/<run> VERL_ROOT=... bash eval/math/run_math_eval.sh 15 16
```

One 8-GPU node, four benchmarks in parallel (2 GPUs each): AIME24, AIME25, HMMT25-Feb, HMMT25-Nov
(`data/eval/math`), `n` samples per problem, T = 1.0, top-p = 1.0, 16,384 tokens, seed 42, non-thinking
(`THINK_MODE=1` switches to T = 0.6, top-p = 0.95, top-k = 20, 38,912 tokens).

`eval_math.py` is G-OPD's harness (`RUCBM/G-OPD@37371a4`, `math_eval/eval_math.py`) with the patch in
`eval/math/g-opd-37371a4-eval_math.patch` (only adds sampling/dtype/memory options; defaults unchanged).
Scoring: the last `\boxed{...}` of each response is compared with the reference via math-verify 0.9.0
(`parse` + `verify`); a response without a box is wrong. Per-problem records (`eval_outputs/<bench>.jsonl`)
keep all responses, extracted answers and correctness.

`summarize_math_eval.py`: mean@n per benchmark (mean over all problem × sample indicators), and the
Math score = equal-weight mean of the four benchmark values (`average_mean_at_k`).

## OOD — `ood/`

```bash
GOPD_ROOT=... bash eval/ood/run_ood_eval.sh <label> <merged_hf_checkpoint_dir>
```

GPQA-Diamond (198, `data/eval/gpqa`), HumanEval+ (164) and LiveCodeBench v6 (175 problems of the v6 release
window) with 8 samples, T = 1.0, top-p = 1.0, 16,384 tokens, prompts rendered with `enable_thinking=False`.
GPQA uses the R1 multiple-choice template and the first `Answer: X` line; HumanEval+ and LCB use G-OPD's code
prompts and the evalplus / LiveCodeBench graders from the G-OPD checkout. `per_task_passes.json` keeps every
sample's pass/fail; mean@8 is the mean over tasks of the per-task pass rate. `summarize_ood_eval.py`
collects the labels into one table.

Avg₄ = (Math + GPQA-D + HumanEval+ + LCB v6) / 4.
