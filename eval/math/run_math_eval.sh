#!/usr/bin/env bash
# Evaluate one G-OPD math GRPO checkpoint on the paper's four math benchmarks.
#
# Usage: run_math_eval.sh <step|checkpoint_dir> [eval_n]
# (a step number requires RUN_DIR=<OPD output dir>; the paper uses eval_n=16)
#
# Runs on a single idle 8-GPU node.  Merges the verl FSDP actor checkpoint into
# Hugging Face format if needed, then runs G-OPD/math_eval/eval_math.py on
# AIME24, AIME25, HMMT25-Feb and HMMT25-Nov concurrently (two GPUs each) with
# the paper's sampling parameters, and writes a mean@k summary.
set -euo pipefail
umask 077

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
TARGET="${1:?usage: run_math_eval.sh <step|checkpoint_dir> [eval_n]}"
EVAL_N="${2:-16}"

MATH_DATA=${MATH_DATA:-"$PROJ/data/eval/math"}
RUN_DIR=${RUN_DIR:-}
RESULT_ROOT=${RESULT_ROOT:-"$PROJ/outputs/math_eval"}
MAX_TOKENS=${MAX_TOKENS:-16384}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95}
MODEL_DTYPE=${MODEL_DTYPE:-auto}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
SEED=${SEED:-42}
VERL_ROOT=${VERL_ROOT:?VERL_ROOT must point at the patched verl checkout}
# Four vLLM engines initialize concurrently. Give each benchmark a disjoint port range so two
# tensor-parallel groups cannot race and select the same rendezvous port.
BASE_VLLM_PORT=${BASE_VLLM_PORT:-43000}
# THINK_MODE=1 adopts Qwen's thinking recommendation: temperature 0.6, top_p 0.95, top_k 20,
# min_p 0, and the 38,912-token output length they suggest for math and programming
# competition benchmarks.  Off by default so every existing number reproduces.
THINK_MODE=${THINK_MODE:-0}
if [ "$THINK_MODE" = "1" ]; then
  TEMPERATURE=${TEMPERATURE_THINK:-0.6}
  TOP_P=${TOP_P_THINK:-0.95}
  MAX_TOKENS=${MAX_TOKENS_THINK:-38912}
  # Four engines start at once on one node, and each profiles free GPU memory while the others
  # are still allocating.  At 256 sequences x 40,960 tokens the activation reserve that profile
  # asks for can leave almost nothing for the KV cache: two engines here came up with 55k tokens
  # of cache, a concurrency of 1.36x, and finished zero requests in 37 minutes.  A budget that
  # cannot hold 32 full-length sequences anyway is the honest setting, and it leaves the cache
  # room to be large.
  MAX_NUM_SEQS=${MAX_NUM_SEQS_THINK:-32}
  GPU_MEM_UTIL=${GPU_MEM_UTIL_THINK:-0.90}
  MAX_MODEL_LEN=${MAX_MODEL_LEN_THINK:-${MAX_MODEL_LEN:-40960}}
  EVAL_EXTRA=(--enable_thinking --top_k "${TOP_K:-20}" --min_p "${MIN_P:-0}")
else
  EVAL_EXTRA=()
fi
MODEL_LEN_EXTRA=()
if [ -n "$MAX_MODEL_LEN" ]; then
  MODEL_LEN_EXTRA=(--max_model_len "$MAX_MODEL_LEN")
fi

if [ -d "$TARGET" ]; then
  CHECKPOINT="$TARGET"
  # LABEL may be pre-set by a caller that evaluates several models whose directory names would
  # collide or would not say which run they belong to.
  LABEL="${LABEL:-$(basename "$TARGET")}"
else
  LABEL="${LABEL:-step_${TARGET}}"
  CHECKPOINT="$RUN_DIR/merged_hf/$LABEL"
  VERL_ROOT="$VERL_ROOT" bash "$PROJ/opd/tools/merge_checkpoint.sh" "$RUN_DIR" "$TARGET"
fi
test -f "$CHECKPOINT/config.json"

EVAL_DIR="$RESULT_ROOT/$LABEL/eval_outputs"
mkdir -p "$EVAL_DIR"

declare -A INPUTS=(
  [aime24]="$MATH_DATA/aime24/test.jsonl"
  [aime25]="$MATH_DATA/aime25/test.jsonl"
  [hmmt25_feb]="$MATH_DATA/hmmt25_feb/test.jsonl"
  [hmmt25_nov]="$MATH_DATA/hmmt25_nov/test.jsonl"
)
declare -A DEVICES=(
  [aime24]="0,1"
  [aime25]="2,3"
  [hmmt25_feb]="4,5"
  [hmmt25_nov]="6,7"
)

pids=()
bench_idx=0
for name in aime24 aime25 hmmt25_feb hmmt25_nov; do
  vllm_port=$((BASE_VLLM_PORT + bench_idx * 16))
  bench_idx=$((bench_idx + 1))
  test -f "${INPUTS[$name]}"
  # One benchmark failing used to cost all four: the script exits before summarising, and a
  # rerun regenerates everything.  Generations are complete-or-absent (eval_math.py writes its
  # output once, at the end), so an existing file is safe to keep.
  if [ -s "$EVAL_DIR/$name.jsonl" ]; then
    echo "[skip] $name already generated"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${DEVICES[$name]}" VLLM_PORT="$vllm_port" \
    python3 "$HERE/eval_math.py" \
    --input_file "${INPUTS[$name]}" \
    --model_path "$CHECKPOINT" \
    --output_file "$EVAL_DIR/$name.jsonl" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --dtype "$MODEL_DTYPE" \
    --n "$EVAL_N" \
    --begin_idx -1 --end_idx -1 --seed "$SEED" \
    "${MODEL_LEN_EXTRA[@]}" \
    "${EVAL_EXTRA[@]}" \
    >"$RESULT_ROOT/$LABEL/$name.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
  echo "ERROR: at least one benchmark failed; see $RESULT_ROOT/$LABEL/*.log" >&2
  exit 1
fi

python3 "$HERE/summarize_math_eval.py" \
  --eval-dir "$EVAL_DIR" \
  --label "$LABEL" \
  --output "$RESULT_ROOT/$LABEL/summary.json"
echo "[done] $RESULT_ROOT/$LABEL/summary.json"
