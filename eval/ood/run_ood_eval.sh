#!/usr/bin/env bash
# Evaluate one checkpoint on the three OOD benchmarks on a single idle 8-GPU node.
#
# Usage: run_ood_eval.sh <label> <model_path>
#
# Benchmarks are processed one at a time, each spread over all GPUs as
# tensor-parallel-1 data shards, then judged with the original graders.  Prompts for
# HumanEval+ and LiveCodeBench v6 are the G-OPD paper's code eval prompts; GPQA-Diamond
# uses the R1 multiple-choice template.  Non-thinking by default; THINK_MODE=1 switches to
# Qwen's recommended thinking settings.
set -euo pipefail
umask 077

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
LABEL="${1:?usage: run_ood_eval.sh <label> <model_path>}"
MODEL="${2:?need model_path}"

N_SAMPLES=${N_SAMPLES:-8}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
BENCHES=${BENCHES:-"gpqa_diamond humaneval_plus livecodebench_v6"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJ/results/ood_eval"}
LOG_ROOT=${LOG_ROOT:-"$PROJ/logs/ood_eval"}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_TOKENS=${MAX_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MODEL_DTYPE=${MODEL_DTYPE:-auto}
SEED=${SEED:-42}
SCORE_WORKERS=${SCORE_WORKERS:-32}
# EvalPlus can occasionally leave an idle process pool with unresolved futures. Bound
# each benchmark's judge phase so the caller can retry without regenerating completed
# shards. A zero value preserves the upstream unbounded behavior.
SCORE_TIMEOUT_SECONDS=${SCORE_TIMEOUT_SECONDS:-7200}
CONDA_ENV=${CONDA_ENV:-verl}
# THINK_MODE=1 switches to Qwen's recommended thinking settings and makes the scorer look only
# past the closing think tag.  Length has to grow with it: the same benchmarks that fit in
# 16,384 non-thinking tokens need Qwen's 38,912 for competition-style problems once the model
# reasons first.  Everything stays off by default so existing labels reproduce byte for byte.
THINK_MODE=${THINK_MODE:-0}
if [ "$THINK_MODE" = "1" ]; then
  TEMPERATURE=${TEMPERATURE_THINK:-0.6}
  TOP_P=${TOP_P_THINK:-0.95}
  TOP_K=${TOP_K:-20}
  MIN_P=${MIN_P:-0}
  MAX_TOKENS=${MAX_TOKENS_THINK:-38912}
  MAX_MODEL_LEN=${MAX_MODEL_LEN_THINK:-40960}
  # Eight engines start together here, one per GPU, and the memory each reserves for a batch of
  # this length is what the KV cache does not get.  A single card cannot hold 64 sequences of
  # 40,960 tokens in any case, so asking for that many only starves the cache.
  MAX_NUM_SEQS=${MAX_NUM_SEQS_THINK:-16}
  GEN_EXTRA=(--enable-thinking --top-k "$TOP_K" --min-p "$MIN_P")
else
  GEN_EXTRA=()
fi
# Keep thinking-mode sampling independent from whether the scorer strips a reasoning block.  This
# supports reasoning checkpoints that use the same sampling budget without tag-delimited output.
STRIP_THINK=${STRIP_THINK:-$THINK_MODE}
if [ "$STRIP_THINK" = "1" ]; then
  SCORE_EXTRA=(--strip-think)
elif [ "$STRIP_THINK" = "0" ]; then
  SCORE_EXTRA=()
else
  echo "ERROR: STRIP_THINK must be 0 or 1, got $STRIP_THINK" >&2
  exit 2
fi
# Concurrently starting engines otherwise race for the same distributed init port, so each
# shard gets its own disjoint range.
BASE_VLLM_PORT=${BASE_VLLM_PORT:-41000}
GEN_ATTEMPTS=${GEN_ATTEMPTS:-2}

if [ -z "${OOD_EVAL_ENV_READY:-}" ]; then
  set +u
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
  set -u
  export OOD_EVAL_ENV_READY=1
fi

test -f "$MODEL/config.json" || { echo "ERROR: no config.json in $MODEL" >&2; exit 1; }

read -r -a gpu_arr <<< "$GPUS"
NUM_SHARDS="${#gpu_arr[@]}"

echo "[start] $(date '+%F %T') label=$LABEL model=$MODEL n=$N_SAMPLES shards=$NUM_SHARDS"

for bench in $BENCHES; do
  GEN_DIR="$RESULT_ROOT/$LABEL/$bench"
  LOG_DIR="$LOG_ROOT/$LABEL/$bench"
  mkdir -p "$GEN_DIR" "$LOG_DIR"

  if [ -f "$GEN_DIR/summary.json" ]; then
    echo "[skip] $LABEL/$bench already scored"
    continue
  fi

  for attempt in $(seq 1 "$GEN_ATTEMPTS"); do
    missing=()
    for shard_id in "${!gpu_arr[@]}"; do
      if [ ! -f "$GEN_DIR/shard_${shard_id}.jsonl" ]; then
        missing+=("$shard_id")
      fi
    done
    if [ "${#missing[@]}" -eq 0 ]; then
      break
    fi
    echo "[gen] $(date '+%F %T') $LABEL/$bench attempt $attempt, ${#missing[@]} shard(s)"

    pids=()
    for shard_id in "${missing[@]}"; do
      gpu="${gpu_arr[$shard_id]}"
      CUDA_VISIBLE_DEVICES="$gpu" VLLM_PORT=$((BASE_VLLM_PORT + shard_id * 16)) \
        python3 "$HERE/ood_generate.py" \
        --bench "$bench" \
        --model "$MODEL" \
        --out "$GEN_DIR/shard_${shard_id}.jsonl" \
        --n "$N_SAMPLES" \
        --shard-id "$shard_id" \
        --num-shards "$NUM_SHARDS" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --max-tokens "$MAX_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --dtype "$MODEL_DTYPE" \
        --seed "$SEED" \
        "${GEN_EXTRA[@]}" \
        >>"$LOG_DIR/shard_${shard_id}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || true
    done
  done

  for shard_id in "${!gpu_arr[@]}"; do
    if [ ! -f "$GEN_DIR/shard_${shard_id}.jsonl" ]; then
      echo "ERROR: $LABEL/$bench shard $shard_id never produced output; see $LOG_DIR/shard_${shard_id}.log" >&2
      exit 1
    fi
  done

  echo "[score] $(date '+%F %T') $LABEL/$bench"
  score_timeout=()
  if [ "$SCORE_TIMEOUT_SECONDS" -gt 0 ]; then
    score_timeout=(
      timeout --signal=TERM --kill-after=60 "$SCORE_TIMEOUT_SECONDS"
    )
  fi
  "${score_timeout[@]}" python3 "$HERE/ood_score.py" \
    --bench "$bench" \
    --gen-dir "$GEN_DIR" \
    --label "$LABEL" \
    --out "$GEN_DIR/summary.json" \
    --workers "$SCORE_WORKERS" \
    "${SCORE_EXTRA[@]}" \
    >"$LOG_DIR/score.log" 2>&1 || {
      echo "ERROR: scoring failed for $LABEL/$bench; see $LOG_DIR/score.log" >&2
      exit 1
    }
  python3 -c "
import json, sys
s = json.load(open(sys.argv[1]))
n = s['n']
print(f\"  {s['benchmark']}: mean@{n}={s[f'mean@{n}']:.4f} hit@{n}={s[f'hit@{n}']:.4f} \"
      f\"avg_tokens={s['avg_output_tokens']} truncated={s['truncated_fraction']}\")
" "$GEN_DIR/summary.json"
done

echo "[done] $(date '+%F %T') $LABEL -> $RESULT_ROOT/$LABEL"
