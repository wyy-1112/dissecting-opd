#!/usr/bin/env bash
# Direct-code SFT of Qwen3-1.7B-Base (produces the SFT-554 student).
#
# MODEL_PATH is Qwen3-1.7B-Base with the minimal chat template of
# models/qwen3_1p7b_base_chat; DATA_DIR holds train.parquet / val.parquet written by
# data_prep/build_sft_code_data.py. The overrides are those of the original run
# (one epoch at batch 128 = 554 steps, checkpoints every quarter epoch).
set -euo pipefail
umask 077

: "${VERL_ROOT:?VERL_ROOT must point at the patched verl checkout}"
test -f "$VERL_ROOT/verl/__init__.py"
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL_PATH:?}"
: "${DATA_DIR:?}"
: "${OUTPUT_DIR:?}"
WANDB_PROJECT=${WANDB_PROJECT:-opd-sft}
WANDB_EXPERIMENT=${WANDB_EXPERIMENT:-$(basename "$OUTPUT_DIR")}
LOGGER=${LOGGER:-'["console"]'}

LR=${LR:-3e-6}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
TBS=${TBS:-128}
MAX_LEN=${MAX_LEN:-4096}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-8192}
SP=${SP:-1}
EPOCHS=${EPOCHS:-1}
SEED=${SEED:-1}
RESUME_MODE=${RESUME_MODE:-auto}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

test -f "$DATA_DIR/train.parquet"
test -f "$DATA_DIR/val.parquet"
mkdir -p "$OUTPUT_DIR"

N=$(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$DATA_DIR/train.parquet').metadata.num_rows)")
STEPS=$(( N / TBS * EPOCHS ))
SAVE_FREQ=${SAVE_FREQ:-$(( STEPS / 4 ))}
[ "$SAVE_FREQ" -lt 1 ] && SAVE_FREQ=1

python3 - "$MODEL_PATH" <<'PY'
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
if tok.eos_token != "<|im_end|>" or tok.eos_token_id != 151645:
    raise SystemExit(f"{sys.argv[1]}: eos must be <|im_end|>/151645 (see models/qwen3_1p7b_base_chat)")
PY
echo "[sft] N=$N TBS=$TBS EPOCHS=$EPOCHS -> STEPS=$STEPS SAVE_FREQ=$SAVE_FREQ"

overrides=(
  data.train_files="$DATA_DIR/train.parquet"
  data.val_files="$DATA_DIR/val.parquet"
  data.messages_key=messages
  data.train_batch_size="$TBS"
  data.max_length="$MAX_LEN"
  data.truncation=error
  data.pad_mode=no_padding
  data.use_dynamic_bsz=True
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN"
  data.ignore_input_ids_mismatch=False
  optim.lr="$LR"
  optim.lr_scheduler_type="$LR_SCHEDULER"
  optim.lr_warmup_steps_ratio="$WARMUP_RATIO"
  engine=fsdp
  engine.ulysses_sequence_parallel_size="$SP"
  model.path="$MODEL_PATH"
  model.use_remove_padding=true
  checkpoint.save_contents='["model","optimizer","hf_model","extra"]'
  trainer.default_local_dir="$OUTPUT_DIR"
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$WANDB_EXPERIMENT"
  trainer.logger="$LOGGER"
  trainer.resume_mode="$RESUME_MODE"
  trainer.seed="$SEED"
  trainer.total_epochs="$EPOCHS"
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$SAVE_FREQ"
)

if [[ " $* " == *" --cfg "* ]]; then
  exec python3 -m verl.trainer.sft_trainer "${overrides[@]}" "$@"
fi

torchrun_args=(--nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE")
if [ "$NNODES" = "1" ]; then
  torchrun_args+=(--standalone)
else
  torchrun_args+=(--node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
fi
exec torchrun "${torchrun_args[@]}" -m verl.trainer.sft_trainer "${overrides[@]}" "$@"
