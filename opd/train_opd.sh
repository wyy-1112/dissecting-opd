#!/usr/bin/env bash
# One OPD training run (verl synchronous trainer + distillation).
#
# Every experiment in this release is this script plus a set of environment
# variables (see configs/ and opd/run.py). The hydra overrides at the bottom
# are exactly those used for the paper runs; defaults below apply only when a
# config leaves the variable unset.
#
# Required: STUDENT_MODEL TEACHER_MODEL TRAIN_FILE DEV_FILE OUTPUT_DIR
# PYTHONPATH_PREPEND (optional) is placed before VERL_ROOT; the IF/Agent runs use it
# for opd/llm_fusion_overlay. Requires a running Ray cluster (RAY_ADDRESS, default "auto") and VERL_ROOT
# pointing at verl 7aed6b2 with opd/verl_patch applied.
set -euo pipefail
umask 077

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VERL_ROOT:?VERL_ROOT must point at the patched verl checkout}"
test -f "$VERL_ROOT/verl/__init__.py"
export PYTHONPATH="${PYTHONPATH_PREPEND:+$PYTHONPATH_PREPEND:}$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPD_PATCH_VLLM_PROMPT_LOGPROBS=${OPD_PATCH_VLLM_PROMPT_LOGPROBS:-1}
export OPD_VLLM_PATCH_PATH=${OPD_VLLM_PATCH_PATH:-"$HERE/vllm_patch"}

: "${STUDENT_MODEL:?}"
: "${TEACHER_MODEL:?}"
: "${TRAIN_FILE:?}"
: "${DEV_FILE:?}"
: "${OUTPUT_DIR:?}"
STUDENT_TOKENIZER=${STUDENT_TOKENIZER:-"$STUDENT_MODEL"}
REWARD_FN=${REWARD_FN:-"$HERE/rewards/noop_reward.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
WANDB_PROJECT=${WANDB_PROJECT:-opd}
WANDB_EXPERIMENT=${WANDB_EXPERIMENT:-$(basename "$OUTPUT_DIR")}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"$OUTPUT_DIR/rollouts"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"$OUTPUT_DIR/validation_generations"}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}

STUDENT_NNODES=${STUDENT_NNODES:-3}
TEACHER_NNODES=${TEACHER_NNODES:-1}
STUDENT_GPUS_PER_NODE=${STUDENT_GPUS_PER_NODE:-8}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-8}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
DATA_SHUFFLE=${DATA_SHUFFLE:-True}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-"$TRAIN_BATCH_SIZE"}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_IS=${ROLLOUT_IS:-null}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-False}
ROLLOUT_BYPASS_MODE=${ROLLOUT_BYPASS_MODE:-False}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-8}
ACTOR_LR=${ACTOR_LR:-5e-7}
TRAINING_SEED=${TRAINING_SEED:-42}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
VAL_N=${VAL_N:-1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-False}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
REWARD_TIMEOUT=${REWARD_TIMEOUT-3.0}
REWARD_MAX_TESTS=${REWARD_MAX_TESTS-10}
LOGGER=${LOGGER:-'["console"]'}
USE_WANDB=${USE_WANDB:-0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
STUDENT_MAX_MODEL_LEN=${STUDENT_MAX_MODEL_LEN:-8192}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-9216}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-20480}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-64}
STUDENT_ROLLOUT_GPU_MEMORY_UTILIZATION=${STUDENT_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.5}
DISTILLATION_ENABLE_RESOURCE_POOL=${DISTILLATION_ENABLE_RESOURCE_POOL:-True}
REWARD_WORKERS=${REWARD_WORKERS:-1}
REWARD_MANAGER=${REWARD_MANAGER:-naive}
RESUME_MODE=${RESUME_MODE:-auto}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
FSDP_USE_ORIG_PARAMS=${FSDP_USE_ORIG_PARAMS:-False}
FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-False}

# Replay of prepared rollout batches. Defaults keep every optimizer update on a
# freshly generated batch, so existing runs are unaffected.
ACTOR_PPO_EPOCHS=${ACTOR_PPO_EPOCHS:-1}
REPLAY_ENABLE=${REPLAY_ENABLE:-False}
REPLAY_BUFFER_BATCHES=${REPLAY_BUFFER_BATCHES:-1}
REPLAY_UPDATES_PER_CYCLE=${REPLAY_UPDATES_PER_CYCLE:-4}
REPLAY_TOKEN_BUDGET_FRACTION=${REPLAY_TOKEN_BUDGET_FRACTION:-0.25}
REPLAY_PRIORITY_SIGNAL=${REPLAY_PRIORITY_SIGNAL:-length}
REPLAY_PRIORITY_EXPONENT=${REPLAY_PRIORITY_EXPONENT:-0.0}
REPLAY_RMS_K1_CLAMP=${REPLAY_RMS_K1_CLAMP:-10.0}
REPLAY_MIN_INCLUSION_PROB=${REPLAY_MIN_INCLUSION_PROB:-0.05}
REPLAY_SEED=${REPLAY_SEED:-"$TRAINING_SEED"}
REPLAY_ADAPTIVE_DEPTH=${REPLAY_ADAPTIVE_DEPTH:-False}
REPLAY_MIN_REPLAYS_PER_CYCLE=${REPLAY_MIN_REPLAYS_PER_CYCLE:-1}
REPLAY_LEARN_GAP_FLOOR_FRACTION=${REPLAY_LEARN_GAP_FLOOR_FRACTION:-0.7}
REPLAY_CLIP_FRACTION_CEILING=${REPLAY_CLIP_FRACTION_CEILING:-0.15}
REPLAY_ESS_FLOOR_FRACTION=${REPLAY_ESS_FLOOR_FRACTION:-0.5}
REPLAY_MAX_OPTIMIZER_UPDATES=${REPLAY_MAX_OPTIMIZER_UPDATES:-null}

test -f "$STUDENT_MODEL/config.json"
test -f "$TEACHER_MODEL/config.json"
test -d "$STUDENT_TOKENIZER"
test -f "$TRAIN_FILE"
IFS=',' read -r -a DEV_FILES <<< "$DEV_FILE"
for dev_file in "${DEV_FILES[@]}"; do
  test -f "$dev_file"
done
DEV_FILE_OVERRIDE="$DEV_FILE"
if [ "${#DEV_FILES[@]}" -gt 1 ]; then
  DEV_FILE_OVERRIDE="$(
    python3 - "${DEV_FILES[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:], separators=(",", ":")))
PY
  )"
fi
test -f "$REWARD_FN"
mkdir -p "$OUTPUT_DIR"


export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

chat_template_args=()
if [[ -n "${OPD_ENABLE_THINKING:-}" ]]; then
  case "$OPD_ENABLE_THINKING" in
    true|false|True|False) ;;
    *) echo "ERROR: OPD_ENABLE_THINKING must be true or false" >&2; exit 2 ;;
  esac
  chat_template_args+=("+data.apply_chat_template_kwargs.enable_thinking=$OPD_ENABLE_THINKING")
fi
reward_kwargs=()
if [[ -n "$REWARD_TIMEOUT" ]]; then
  reward_kwargs+=("+reward.custom_reward_function.reward_kwargs.timeout=$REWARD_TIMEOUT")
fi
if [[ -n "$REWARD_MAX_TESTS" ]]; then
  reward_kwargs+=("+reward.custom_reward_function.reward_kwargs.max_tests=$REWARD_MAX_TESTS")
fi

python3 -m verl.trainer.main_ppo \
  "${chat_template_args[@]}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is="$ROLLOUT_IS" \
  algorithm.rollout_correction.rollout_is_threshold="$ROLLOUT_IS_THRESHOLD" \
  algorithm.rollout_correction.rollout_is_batch_normalize="$ROLLOUT_IS_BATCH_NORMALIZE" \
  algorithm.rollout_correction.bypass_mode="$ROLLOUT_BYPASS_MODE" \
  algorithm.replay.enable="$REPLAY_ENABLE" \
  algorithm.replay.buffer_batches="$REPLAY_BUFFER_BATCHES" \
  algorithm.replay.updates_per_cycle="$REPLAY_UPDATES_PER_CYCLE" \
  algorithm.replay.token_budget_fraction="$REPLAY_TOKEN_BUDGET_FRACTION" \
  algorithm.replay.priority_signal="$REPLAY_PRIORITY_SIGNAL" \
  algorithm.replay.priority_exponent="$REPLAY_PRIORITY_EXPONENT" \
  algorithm.replay.rms_k1_clamp="$REPLAY_RMS_K1_CLAMP" \
  algorithm.replay.min_inclusion_prob="$REPLAY_MIN_INCLUSION_PROB" \
  algorithm.replay.seed="$REPLAY_SEED" \
  algorithm.replay.adaptive_depth="$REPLAY_ADAPTIVE_DEPTH" \
  algorithm.replay.min_replays_per_cycle="$REPLAY_MIN_REPLAYS_PER_CYCLE" \
  algorithm.replay.learn_gap_floor_fraction="$REPLAY_LEARN_GAP_FLOOR_FRACTION" \
  algorithm.replay.clip_fraction_ceiling="$REPLAY_CLIP_FRACTION_CEILING" \
  algorithm.replay.ess_floor_fraction="$REPLAY_ESS_FLOOR_FRACTION" \
  algorithm.replay.max_optimizer_updates="$REPLAY_MAX_OPTIMIZER_UPDATES" \
  actor_rollout_ref.actor.use_kl_loss=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$DEV_FILE_OVERRIDE" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.shuffle="$DATA_SHUFFLE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed="$TRAINING_SEED" \
  data.dataloader_num_workers=8 \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.model.tokenizer_path="$STUDENT_TOKENIZER" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs="$ACTOR_PPO_EPOCHS" \
  actor_rollout_ref.actor.data_loader_seed="$TRAINING_SEED" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.use_orig_params="$FSDP_USE_ORIG_PARAMS" \
  actor_rollout_ref.actor.fsdp_config.seed="$TRAINING_SEED" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$STUDENT_ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.max_model_len="$STUDENT_MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.calculate_log_probs="$ROLLOUT_CALCULATE_LOG_PROBS" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
  distillation.enabled=True \
  distillation.enable_resource_pool="$DISTILLATION_ENABLE_RESOURCE_POOL" \
  distillation.n_gpus_per_node="$TEACHER_GPUS_PER_NODE" \
  distillation.nnodes="$TEACHER_NNODES" \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$TEACHER_TP" \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEMORY_UTILIZATION" \
  distillation.teacher_models.teacher_model.inference.max_model_len="$TEACHER_MAX_MODEL_LEN" \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="$TEACHER_MAX_NUM_BATCHED_TOKENS" \
  distillation.teacher_models.teacher_model.inference.max_num_seqs="$TEACHER_MAX_NUM_SEQS" \
  distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True \
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True \
  distillation.teacher_models.teacher_model.inference.temperature="$TEACHER_TEMPERATURE" \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  reward.custom_reward_function.path="$REWARD_FN" \
  reward.custom_reward_function.name="$REWARD_FN_NAME" \
  "${reward_kwargs[@]}" \
  reward.reward_manager.name="$REWARD_MANAGER" \
  reward.num_workers="$REWARD_WORKERS" \
  reward.reward_model.enable=False \
  trainer.balance_batch=True \
  trainer.logger="$LOGGER" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$WANDB_EXPERIMENT" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
  trainer.n_gpus_per_node="$STUDENT_GPUS_PER_NODE" \
  trainer.nnodes="$STUDENT_NNODES" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.resume_from_path="$RESUME_FROM_PATH" \
  +ray_kwargs.ray_init.address=auto \
  "$@"
