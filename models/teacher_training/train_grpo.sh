#!/usr/bin/env bash
# GRPO teacher training for the Qwen3-4B teachers (Math GRPO-500, Code GRPO-300).
#
# The hydra overrides are those of the original runs; per-teacher values come in as
# environment variables (configs/teacher_training/*.yaml, launched with opd/run.py).
# Requires a running Ray cluster (opd/start_ray.sh) and VERL_ROOT pointing at verl 7aed6b2
# with opd/verl_patch applied.
set -euo pipefail
umask 077

: "${VERL_ROOT:?VERL_ROOT must point at the patched verl checkout}"
test -f "$VERL_ROOT/verl/__init__.py"
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL:?}"
: "${TRAIN_FILE:?}"
: "${VAL_FILES:?comma-separated validation parquets}"
: "${REWARD_FN:?}"
: "${OUTPUT_DIR:?}"
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"$OUTPUT_DIR/validation_generations"}
WANDB_PROJECT=${WANDB_PROJECT:-opd-teacher-grpo}
WANDB_EXPERIMENT=${WANDB_EXPERIMENT:-$(basename "$OUTPUT_DIR")}
LOGGER=${LOGGER:-'["console"]'}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-8}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
USE_KL_LOSS=${USE_KL_LOSS:-False}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_IS=${ROLLOUT_IS:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-5.0}
ROLLOUT_BYPASS_MODE=${ROLLOUT_BYPASS_MODE:-false}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
DATA_SEED=${DATA_SEED:-42}
VAL_N=${VAL_N:-4}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-50}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-10}
RESUME_MODE=${RESUME_MODE:-auto}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

test -f "$MODEL/config.json"
test -f "$TRAIN_FILE"
test -f "$REWARD_FN"
val_list=""
IFS=',' read -r -a val_files <<< "$VAL_FILES"
for file in "${val_files[@]}"; do
  test -f "$file"
  val_list="${val_list:+$val_list,}'$file'"
done
mkdir -p "$OUTPUT_DIR" "$VALIDATION_DATA_DIR"

# Only the code teacher sets these (the math reward takes no kwargs).
reward_args=()
[[ -n "${REWARD_TIMEOUT:-}" ]] && reward_args+=("+reward.custom_reward_function.reward_kwargs.timeout=$REWARD_TIMEOUT")
[[ -n "${REWARD_MAX_TESTS:-}" ]] && reward_args+=("+reward.custom_reward_function.reward_kwargs.max_tests=$REWARD_MAX_TESTS")
[[ -n "${REWARD_NUM_WORKERS:-}" ]] && reward_args+=("reward.num_workers=$REWARD_NUM_WORKERS")

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.rollout_correction.rollout_is="$ROLLOUT_IS" \
  algorithm.rollout_correction.rollout_is_threshold="$ROLLOUT_IS_THRESHOLD" \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode="$ROLLOUT_BYPASS_MODE" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="[$val_list]" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  data.seed="$DATA_SEED" \
  data.dataloader_num_workers=8 \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$REWARD_FN" \
  reward.custom_reward_function.name="$REWARD_FN_NAME" \
  "${reward_args[@]}" \
  reward.reward_manager.name=naive \
  reward.reward_model.enable=False \
  trainer.balance_batch=True \
  trainer.logger="$LOGGER" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$WANDB_EXPERIMENT" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode="$RESUME_MODE" \
  +ray_kwargs.ray_init.address=auto \
  "$@"
