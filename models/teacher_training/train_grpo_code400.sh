#!/usr/bin/env bash
# GRPO training of the Code GRPO-400 teacher from the Qwen3-1.7B-Base SFT-554 student.
#
# This teacher predates the Qwen3-4B teacher recipe (train_grpo.sh) and keeps its own
# settings: KL loss 0.001, sampling T = 0.8 / top-p 0.95, no rollout importance sampling,
# remote reward manager. The hydra overrides are those of the original run.
set -euo pipefail
umask 077

: "${VERL_ROOT:?VERL_ROOT must point at the patched verl checkout}"
test -f "$VERL_ROOT/verl/__init__.py"
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL:?}"
: "${TOKENIZER:?}"
: "${TRAIN_FILE:?}"
: "${VAL_FILE:?}"
: "${REWARD_FN:?}"
: "${OUTPUT_DIR:?}"
WANDB_PROJECT=${WANDB_PROJECT:-opd-teacher-grpo}
WANDB_EXPERIMENT=${WANDB_EXPERIMENT:-$(basename "$OUTPUT_DIR")}
LOGGER=${LOGGER:-'["console"]'}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
ACTOR_LR=${ACTOR_LR:-5e-7}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
REWARD_WORKERS=${REWARD_WORKERS:-4}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
RESUME_MODE=${RESUME_MODE:-auto}
NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

test -f "$MODEL/config.json"
test -d "$TOKENIZER"
test -f "$TRAIN_FILE"
test -f "$VAL_FILE"
test -f "$REWARD_FN"
mkdir -p "$OUTPUT_DIR"

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=8 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.tokenizer_path="$TOKENIZER" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$REWARD_FN" \
  reward.custom_reward_function.name=compute_score_compact \
  +reward.custom_reward_function.reward_kwargs.timeout=3.0 \
  +reward.custom_reward_function.reward_kwargs.max_tests=10 \
  reward.reward_manager.name=remote \
  reward.num_workers="$REWARD_WORKERS" \
  reward.reward_model.enable=False \
  trainer.balance_batch=True \
  trainer.logger="$LOGGER" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$WANDB_EXPERIMENT" \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.max_actor_ckpt_to_keep=5 \
  +ray_kwargs.ray_init.address=auto \
  "$@"
