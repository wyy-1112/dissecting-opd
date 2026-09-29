"""Mapping between train_opd.sh environment variables and resolved verl config keys.

Each entry is (ENV_NAME, dotted verl config path). train_opd.sh forwards the
variable to exactly that hydra key, so a resolved config determines the
environment of a run and vice versa.
"""

ENV_TO_CONFIG = [
    ("TRAIN_FILE", "data.train_files"),
    ("DEV_FILE", "data.val_files"),
    ("TRAIN_MAX_SAMPLES", "data.train_max_samples"),
    ("DATA_SHUFFLE", "data.shuffle"),
    ("TRAIN_BATCH_SIZE", "data.train_batch_size"),
    ("MAX_PROMPT_LENGTH", "data.max_prompt_length"),
    ("MAX_RESPONSE_LENGTH", "data.max_response_length"),
    ("TRAINING_SEED", "data.seed"),
    ("OPD_ENABLE_THINKING", "data.apply_chat_template_kwargs.enable_thinking"),
    ("STUDENT_MODEL", "actor_rollout_ref.model.path"),
    ("STUDENT_TOKENIZER", "actor_rollout_ref.model.tokenizer_path"),
    ("ACTOR_LR", "actor_rollout_ref.actor.optim.lr"),
    ("PPO_MINI_BATCH_SIZE", "actor_rollout_ref.actor.ppo_mini_batch_size"),
    ("ACTOR_PPO_EPOCHS", "actor_rollout_ref.actor.ppo_epochs"),
    ("PPO_MAX_TOKEN_LEN_PER_GPU", "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"),
    ("FSDP_OPTIMIZER_OFFLOAD", "actor_rollout_ref.actor.fsdp_config.optimizer_offload"),
    ("FSDP_USE_ORIG_PARAMS", "actor_rollout_ref.actor.fsdp_config.use_orig_params"),
    ("STUDENT_ROLLOUT_GPU_MEMORY_UTILIZATION", "actor_rollout_ref.rollout.gpu_memory_utilization"),
    ("ROLLOUT_ENFORCE_EAGER", "actor_rollout_ref.rollout.enforce_eager"),
    ("STUDENT_MAX_MODEL_LEN", "actor_rollout_ref.rollout.max_model_len"),
    ("ROLLOUT_N", "actor_rollout_ref.rollout.n"),
    ("ROLLOUT_TEMPERATURE", "actor_rollout_ref.rollout.temperature"),
    ("ROLLOUT_TOP_P", "actor_rollout_ref.rollout.top_p"),
    ("ROLLOUT_MAX_NUM_SEQS", "actor_rollout_ref.rollout.max_num_seqs"),
    ("ROLLOUT_MAX_NUM_BATCHED_TOKENS", "actor_rollout_ref.rollout.max_num_batched_tokens"),
    ("ROLLOUT_CALCULATE_LOG_PROBS", "actor_rollout_ref.rollout.calculate_log_probs"),
    ("LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"),
    ("LOG_PROB_MAX_TOKEN_LEN_PER_GPU", "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"),
    ("VAL_TEMPERATURE", "actor_rollout_ref.rollout.val_kwargs.temperature"),
    ("VAL_TOP_P", "actor_rollout_ref.rollout.val_kwargs.top_p"),
    ("VAL_N", "actor_rollout_ref.rollout.val_kwargs.n"),
    ("ROLLOUT_IS", "algorithm.rollout_correction.rollout_is"),
    ("ROLLOUT_IS_THRESHOLD", "algorithm.rollout_correction.rollout_is_threshold"),
    ("ROLLOUT_IS_BATCH_NORMALIZE", "algorithm.rollout_correction.rollout_is_batch_normalize"),
    ("ROLLOUT_BYPASS_MODE", "algorithm.rollout_correction.bypass_mode"),
    ("REPLAY_ENABLE", "algorithm.replay.enable"),
    ("REPLAY_BUFFER_BATCHES", "algorithm.replay.buffer_batches"),
    ("REPLAY_UPDATES_PER_CYCLE", "algorithm.replay.updates_per_cycle"),
    ("REPLAY_TOKEN_BUDGET_FRACTION", "algorithm.replay.token_budget_fraction"),
    ("REPLAY_PRIORITY_SIGNAL", "algorithm.replay.priority_signal"),
    ("REPLAY_PRIORITY_EXPONENT", "algorithm.replay.priority_exponent"),
    ("REPLAY_RMS_K1_CLAMP", "algorithm.replay.rms_k1_clamp"),
    ("REPLAY_MIN_INCLUSION_PROB", "algorithm.replay.min_inclusion_prob"),
    ("REPLAY_SEED", "algorithm.replay.seed"),
    ("REPLAY_ADAPTIVE_DEPTH", "algorithm.replay.adaptive_depth"),
    ("REPLAY_MIN_REPLAYS_PER_CYCLE", "algorithm.replay.min_replays_per_cycle"),
    ("REPLAY_LEARN_GAP_FLOOR_FRACTION", "algorithm.replay.learn_gap_floor_fraction"),
    ("REPLAY_CLIP_FRACTION_CEILING", "algorithm.replay.clip_fraction_ceiling"),
    ("REPLAY_ESS_FLOOR_FRACTION", "algorithm.replay.ess_floor_fraction"),
    ("REPLAY_MAX_OPTIMIZER_UPDATES", "algorithm.replay.max_optimizer_updates"),
    ("DISTILLATION_ENABLE_RESOURCE_POOL", "distillation.enable_resource_pool"),
    ("TEACHER_GPUS_PER_NODE", "distillation.n_gpus_per_node"),
    ("TEACHER_NNODES", "distillation.nnodes"),
    ("TEACHER_MODEL", "distillation.teacher_models.teacher_model.model_path"),
    ("TEACHER_TP", "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size"),
    ("TEACHER_GPU_MEMORY_UTILIZATION", "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization"),
    ("TEACHER_MAX_MODEL_LEN", "distillation.teacher_models.teacher_model.inference.max_model_len"),
    ("TEACHER_MAX_NUM_BATCHED_TOKENS", "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens"),
    ("TEACHER_MAX_NUM_SEQS", "distillation.teacher_models.teacher_model.inference.max_num_seqs"),
    ("TEACHER_TEMPERATURE", "distillation.teacher_models.teacher_model.inference.temperature"),
    ("REWARD_FN", "reward.custom_reward_function.path"),
    ("REWARD_FN_NAME", "reward.custom_reward_function.name"),
    ("REWARD_TIMEOUT", "reward.custom_reward_function.reward_kwargs.timeout"),
    ("REWARD_MAX_TESTS", "reward.custom_reward_function.reward_kwargs.max_tests"),
    ("REWARD_MANAGER", "reward.reward_manager.name"),
    ("REWARD_WORKERS", "reward.num_workers"),
    ("STUDENT_GPUS_PER_NODE", "trainer.n_gpus_per_node"),
    ("STUDENT_NNODES", "trainer.nnodes"),
    ("VAL_BEFORE_TRAIN", "trainer.val_before_train"),
    ("SAVE_FREQ", "trainer.save_freq"),
    ("TEST_FREQ", "trainer.test_freq"),
    ("TOTAL_EPOCHS", "trainer.total_epochs"),
    ("TOTAL_TRAINING_STEPS", "trainer.total_training_steps"),
    ("OUTPUT_DIR", "trainer.default_local_dir"),
    ("RESUME_MODE", "trainer.resume_mode"),
    ("RESUME_FROM_PATH", "trainer.resume_from_path"),
    ("LOG_VAL_GENERATIONS", "trainer.log_val_generations"),
    ("ROLLOUT_DATA_DIR", "trainer.rollout_data_dir"),
    ("VALIDATION_DATA_DIR", "trainer.validation_data_dir"),
    ("WANDB_PROJECT", "trainer.project_name"),
    ("WANDB_EXPERIMENT", "trainer.experiment_name"),
]

# Keys fixed inside train_opd.sh (identical for every run of the paper).
FIXED_IN_DRIVER = {
    "algorithm.adv_estimator": "grpo",
    "algorithm.use_kl_in_reward": False,
    "actor_rollout_ref.actor.use_kl_loss": False,
    "actor_rollout_ref.actor.entropy_coeff": 0,
    "distillation.enabled": True,
    "distillation.distillation_loss.loss_mode": "k1",
    "distillation.distillation_loss.use_task_rewards": False,
    "distillation.distillation_loss.use_policy_gradient": True,
    "distillation.distillation_loss.loss_max_clamp": 10.0,
    "distillation.distillation_loss.log_prob_min_clamp": -10.0,
}

# Paths / bookkeeping that legitimately differ between the original cluster and a re-run.
PATH_KEYS = {
    "TRAIN_FILE", "DEV_FILE", "STUDENT_MODEL", "STUDENT_TOKENIZER", "TEACHER_MODEL", "REWARD_FN",
    "OUTPUT_DIR", "RESUME_FROM_PATH", "ROLLOUT_DATA_DIR", "VALIDATION_DATA_DIR", "WANDB_PROJECT", "WANDB_EXPERIMENT",
}


def get_path(config: dict, dotted: str):
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node
