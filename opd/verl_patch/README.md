# verl patch

`verl-7aed6b2-opd.patch` turns upstream verl commit `7aed6b230776f963fa09509c10d9c3a767d1102c` into the
tree used for every OPD and GRPO run in this release.

```bash
git clone https://github.com/verl-project/verl && cd verl
git checkout 7aed6b230776f963fa09509c10d9c3a767d1102c
git apply /path/to/dissecting-opd/opd/verl_patch/verl-7aed6b2-opd.patch
```

Applying it to a clean checkout reproduces the training tree file for file (checked on the release build).

## What it changes

Used by the paper runs:

- `experimental/teacher_loop/*`, `experimental/agent_loop/agent_loop.py`: carry the token ids the teacher
  scored together with its log-probs, so student and teacher log-probs are aligned token by token; bounded
  per-sample and per-batch rollout timeouts (`VERL_AGENT_SAMPLE_TIMEOUT_SECONDS`,
  `VERL_AGENT_BATCH_TIMEOUT_SECONDS`) with progress logging.
- `workers/rollout/vllm_rollout/vllm_async_server.py`, `utils.py`: load `opd/vllm_patch/sitecustomize.py` in
  vLLM worker processes when `OPD_PATCH_VLLM_PROMPT_LOGPROBS=1` (prompt log-probs computed without a
  sampling temperature, needed for teacher scoring). With vLLM 0.20.3 `Sampler.compute_logprobs` already
  takes no temperature, so this part of the patch is a no-op there.
- `trainer/distillation/losses.py`, `workers/config/distillation.py`, `trainer/config/distillation/*.yaml`:
  k1 distillation reward used as a policy-gradient advantage with a ±10 loss clamp (upstream semantics),
  plus the response loss-mask options below. `log_prob_min_clamp` only applies to top-k losses and has no
  effect in k1 mode.
- `experimental/reward_loop/reward_manager/{naive,remote}.py`, `utils/reward_score/*`: reward timeouts and
  validation resilience (rewards are only monitored in OPD).
- `workers/utils/padding.py`: padding of the teacher token ids / log-probs.
- `utils/tracking.py`: wandb init timeout, proxy and run group from environment variables.

Present but disabled in every released config (defaults keep upstream behaviour):

- `trainer/ppo/replay.py`, `utils/replay_weights.py`, `trainer/config/algorithm.py` (`algorithm.replay.*`):
  trajectory replay buffer for replay-extension experiments (`REPLAY_ENABLE=False`).
- `trainer/distillation/loss_masks.py` (`max_response_loss_tokens`, `response_loss_suffix_tokens`).
- `trainer/distillation/virtual_teacher.py` (`virtual_teacher_mode: none`).
- `utils/opd_gradient_sketch.py`, `workers/engine/*` (`OPD_GRAD_DIAGNOSTICS=0`): per-prompt gradient
  sketches used to precompute gradient features for selection.

`tests/` contains the CPU tests written for these changes.
