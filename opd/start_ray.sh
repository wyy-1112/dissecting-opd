#!/usr/bin/env bash
# Start or join the Ray cluster used by opd/run.py.
# Ray workers inherit the environment of `ray start`, not the driver's, so the
# patched verl, the vLLM prompt-logprob patch and the timeouts are set here.
#
#   head:    VERL_ROOT=/path/to/verl bash opd/start_ray.sh --head --num-gpus=8
#   others:  VERL_ROOT=/path/to/verl bash opd/start_ray.sh --address=<head_ip>:6379 --num-gpus=8
#
# Inst-Mix Agent48 / Mix5 runs also need LLM_FUSION_EXTENSION_ROOT=/path/to/LLM-Fusion.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VERL_ROOT:?set VERL_ROOT to verl 7aed6b2 with opd/verl_patch applied}"

prepend=""
if [[ -n "${LLM_FUSION_EXTENSION_ROOT:-}" ]]; then
  prepend="$HERE/llm_fusion_overlay:"
  export LLM_FUSION_EXTENSION_ROOT
fi
# opd/rewards: the remote reward manager unpickles reward functions in separate workers,
# so their helper modules (judge_contract, ...) must be importable by name.
export PYTHONPATH="${prepend}${VERL_ROOT}:$HERE/rewards${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export OPD_PATCH_VLLM_PROMPT_LOGPROBS=1
export OPD_VLLM_PATCH_PATH="$HERE/vllm_patch"
export VERL_REWARD_RPC_TIMEOUT_SECONDS=${VERL_REWARD_RPC_TIMEOUT_SECONDS:-300}
export VERL_AGENT_SAMPLE_TIMEOUT_SECONDS=${VERL_AGENT_SAMPLE_TIMEOUT_SECONDS:-1800}
export VERL_AGENT_BATCH_TIMEOUT_SECONDS=${VERL_AGENT_BATCH_TIMEOUT_SECONDS:-5400}
export VERL_AGENT_PROGRESS_INTERVAL_SECONDS=${VERL_AGENT_PROGRESS_INTERVAL_SECONDS:-30}

exec ray start --disable-usage-stats "$@"
