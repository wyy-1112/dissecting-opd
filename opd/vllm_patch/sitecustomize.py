"""Compatibility loaded only by OPD vLLM EngineCore / Worker subprocesses."""
from __future__ import annotations

import inspect
import os
import sys
from contextvars import ContextVar


def _log(message: str) -> None:
    # EngineCore workers speak a JSON protocol on stdout. Never print there.
    print(message, file=sys.stderr, flush=True)


def patch_prompt_logprobs() -> None:
    if os.getenv("OPD_PATCH_VLLM_PROMPT_LOGPROBS", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return

    import torch
    from vllm.v1.sample.sampler import Sampler

    descriptor = Sampler.__dict__.get("compute_logprobs")
    if descriptor is None:
        return
    function = descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
    if getattr(function, "_opd_optional_temperature", False):
        return
    if "temp" not in inspect.signature(function).parameters:
        return

    def identity_temperature(logits):
        return torch.ones(
            logits.shape[0],
            device=logits.device,
            dtype=logits.dtype,
        )

    if isinstance(descriptor, staticmethod):
        original = function

        def compute_logprobs(logits, temp=None):
            return original(
                logits,
                identity_temperature(logits) if temp is None else temp,
            )

        compute_logprobs._opd_optional_temperature = True
        Sampler.compute_logprobs = staticmethod(compute_logprobs)
    else:
        original = function

        def compute_logprobs(self, logits, temp=None):
            return original(
                self,
                logits,
                identity_temperature(logits) if temp is None else temp,
            )

        compute_logprobs._opd_optional_temperature = True
        Sampler.compute_logprobs = compute_logprobs

    _log("[opd EngineCore] patched Sampler.compute_logprobs default temperature")


def _extra_logprob_token_ids() -> list[int]:
    raw = os.getenv("OPD_EXTRA_LOGPROB_TOKEN_IDS", "")
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _install_append_keep_extras(module) -> bool:
    fn = getattr(module, "append_logprobs_for_next_position", None)
    if fn is None or getattr(fn, "_opd_extra_tokens", False):
        return False

    def patched_append(request_logprobs, token_ids, logprobs, decoded_tokens, rank, num_logprobs):
        # Teacher scoring requests prompt_logprobs=0, so the unpatched helper
        # zips only the sampled token even when extra columns are present.
        if num_logprobs != -1:
            num_logprobs = max(int(num_logprobs), max(0, len(token_ids) - 1))
        return fn(
            request_logprobs, token_ids, logprobs, decoded_tokens, rank, num_logprobs
        )

    patched_append._opd_extra_tokens = True
    module.append_logprobs_for_next_position = patched_append
    return True


def _patch_v1_prompt_logprob_tokens(
    extra_ids: list[int], torch, LogprobsTensors
) -> bool:
    """Patch the default (V1) GPUModelRunner prompt-scoring path.

    V1 computes prompt logprobs through ``Sampler.gather_logprobs`` and copies
    them into a preallocated CPU tensor. Scope both patches to
    ``_get_prompt_logprobs_dict`` so ordinary rollout logprobs are unchanged.
    """

    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_method = GPUModelRunner._get_prompt_logprobs_dict
    if getattr(original_method, "_opd_extra_tokens", False):
        return False

    prompt_scoring_active: ContextVar[bool] = ContextVar(
        "opd_v1_prompt_scoring_active", default=False
    )
    original_gather = Sampler.gather_logprobs
    original_empty_cpu = LogprobsTensors.empty_cpu

    def gather_logprobs(logprobs, num_logprobs, token_ids):
        result = original_gather(logprobs, num_logprobs, token_ids)
        if not prompt_scoring_active.get():
            return result

        extra = torch.as_tensor(
            extra_ids,
            device=result.logprob_token_ids.device,
            dtype=result.logprob_token_ids.dtype,
        )
        extra = extra.unsqueeze(0).expand(result.logprob_token_ids.shape[0], -1)
        extra_logprobs = logprobs.index_select(
            -1, extra[0].to(dtype=torch.int64)
        )
        return LogprobsTensors(
            logprob_token_ids=torch.cat(
                [result.logprob_token_ids, extra], dim=-1
            ),
            logprobs=torch.cat([result.logprobs, extra_logprobs], dim=-1),
            selected_token_ranks=result.selected_token_ranks,
            cu_num_generated_tokens=result.cu_num_generated_tokens,
        )

    def empty_cpu(num_positions, num_tokens_per_position):
        if prompt_scoring_active.get():
            num_tokens_per_position += len(extra_ids)
        return original_empty_cpu(num_positions, num_tokens_per_position)

    def _get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
        token = prompt_scoring_active.set(True)
        try:
            return original_method(self, hidden_states, num_scheduled_tokens)
        finally:
            prompt_scoring_active.reset(token)

    gather_logprobs._opd_extra_tokens = True
    empty_cpu._opd_extra_tokens = True
    _get_prompt_logprobs_dict._opd_extra_tokens = True
    Sampler.gather_logprobs = staticmethod(gather_logprobs)
    LogprobsTensors.empty_cpu = staticmethod(empty_cpu)
    GPUModelRunner._get_prompt_logprobs_dict = _get_prompt_logprobs_dict
    return True


def patch_extra_prompt_logprob_tokens() -> None:
    extra_ids = _extra_logprob_token_ids()
    if not extra_ids:
        return

    import torch
    import vllm.logprobs as logprobs_mod
    import vllm.v1.engine.logprobs as engine_logprobs_mod
    import vllm.v1.worker.gpu.sample.prompt_logprob as prompt_logprob_mod
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.sample import logprob as logprob_mod

    if not getattr(logprob_mod.compute_topk_logprobs, "_opd_extra_tokens", False):
        original = logprob_mod.compute_topk_logprobs

        def compute_topk_logprobs(logits, num_logprobs, sampled_token_ids, cu_num_logits=None):
            result = original(logits, num_logprobs, sampled_token_ids, cu_num_logits)
            extra = torch.tensor(
                extra_ids, device=logits.device, dtype=result.logprob_token_ids.dtype
            )
            extra = extra.unsqueeze(0).expand(result.logprob_token_ids.shape[0], -1)
            extra_logprobs = logprob_mod.compute_token_logprobs(
                logits, extra.to(torch.int64)
            )
            return LogprobsTensors(
                logprob_token_ids=torch.cat([result.logprob_token_ids, extra], dim=1),
                logprobs=torch.cat([result.logprobs, extra_logprobs], dim=1),
                selected_token_ranks=result.selected_token_ranks,
                cu_num_generated_tokens=result.cu_num_generated_tokens,
            )

        compute_topk_logprobs._opd_extra_tokens = True
        logprob_mod.compute_topk_logprobs = compute_topk_logprobs

    wrapped = logprob_mod.compute_topk_logprobs
    # Live call site imports the function by name. Rebind after that import.
    prompt_logprob_mod.compute_topk_logprobs = wrapped
    for module_name in (
        "vllm.v1.worker.gpu.sample.sampler",
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "compute_topk_logprobs"):
            module.compute_topk_logprobs = wrapped

    _install_append_keep_extras(logprobs_mod)
    _install_append_keep_extras(engine_logprobs_mod)
    patched_v1 = _patch_v1_prompt_logprob_tokens(
        extra_ids, torch, LogprobsTensors
    )

    processor = getattr(engine_logprobs_mod, "LogprobsProcessor", None)
    if processor is not None and not getattr(
        processor._update_prompt_logprobs, "_opd_extra_tokens", False
    ):
        original_update = processor._update_prompt_logprobs

        def _update_prompt_logprobs(self, prompt_logprobs_tensors):
            _token_ids, logprobs, _ranks, _rest = prompt_logprobs_tensors
            saved = self.num_prompt_logprobs
            ncols = int(logprobs.shape[1])
            if saved is not None and saved != -1 and ncols > saved + 1:
                self.num_prompt_logprobs = ncols - 1
            try:
                return original_update(self, prompt_logprobs_tensors)
            finally:
                self.num_prompt_logprobs = saved

        _update_prompt_logprobs._opd_extra_tokens = True
        processor._update_prompt_logprobs = _update_prompt_logprobs

    if patched_v1:
        _log("[opd EngineCore] patched V1 GPUModelRunner prompt logprob extras")
    _log(f"[opd EngineCore] patched extra prompt logprob tokens {extra_ids}")


try:
    patch_prompt_logprobs()
    patch_extra_prompt_logprob_tokens()
except Exception as error:
    _log(
        "[opd EngineCore] prompt-logprob patch failed: "
        f"{type(error).__name__}: {error}"
    )
