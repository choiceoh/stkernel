# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter: run the aux-hidden projection (`fc`) under the target's
head + sampler instead of after them (VLLM_GLM53_DFLASH_EARLY_FC).

The decode step's tail (MEASUREMENTS 25차/27차) is target head -> logits
AllGather -> rejection sampler -> drafter. The drafter's first GEMM, `fc`
([tokens, 5 x 4096] -> [tokens, 4096], 168 MB bf16 / 301 us on the MK W4
lane), only needs the target's aux hidden states, which exist the moment the
target forward returns -- but the stock speculator computes it inside
`propose()`, after the sampler. Everything between is not DRAM-bound (the
AllGather is the fabric, the sampler is small kernels), so the fc can stream
its weights there for free.

Two halves, both above any compiled or captured region:

  producer  a wrapper on `GPUModelRunner.execute_model`: once the target
            forward has returned (execute_model_state carries the aux hidden
            states), a side stream cats the aux states and runs the drafter's
            `fc` into a persistent buffer, and records an event
  consumer  `DFlash2Qwen3ForCausalLM.combine_hidden_states` (the drafter
            overlay): if a result is pending for this step and this token
            count, wait on the event and return the buffer; otherwise the
            stock computation runs (so a missed step costs nothing but the
            overlap)

The consumer waits BEFORE `precompute_and_store_context_kv` and before the
drafter graph, so no megakernel launch ever runs concurrently with the fc's
own MK launch (the lane's monotonic ticket barrier assumes one launch at a
time). Numerics are identical: same kernel, same inputs, one stream apart.

Knob: VLLM_GLM53_DFLASH_EARLY_FC=1 (exact). Anything else is off, and the
installer then does not touch the runner at all.
"""
from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False
_ORIG_EXECUTE_MODEL = None
_DISABLED = False
_SEQ = 0
_STREAM = None


def early_fc_enabled() -> bool:
    return (os.environ.get("VLLM_GLM53_DFLASH_EARLY_FC") or "0").strip() == "1"


def _drafter_of(runner):
    """The drafter ForCausalLM that owns `fc`, or None."""
    spec = getattr(runner, "speculator", None)
    model = getattr(spec, "model", None)
    inner = getattr(model, "model", None)
    if inner is None or getattr(inner, "fc", None) is None:
        return None
    if not getattr(inner, "use_aux_hidden_state", False):
        return None
    if not hasattr(model, "combine_hidden_states"):
        return None
    return model


def launch_early_fc(runner) -> bool:
    """Producer. Called after execute_model; returns whether it launched."""
    global _SEQ, _STREAM
    import torch

    st = getattr(runner, "execute_model_state", None)
    if st is None:
        return False
    aux = getattr(st, "aux_hidden_states", None)
    if not aux:
        return False
    drafter = _drafter_of(runner)
    if drafter is None:
        return False
    input_batch = getattr(st, "input_batch", None)
    n = int(getattr(input_batch, "num_tokens", 0) or 0)
    if n <= 0 or n > aux[0].shape[0]:
        return False
    fc = drafter.model.fc
    width = sum(int(a.shape[-1]) for a in aux)
    if width != int(fc.input_size):
        return False  # a draft/target pair the stock path would reject too
    cur = torch.cuda.current_stream()
    if _STREAM is None:
        _STREAM = torch.cuda.Stream()
    cat_buf = getattr(drafter, "_deneb_early_fc_cat", None)
    if cat_buf is None or cat_buf.shape[0] < aux[0].shape[0]:
        rows = int(aux[0].shape[0])
        cat_buf = torch.empty(rows, width, dtype=aux[0].dtype, device=aux[0].device)
        out_buf = torch.empty(rows, int(fc.output_size), dtype=aux[0].dtype,
                              device=aux[0].device)
        drafter._deneb_early_fc_cat = cat_buf
        drafter._deneb_early_fc_out = out_buf
        drafter._deneb_early_fc_event = torch.cuda.Event()
    out_buf = drafter._deneb_early_fc_out
    _STREAM.wait_stream(cur)
    with torch.cuda.stream(_STREAM):
        torch.cat([a[:n] for a in aux], dim=-1, out=cat_buf[:n])
        h = fc(cat_buf[:n])
        if isinstance(h, tuple):
            h = h[0]
        out_buf[:n].copy_(h)
        drafter._deneb_early_fc_event.record(_STREAM)
    _SEQ += 1
    drafter._deneb_early_fc_pending = (n, _SEQ)
    return True


def take_early_fc(drafter, num_tokens: int):
    """Consumer. The pending buffer for this step, or None (stock path)."""
    import torch

    pend = getattr(drafter, "_deneb_early_fc_pending", None)
    if pend is None:
        return None
    drafter._deneb_early_fc_pending = None  # consumed once, whatever happens
    n, _seq = pend
    if n != num_tokens:
        return None
    torch.cuda.current_stream().wait_event(drafter._deneb_early_fc_event)
    return drafter._deneb_early_fc_out[:n]


def _patched_execute_model(self, *args, **kwargs):
    global _DISABLED
    out = _ORIG_EXECUTE_MODEL(self, *args, **kwargs)
    if not _DISABLED:
        try:
            launch_early_fc(self)
        except Exception:
            _DISABLED = True
            logger.exception("[dflash-early-fc] producer failed -> stock path "
                             "for the rest of the boot")
    return out


def install_glm53_dflash_early_fc() -> bool:
    """Wrap the runner's execute_model once. Safe to call from model import."""
    global _INSTALLED, _ORIG_EXECUTE_MODEL
    if _INSTALLED or not early_fc_enabled():
        return _INSTALLED
    import sys

    mr = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if mr is None:
        try:
            import vllm.v1.worker.gpu.model_runner as mr  # noqa: F811
        except Exception:
            logger.exception("[dflash-early-fc] runner module not importable -> off")
            return False
    Runner = getattr(mr, "GPUModelRunner", None)
    if Runner is None or not hasattr(Runner, "execute_model"):
        logger.warning("[dflash-early-fc] GPUModelRunner.execute_model missing -> off")
        return False
    _ORIG_EXECUTE_MODEL = Runner.execute_model
    Runner.execute_model = _patched_execute_model
    _INSTALLED = True
    logger.warning("[dflash-early-fc] installed: the drafter's fc runs under the "
                   "target head + sampler on a side stream")
    return True
