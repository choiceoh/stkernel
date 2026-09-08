# SPDX-License-Identifier: Apache-2.0
# deneb fork (2026-09-06, 37차 night round): dump the target's per-token
# features for drafter training -- the five aux hidden states the DFlash2
# drafter consumes (target_layer_ids), the token ids and positions, and the
# target's top-k logits with the log-sum-exp -- from PREFILL steps, to NVMe.
#
# Inert unless VLLM_GLM53_DRAFT_DUMP names a directory. Installed from the
# model wiring like glm53_prep_fused: wraps GPUModelRunner.execute_model and
# runs after it, reading execute_model_state. compute_logits over every
# prefill token is a TP collective (the head is vocab-parallel), so EVERY
# rank computes it in identical 1024-token chunks; only rank 0 writes.
# Anything raised inside the hook disables it for the boot (logged once);
# the step itself is never affected. Records: <dir>/rank<r>_<step>.pt.
import logging
import os
import time

import torch

logger = logging.getLogger("vllm.glm53.draft_dump")

ENV_DIR = "VLLM_GLM53_DRAFT_DUMP"
ENV_TOPK = "VLLM_GLM53_DRAFT_DUMP_TOPK"
ENV_MAX_TOKENS = "VLLM_GLM53_DRAFT_DUMP_MAX_TOKENS"
ENV_MIN_TOKENS = "VLLM_GLM53_DRAFT_DUMP_MIN_STEP_TOKENS"
CHUNK = 1024

_STATE = {"installed": False, "disabled": False, "steps": 0, "tokens": 0, "files": 0,
          "t_hook": 0.0, "logged_first": False}


def dump_dir() -> str | None:
    d = os.environ.get(ENV_DIR, "").strip()
    return d or None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _topk_logits(runner, hidden: torch.Tensor, k: int):
    """(ids int32 [n,k], logits fp16 [n,k], lse fp32 [n]) over the vocab head,
    in fixed chunks on every rank (the head gathers across TP)."""
    ids, vals, lses = [], [], []
    for i in range(0, hidden.shape[0], CHUNK):
        logits = runner.model.compute_logits(hidden[i:i + CHUNK])
        lf = logits.float()
        v, ix = torch.topk(lf, k, dim=-1)
        ids.append(ix.to(torch.int32))
        vals.append(v.to(torch.float16))
        lses.append(torch.logsumexp(lf, dim=-1))
        del logits, lf
    return torch.cat(ids), torch.cat(vals), torch.cat(lses)


def _dump_step(runner) -> None:
    st = getattr(runner, "execute_model_state", None)
    if st is None or st.hidden_states is None or not st.aux_hidden_states:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    ib = st.input_batch
    n = int(ib.num_tokens)
    if n < _int_env(ENV_MIN_TOKENS, 64):
        return  # a decode step (num_reqs x (k+1) tokens): not training data
    if _STATE["tokens"] >= _int_env(ENV_MAX_TOKENS, 5_000_000):
        return
    t0 = time.perf_counter()
    k = _int_env(ENV_TOPK, 16)
    hidden = st.hidden_states[:n]
    top_ids, top_vals, lse = _topk_logits(runner, hidden, k)   # collective: all ranks
    aux = torch.stack([a[:n] for a in st.aux_hidden_states], dim=1)  # [n, L, H]
    rank = _rank()
    if rank == 0:
        d = dump_dir()
        os.makedirs(d, exist_ok=True)
        rec = {
            "req_ids": list(ib.req_ids),
            "query_start_loc": torch.as_tensor(ib.query_start_loc_np[: ib.num_reqs + 1]).clone(),
            "num_computed_tokens": torch.as_tensor(ib.num_computed_tokens_np[: ib.num_reqs]).clone(),
            "input_ids": ib.input_ids[:n].to(torch.int32).cpu(),
            "positions": ib.positions[:n].cpu(),
            "aux_hidden": aux.to(torch.bfloat16).cpu(),
            "topk_ids": top_ids.cpu(),
            "topk_logits": top_vals.cpu(),
            "lse": lse.cpu(),
            "aux_layers": list(getattr(runner.speculative_config, "eagle_aux_hidden_state_layer_ids", []) or []),
            "step": _STATE["steps"],
        }
        path = os.path.join(d, f"rank{rank}_{_STATE['steps']:07d}.pt")
        torch.save(rec, path + ".tmp")
        os.replace(path + ".tmp", path)
        _STATE["files"] += 1
    _STATE["steps"] += 1
    _STATE["tokens"] += n
    _STATE["t_hook"] += time.perf_counter() - t0
    if not _STATE["logged_first"] or _STATE["steps"] % 100 == 0:
        _STATE["logged_first"] = True
        logger.warning("[draft-dump] rank%d step %d: %d tokens this step, %d total, %d files, hook %.2fs total",
                       rank, _STATE["steps"], n, _STATE["tokens"], _STATE["files"], _STATE["t_hook"])


def _patched_execute_model(orig):
    def execute_model(self, scheduler_output, *args, **kwargs):
        out = orig(self, scheduler_output, *args, **kwargs)
        if not _STATE["disabled"]:
            try:
                _dump_step(self)
            except Exception:
                _STATE["disabled"] = True
                logger.exception("[draft-dump] hook failed -> disabled for this boot")
        return out
    return execute_model


def install_glm53_draft_dump() -> bool:
    """Inert without VLLM_GLM53_DRAFT_DUMP. Safe to call from model __init__."""
    if _STATE["installed"] or dump_dir() is None:
        return _STATE["installed"]
    try:
        import vllm.v1.worker.gpu.model_runner as mr
    except Exception:
        logger.exception("[draft-dump] runner module not importable -> not installed")
        return False
    Runner = mr.GPUModelRunner
    Runner.execute_model = _patched_execute_model(Runner.execute_model)
    _STATE["installed"] = True
    logger.warning("[draft-dump] installed: dir=%s topk=%d max_tokens=%d (prefill steps >= %d tokens)",
                   dump_dir(), _int_env(ENV_TOPK, 16), _int_env(ENV_MAX_TOKENS, 5_000_000),
                   _int_env(ENV_MIN_TOKENS, 64))
    return True
