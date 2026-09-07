# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter: no per-step host build of the draft attention metadata
when the drafter step is a FULL cudagraph replay (VLLM_GLM53_DRAFTER_PREP).

What the 34차 trace showed (2026-09-05, production boot, rank 0 / rank 3):
between the last `reshape_and_cache` of `precompute_and_store_context_kv`
and the first kernel of the drafter graph the GPU idles 420 / 640 us per
step. Inside that gap the host does exactly this, in `DFlashSpeculator.
propose`: `_build_draft_attn_metadata` -> `build_attn_metadata` -> the
FlashInfer builder, which -- because the drafter's block attention is
non-causal, so `all_uses_trtllm` is False -- reads
`common_attn_metadata.seq_lens_cpu`: a `.cpu()` of the GPU seq_lens, i.e. a
pageable DtoH plus `cudaStreamSynchronize` (both in the trace), and then
~300-460 us of Python building metadata objects that the FULL branch of
`propose` never reads: `run_fullgraph(batch_desc)` replays the captured
graph, which reads the GPU buffers `_prepare_dflash_inputs_kernel` and the
runner's block-table gathers already wrote. The metadata is only consumed
by the eager (`_generate_draft`) branch.

The sync itself is free (the GPU was busy); the Python after it is not --
it runs while the GPU has nothing queued, and rank 3's slower host then
holds every rank at the drafter's first all-reduce (245 us on rank 0).

This module wraps `DFlashSpeculator._build_draft_attn_metadata`:

  time    stock build every step, host wall time logged (median/p90 us)
          every 256 calls -- the un-profiled number for the ledger
  shadow  stock build served; alongside, the cached dict of the same shape
          is compared field by field (GPU tensors: identity/shape/dtype;
          host scalars: equality, the expected per-step ones listed apart)
          and drift is logged every 256 calls
  1       when the drafter's own dispatch says this batch is a FULL replay,
          the dict built once for this shape is returned and the stock
          build is skipped; anything else (eager, padded, profile run,
          DP > 1, a query_start_loc override) runs the stock build

The knob is exact-match; unknown values are off. The installer pins the
image files whose internals this relies on (DISARM on drift), and any
exception inside the wrapper falls back to the stock build for the rest
of the boot, loudly. Numerics: none touched -- the same graph replays the
same buffers; only host work is removed. Speed only: bracket on C=1
step/s with prefill alongside. EXP-24.
"""
from __future__ import annotations

import hashlib
import os
import statistics
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_GLM53_DRAFTER_PREP"
MODES = ("time", "shadow", "1")

# The image files whose contracts the wrapper relies on: the DFlash
# speculator (propose's FULL branch ignores the metadata; the dispatch call
# shape), its base (the _build_draft_attn_metadata signature) and the
# cudagraph manager (dispatch signature / FULL descriptor). Any edit ->
# DISARM, like glm53_prep_fused.
PREIMAGES: dict[str, str] = {
    "v1/worker/gpu/spec_decode/dflash/speculator.py":
        "bd7f4c63d1196cb53bee0a81339aa5651e36938fc38b73c4ce89d978e0176a87",
    "v1/worker/gpu/spec_decode/speculator.py":
        "325dca3bb1aee7f3ce237f913e301f3008d765525ebf650ea4b88f195692471e",
    "v1/worker/gpu/cudagraph_utils.py":
        "c183937e6eb5b9c28c79d98fb4c64f562e7649d5f6d65743e6640b2f378ecf9f",
}

# host scalars a FULL replay never reads and that legitimately change per
# step; every other difference between the cached and the stock dict counts
# as drift in shadow mode
PER_STEP_SCALARS = ("max_seq_len", "seq_lens_cpu_upper_bound", "_seq_lens_cpu",
                    "seq_lens_cpu", "max_query_len", "num_actual_tokens")

_INSTALLED = False
_ORIG_BUILD = None
_DISABLED = False
_MODE = "0"
_CACHE: dict = {}
_TIMES: list = []
_STATS = {"calls": 0, "full": 0, "served": 0, "stock": 0, "drift": 0, "scalar": 0}
_ANNOUNCED: set = set()
LOG_EVERY = 256
TALLY_EVERY = 1024
# mode 1 audits itself: every SELFCHECK_EVERY served replays the stock build
# runs once more and is compared against the cached dict (the shadow check,
# inside the armed boot). Drift beyond the per-step scalars DISARMs the boot,
# loudly. 0 disables the audit.
ENV_SELFCHECK_EVERY = "VLLM_GLM53_DRAFTER_PREP_SELFCHECK_EVERY"


def selfcheck_every() -> int:
    raw = (os.environ.get(ENV_SELFCHECK_EVERY) or "256").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 256


def drafter_prep_mode() -> str:
    raw = (os.environ.get(ENV) or "0").strip()
    return raw if raw in MODES else "0"


def check_preimages(root: str) -> list[str]:
    bad = []
    for rel, want in PREIMAGES.items():
        try:
            with open(os.path.join(root, rel), "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            got = "absent"
        if got != want:
            bad.append(f"{rel}: {got[:12]} != {want[:12]}")
    return bad


def _is_full_replay(spec, num_reqs: int, num_tokens_padded: int) -> bool:
    """Replicates propose()'s own dispatch: FULL iff the drafter's cudagraph
    manager maps (num_reqs, num_reqs x num_query_per_req) onto a captured
    FULL descriptor of exactly num_tokens_padded tokens. Profile runs (need_
    eager) happen before capture, so the manager answers NONE for them."""
    from vllm.config.compilation import CUDAGraphMode

    mgr = getattr(spec, "query_cudagraph_manager", None)
    if mgr is None or int(getattr(spec, "dp_size", 1)) != 1:
        return False
    per_req = int(spec.num_query_per_req)
    desc = mgr.dispatch(num_reqs, num_reqs * per_req, per_req, num_active_loras=0)
    return (desc.cg_mode == CUDAGraphMode.FULL
            and int(desc.num_tokens) == int(num_tokens_padded))


def _key(num_reqs, num_reqs_padded, num_tokens_padded, step, causal):
    if isinstance(causal, dict):
        c = tuple(sorted((int(k), bool(v)) for k, v in causal.items()))
    else:
        c = bool(causal)
    return (int(num_reqs), int(num_reqs_padded), int(num_tokens_padded), int(step), c)


def _fields(obj):
    if isinstance(obj, dict):
        return obj.items()
    d = getattr(obj, "__dict__", None)
    if d is not None:
        return d.items()
    return None


def compare(cached, fresh, path: str = "") -> tuple[int, int]:
    """(drift, scalar_diff): drift = any GPU tensor identity / shape / dtype
    difference or a structural difference; scalar_diff = a host value that
    differs under a PER_STEP_SCALARS name (expected, unread by replay)."""
    import torch

    if isinstance(cached, torch.Tensor) or isinstance(fresh, torch.Tensor):
        if not (isinstance(cached, torch.Tensor) and isinstance(fresh, torch.Tensor)):
            return 1, 0
        if cached.is_cuda or fresh.is_cuda:
            same = (cached.data_ptr() == fresh.data_ptr() and cached.shape == fresh.shape
                    and cached.dtype == fresh.dtype and cached.stride() == fresh.stride())
            return (0 if same else 1), 0
        if cached.shape != fresh.shape or cached.dtype != fresh.dtype:
            return 1, 0
        if torch.equal(cached, fresh):
            return 0, 0
        return (0, 1) if path.split(".")[-1] in PER_STEP_SCALARS else (1, 0)
    fc, ff = _fields(cached), _fields(fresh)
    if fc is not None and ff is not None:
        fc, ff = dict(fc), dict(ff)
        if set(fc) != set(ff):
            return 1, 0
        drift = scalar = 0
        for k in fc:
            d, s = compare(fc[k], ff[k], f"{path}.{k}" if path else str(k))
            drift += d
            scalar += s
        return drift, scalar
    if isinstance(cached, (list, tuple)) and isinstance(fresh, (list, tuple)):
        if len(cached) != len(fresh):
            return 1, 0
        drift = scalar = 0
        for i, (a, b) in enumerate(zip(cached, fresh)):
            d, s = compare(a, b, f"{path}[{i}]")
            drift += d
            scalar += s
        return drift, scalar
    try:
        equal = bool(cached == fresh)
    except Exception:
        equal = cached is fresh
    if equal:
        return 0, 0
    return (0, 1) if path.split(".")[-1] in PER_STEP_SCALARS else (1, 0)


def _say(key: str, msg: str, *args) -> None:
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        logger.warning(msg, *args)


def _patched_build(self, num_reqs, num_reqs_padded, num_tokens_padded,
                   seq_lens_cpu_upper_bound, step, num_query_per_req=None,
                   causal=False, query_start_loc_np=None):
    global _DISABLED
    args = (num_reqs, num_reqs_padded, num_tokens_padded, seq_lens_cpu_upper_bound, step)
    kw = dict(num_query_per_req=num_query_per_req, causal=causal,
              query_start_loc_np=query_start_loc_np)
    if _DISABLED or _MODE == "0":
        return _ORIG_BUILD(self, *args, **kw)
    try:
        _STATS["calls"] += 1
        if _MODE == "time":
            t0 = time.perf_counter()
            md = _ORIG_BUILD(self, *args, **kw)
            _TIMES.append((time.perf_counter() - t0) * 1e6)
            if len(_TIMES) % LOG_EVERY == 0:
                s = sorted(_TIMES[-LOG_EVERY:])
                logger.warning("[drafter-prep] stock build host us: n=%d median=%.0f p90=%.0f "
                               "max=%.0f (the time mode 1 removes on FULL replays)",
                               len(_TIMES), statistics.median(s), s[int(len(s) * 0.9)], s[-1])
            return md
        full = (query_start_loc_np is None
                and _is_full_replay(self, num_reqs, num_tokens_padded))
        if not full:
            _STATS["stock"] += 1
            return _ORIG_BUILD(self, *args, **kw)
        _STATS["full"] += 1
        k = _key(num_reqs, num_reqs_padded, num_tokens_padded, step, causal)
        if _MODE == "shadow":
            t0 = time.perf_counter()
            md = _ORIG_BUILD(self, *args, **kw)
            _TIMES.append((time.perf_counter() - t0) * 1e6)
            cached = _CACHE.get(k)
            if cached is None:
                _CACHE[k] = md
            else:
                d, s = compare(cached, md)
                _STATS["drift"] += int(d > 0)
                _STATS["scalar"] += int(s > 0)
            if _STATS["full"] % LOG_EVERY == 0:
                t = sorted(_TIMES[-LOG_EVERY:])
                logger.warning("[drafter-prep] shadow: full=%d drift=%d (per-step scalars differ "
                               "in %d) stock=%d keys=%d; stock build host us median=%.0f p90=%.0f",
                               _STATS["full"], _STATS["drift"], _STATS["scalar"], _STATS["stock"],
                               len(_CACHE), statistics.median(t), t[int(len(t) * 0.9)])
            return md
        # _MODE == "1"
        md = _CACHE.get(k)
        if md is None:
            md = _ORIG_BUILD(self, *args, **kw)
            _CACHE[k] = md
            _say(("first", k), "[drafter-prep] serving: FULL replay -> draft metadata cached "
                 "for key=%s (num_reqs=%d, tokens=%d, step=%d); the stock build is skipped "
                 "from the next step", k, num_reqs, num_tokens_padded, step)
            return md
        _STATS["served"] += 1
        every = selfcheck_every()
        if every and _STATS["served"] % every == 0:
            # the armed boot's own shadow: rebuild once, compare, DISARM on drift
            fresh = _ORIG_BUILD(self, *args, **kw)
            d, sc = compare(md, fresh)
            _STATS["scalar"] += int(sc > 0)
            if d:
                _DISABLED = True
                _STATS["drift"] += 1
                logger.warning("[drafter-prep] selfcheck DRIFT at served=%d key=%s (%d fields) -> "
                               "DISARM: stock build for the rest of the boot", _STATS["served"], k, d)
                return fresh
        if _STATS["served"] % TALLY_EVERY == 0:
            logger.warning("[drafter-prep] tally: served=%d (build skipped) stock=%d keys=%d "
                           "selfchecks clean (every %d)", _STATS["served"], _STATS["stock"],
                           len(_CACHE), every)
        return md
    except Exception:
        _DISABLED = True
        logger.exception("[drafter-prep] wrapper failed -> stock build for the rest of the boot")
        return _ORIG_BUILD(self, *args, **kw)


def install_glm53_drafter_prep() -> bool:
    """Wrap DFlashSpeculator._build_draft_attn_metadata once. Safe to call
    from the model import; inert unless the knob names a mode."""
    global _INSTALLED, _ORIG_BUILD, _MODE
    mode = drafter_prep_mode()
    if _INSTALLED or mode == "0":
        return _INSTALLED
    import vllm

    root = os.path.dirname(os.path.abspath(vllm.__file__))
    bad = check_preimages(root)
    if bad:
        logger.warning("[drafter-prep] preimage drift -> DISARM (stock build): %s", bad)
        return False
    try:
        import vllm.v1.worker.gpu.spec_decode.dflash.speculator as ds
    except Exception:
        logger.exception("[drafter-prep] speculator module not importable -> DISARM")
        return False
    Spec = ds.DFlashSpeculator
    _ORIG_BUILD = Spec._build_draft_attn_metadata
    Spec._build_draft_attn_metadata = _patched_build
    _MODE = mode
    _INSTALLED = True
    logger.warning("[drafter-prep] installed mode=%s (preimages %d ok): the draft attention "
                   "metadata build is %s on FULL replays", mode, len(PREIMAGES),
                   {"time": "timed", "shadow": "shadowed", "1": "skipped (cached per shape)"}[mode])
    return True
