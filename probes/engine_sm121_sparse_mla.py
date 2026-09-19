"""FlashInfer's SM120 sparse MLA, the seed image's, judged on one GB10 at a geometry the engine has no lane for (probe,
single-GPU lane).

    U6  engine/SM121_INTAKE.md: flashinfer 0.6.18.dev20260819 in the image carries sparse-MLA kernels for SM120
        (mla/_sparse_mla_sm120.py, data/csrc/sparse_mla_sm120{,_decode_dsv3_2,_decode_dsv4,_prefill}.cu --
        measurements/sm121_inventory_20260919). The engine's sparse MLA is a megakernel built for GLM-5.3's 16 heads x
        512 latent; DeepSeek-V3.2 -- 128 heads over TP=4 = 32 a rank, keys kv_lora_rank 512 + rope 64 = 576 wide,
        values the 512 latent, index_topk 2048 -- has no lane. Here the image's kernel, through the public entry a lane
        would bind (flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla, backend "sparse"), reads the packed FP8 cache
        it defines (656 bytes a position: 512 e4m3 in four 128-wide tiles, their power-of-two fp32 scales, 64 bf16
        rope; pages of 64) and is held to engine/modules/sparse_attention.mla_sparse_mqa on that cache dequantized:
          decode   batch 1 and 4 over 40,000 cached positions, -1 slots scattered through the rows and one row's last
                   three 64-slot splits empty: rel error, finite, first-call seconds (JIT), median us
          prefill  256 tokens, which the entry routes to its prefill orchestrator (> 64 tokens): the same numbers
          load     vllm#54929 reports this kernel livelocking under sustained load (GPU at 100 %, no output until a
                   watchdog kills the server): the batch-4 decode back to back for 60 s or 20,000 calls, synchronized
                   every 100 calls; a sync gap over 10 s, or no return within 120 s, is `stalled` with the calls reached

    python3 probes/engine_kernel_check.py --lanes sm121_sparse_mla --output /cache/sm121-sparse-mla.json

Numbers, not a verdict: binding the kernel is a pull request that says so (U6's Triton lane is the alternative). Every
GPU stage runs in a daemon thread under a deadline; one that does not return is recorded as stalled, the report is
written, and the process leaves with os._exit(0) -- a hung CUDA context cannot be torn down. An arm that fails records
its error and the others still run.
"""
from __future__ import annotations

import functools
import inspect
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEADS, D_QK, D_V, ROPE = 32, 576, 512, 64       # DeepSeek-V3.2 at TP=4: 128 / 4 heads, 512 + 64 keys, 512 values
QK_NOPE = 128                                   # the model's per-head nope width (the sparse backend ignores it)
TOPK = 2048                                     # index_topk
SCALE = 192 ** -0.5 * (0.1 * math.log(40) + 1) ** 2   # softmax_scale: qk_head_dim 128 + 64, YaRN factor 40, mscale 1
POSITIONS, PAGE = 40_000, 64                    # 625 pages; the decode kernels dispatch only on pages of 64
BYTES = 656                                     # a position: 512 e4m3, 4 fp32 scales, 64 bf16 rope
TILE = 128                                      # one scale per 128 latent values
DECODE_BATCHES = (1, 4)
PREFILL_TOKENS = 256
PADDING = (300, 0, 777, 1500)                   # -1 slots scattered through row r, by r % 4
EMPTY_TAIL = 192                                # the last row's final three 64-slot splits are all -1
REFERENCE_CHUNK = 32                            # oracle tokens a step: [32, 2048, 576] fp32 is 151 MB
WORKSPACE_BYTES = 16 << 20                      # the decode split-K scratch at batch 4 is 4.2 MB of it
MEMORY_CAP_GIB = 4                              # the packed cache is 26 MB, dequantized 46 MB
LOAD_SECONDS, LOAD_CALLS, LOAD_SYNC_EVERY = 60, 20_000, 100
STALL_GAP_S, LOAD_DEADLINE_S = 10, 120
BUILD_DEADLINE_S, STAGE_DEADLINE_S = 1800, 300  # nvcc builds the whole sparse_mla_sm120 module on first use

ENTRY = "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla"
ENTRY_ARGS = ("query", "kv_cache", "workspace_buffer", "qk_nope_head_dim", "kv_lora_rank", "qk_rope_head_dim",
              "block_tables", "seq_lens", "max_seq_len", "sparse_mla_top_k", "out", "bmm1_scale", "bmm2_scale",
              "backend")


# -- the cache the kernel reads, and the oracle ----------------------------------------------------------------------
def pack_cache(kv):
    """[S, 576] bf16 -> the DSv3.2 FP8 cache [S // 64, 64, 656] uint8 the kernel reads (FlashInfer's
    sparse_mla_sm120/model/kv_cache_traits.cuh, KVCacheTraits<DSV3_2>): per position the latent's 512 values as e4m3
    in four 128-wide tiles, the tiles' four fp32 scales -- powers of two, FlashMLA's amax / 448 rounded up, which
    kv_scale_format "auto" reads -- then the 64 rope values as bf16."""
    import torch
    s = kv.shape[0]
    nope = kv[:, :D_V].float().reshape(s, D_V // TILE, TILE)
    scale = torch.pow(2.0, (nope.abs().amax(-1).clamp_min(1e-4) / 448.0).clamp_min(1e-4).log2().ceil())
    fp8 = (nope / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    packed = torch.empty(s, BYTES, dtype=torch.uint8, device=kv.device)
    packed[:, :D_V] = fp8.reshape(s, D_V).view(torch.uint8)
    packed[:, D_V:D_V + 16] = scale.contiguous().view(torch.uint8)
    packed[:, D_V + 16:] = kv[:, D_V:].contiguous().view(torch.uint8)
    return packed.reshape(s // PAGE, PAGE, BYTES)


def unpack_cache(packed):
    """The cache as the kernel reads it, from its bytes: [S, 576] bf16. Exact -- an e4m3 value times a power of two
    and a bf16 rope value are both bf16."""
    import torch
    rows = packed.reshape(-1, BYTES)
    s = rows.shape[0]
    fp8 = rows[:, :D_V].contiguous().view(torch.float8_e4m3fn).float().reshape(s, D_V // TILE, TILE)
    scale = rows[:, D_V:D_V + 16].contiguous().view(torch.float32)                  # [S, 4]
    rope = rows[:, D_V + 16:].contiguous().view(torch.bfloat16).float()             # [S, 64]
    return torch.cat([(fp8 * scale[..., None]).reshape(s, D_V), rope], dim=1).to(torch.bfloat16)


def make_slots(rows, generator):
    """[rows, 2048] int32 on the CPU: TOPK distinct positions of the pool per row, then -1 in PADDING[r % 4] scattered
    slots, and the last row's final EMPTY_TAIL slots -1 (whole 64-slot splits with nothing in them). Positions are flat,
    page * 64 + offset, as the kernel addresses them (sparse_mla_sm120/common/kv_cache_io.cuh: kv_ptr + idx * 656)."""
    import torch
    slots = torch.empty(rows, TOPK, dtype=torch.int32)
    for r in range(rows):
        row = torch.randperm(POSITIONS, generator=generator)[:TOPK]
        row[torch.randperm(TOPK, generator=generator)[:PADDING[r % len(PADDING)]]] = -1
        slots[r] = row
    slots[-1, -EMPTY_TAIL:] = -1
    return slots


def reference(q, cache, slots):
    """engine/modules/sparse_attention.mla_sparse_mqa on the dequantized cache, fp32: every head attends the same
    selected rows, keys 576 wide, and the value is the key row's first 512 (the latent), so the oracle's 576-wide
    context is cut to 512. No sink. The kernel skips -1 wherever it sits; the oracle counts a valid prefix -- softmax
    over a set does not see order, so each row's valid slots are moved first (stable) and counted."""
    import torch
    from engine.modules.sparse_attention import mla_sparse_mqa
    padding = slots < 0
    compact = torch.gather(slots, 1, torch.sort(padding.int(), dim=1, stable=True).indices)
    valid = (~padding).sum(1).to(torch.int32)
    q = q.float()
    return torch.cat([mla_sparse_mqa(q[i:i + REFERENCE_CHUNK], cache, compact[i:i + REFERENCE_CHUNK],
                                     valid[i:i + REFERENCE_CHUNK], SCALE)[..., :D_V]
                      for i in range(0, q.shape[0], REFERENCE_CHUNK)])


def entry_args(q, packed, slots, out, workspace) -> dict:
    """The public entry's keywords (flashinfer/mla/_core.py trtllm_batch_decode_with_kv_cache_mla, the SM12x "sparse"
    backend): query [T, 1, H, 576] bf16, the packed uint8 cache [pages, 64, 656], block_tables the sparse slots
    [T, 1, K] int32, seq_lens None (every column active; -1 is skipped), bmm1_scale the softmax scale, bmm2_scale 1,
    out [T, 1, H, 512] bf16."""
    return dict(query=q.unsqueeze(1), kv_cache=packed, workspace_buffer=workspace, qk_nope_head_dim=QK_NOPE,
                kv_lora_rank=D_V, qk_rope_head_dim=ROPE, block_tables=slots.unsqueeze(1), seq_lens=None,
                max_seq_len=TOPK, sparse_mla_top_k=TOPK, out=out.unsqueeze(1), bmm1_scale=SCALE, bmm2_scale=1.0,
                backend="sparse")


def _dispatch(sm120, tokens) -> str:
    """Which kernel the entry picks, read from the module's own tables (_DECODE_MAX_TOKENS, _DECODE_DSV3_2_DISPATCH)."""
    limit = getattr(sm120, "_DECODE_MAX_TOKENS", None)
    table = getattr(sm120, "_DECODE_DSV3_2_DISPATCH", None)
    if limit is None or table is None:
        return "unknown: the module has no dispatch tables"
    if tokens > limit:
        return "prefill orchestrator"
    return "decode_dsv3_2" if (HEADS, TOPK) in table else "none: the entry raises for this shape"


# -- stages under a deadline -----------------------------------------------------------------------------------------
def _abandon(report, output):
    """A stage that did not return: write what is known and leave without tearing the CUDA context down."""
    from probes.engine_sm121_candidates import _write
    print(_write(output, report), flush=True)
    sys.stderr.flush()
    os._exit(0)


def _guarded(report, output, stage, fn, deadline):
    """fn() in a daemon thread. Its exception is raised here; if it has not returned by the deadline the GPU is taken
    as hung -- the stage is recorded as stalled, the report written, and the process leaves."""
    box = {}

    def body():
        import torch
        try:
            with torch.inference_mode():
                box["value"] = fn()
        except BaseException as exc:                                    # noqa: BLE001 -- raised on the main thread
            box["error"] = exc

    thread = threading.Thread(target=body, name=f"sm121-sparse-mla {stage}", daemon=True)
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        report["stalled"][stage] = f"no return after {deadline} s"
        _abandon(report, output)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_load(report, output, call, out, want) -> dict:
    """vllm#54929's livelock, looked for: `call` back to back in a daemon thread for LOAD_SECONDS or LOAD_CALLS, a
    synchronize every LOAD_SYNC_EVERY calls and its time kept. The main thread waits LOAD_DEADLINE_S at most."""
    import torch
    from probes.engine_sm121_candidates import _error
    state = {"launched": 0, "completed": 0, "syncs": [], "error": None}
    torch.cuda.synchronize()
    began = time.perf_counter()

    def body():
        try:
            with torch.inference_mode():
                for i in range(1, LOAD_CALLS + 1):
                    call()
                    state["launched"] = i
                    if i % LOAD_SYNC_EVERY == 0:
                        torch.cuda.synchronize()
                        state["completed"] = i
                        state["syncs"].append(time.perf_counter() - began)
                        if state["syncs"][-1] >= LOAD_SECONDS:
                            break
        except BaseException as exc:                                    # noqa: BLE001 -- the failure is the answer
            state["error"] = f"{type(exc).__name__}: {exc}"[:300]

    thread = threading.Thread(target=body, name="sm121-sparse-mla load", daemon=True)
    thread.start()
    thread.join(LOAD_DEADLINE_S)
    syncs = list(state["syncs"])
    gaps = [b - a for a, b in zip([0.0] + syncs, syncs)]
    result = {"calls_launched": state["launched"], "calls_completed": state["completed"],
              "seconds": round(syncs[-1], 2) if syncs else 0.0,
              "calls_per_s": round(state["completed"] / syncs[-1], 1) if syncs else None,
              "max_sync_gap_s": round(max(gaps), 3) if gaps else None,
              "median_sync_gap_ms": round(statistics.median(gaps) * 1000, 2) if gaps else None}
    if thread.is_alive():
        open_gap = time.perf_counter() - began - (syncs[-1] if syncs else 0.0)
        result["stalled"] = True
        result["why"] = (f"no return after {LOAD_DEADLINE_S} s: {state['completed']} calls completed, "
                         f"{state['launched']} launched, the last synchronize returned {open_gap:.1f} s before")
        report["load"] = result
        report["stalled"]["load"] = result["why"]
        _abandon(report, output)
    if state["error"]:
        report["unavailable"]["load"] = state["error"]
    long_gaps = [round(g, 2) for g in gaps if g > STALL_GAP_S]
    result["stalled"] = bool(long_gaps)
    if long_gaps:
        result["why"] = f"{len(long_gaps)} sync gaps over {STALL_GAP_S} s (first {long_gaps[:5]})"
        report["stalled"]["load"] = result["why"]
    if not state["error"]:
        result["after_load_rel_error"] = _error(out, want)
        result["after_load_finite"] = bool(torch.isfinite(out).all())
    return result


# -- U6: sparse MLA at DeepSeek-V3.2's rank ----------------------------------------------------------------------------
def run(output=None) -> dict:
    import torch
    from probes.engine_sm121_candidates import _device, _error, _time, _write
    report = {"lane": "sm121_sparse_mla", "entry": {"name": ENTRY, "backend": "sparse"},
              "geometry": dict(heads=HEADS, d_qk=D_QK, d_v=D_V, topk=TOPK, positions=POSITIONS, page=PAGE,
                               bytes_per_position=BYTES, sm_scale=round(SCALE, 6),
                               kv="e4m3 latent, fp32 power-of-two scale per 128, bf16 rope (kv_scale_format auto)"),
              "unavailable": {}, "unsupported": {}, "stalled": {}, "cases": {}}
    _device(report, MEMORY_CAP_GIB)
    entry = sm120 = None
    try:
        import flashinfer
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
        report["flashinfer"] = getattr(flashinfer, "__version__", None)
        signature = inspect.signature(trtllm_batch_decode_with_kv_cache_mla)
        report["entry"]["signature"] = str(signature)[:1500]
        missing = [a for a in ENTRY_ARGS if a not in signature.parameters]
        if missing:
            report["unsupported"]["entry"] = f"{ENTRY} takes no {missing}"
        else:
            entry = trtllm_batch_decode_with_kv_cache_mla
    except Exception as exc:                                        # noqa: BLE001 -- the import's failure is the answer
        report["unavailable"]["flashinfer"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        import flashinfer.mla._sparse_mla_sm120 as sm120
    except Exception as exc:                                            # noqa: BLE001
        report["unavailable"]["_sparse_mla_sm120"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        from flashinfer.jit import mla as jit_mla
        flags = jit_mla.current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
        report["nvcc_arch_flags"] = [f for f in flags if "arch" in f or "sm_" in f][:8]
    except Exception as exc:                                            # noqa: BLE001
        report["unavailable"]["nvcc flags"] = f"{type(exc).__name__}: {exc}"[:300]
    if sm120 is not None and hasattr(sm120, "get_sparse_mla_sm120_module"):
        try:
            began = time.perf_counter()
            _guarded(report, output, "build", sm120.get_sparse_mla_sm120_module, BUILD_DEADLINE_S)
            report["build_s"] = round(time.perf_counter() - began, 2)
        except Exception as exc:                                        # noqa: BLE001
            report["unavailable"]["build"] = f"{type(exc).__name__}: {exc}"[:300]
    with torch.inference_mode():
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(0)
        kv = torch.randn(POSITIONS, D_QK, device="cuda").to(torch.bfloat16)
        packed = pack_cache(kv)
        cache = unpack_cache(packed)
        report["cache"] = dict(pages=POSITIONS // PAGE, packed_MB=round(packed.numel() / 1e6, 1),
                               quantization_vs_bf16=_error(cache, kv))
        del kv
        workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
        load = None
        cases = [(f"decode b{b}", b) for b in DECODE_BATCHES] + [(f"prefill {PREFILL_TOKENS}", PREFILL_TOKENS)]
        for name, tokens in cases:
            q = torch.randn(tokens, HEADS, D_QK, device="cuda").to(torch.bfloat16)
            slots = make_slots(tokens, generator).cuda()
            valid = (slots >= 0).sum(1)
            row = {"tokens": tokens, "valid_slots": [int(valid.min()), int(valid.max())],
                   "dispatch": _dispatch(sm120, tokens) if sm120 is not None else None}
            want = reference(q, cache, slots)
            if entry is not None:
                out = torch.zeros(tokens, HEADS, D_V, dtype=torch.bfloat16, device="cuda")
                call = functools.partial(entry, **entry_args(q, packed, slots, out, workspace))
                try:
                    began = time.perf_counter()
                    _guarded(report, output, f"{name} first call", lambda: (call(), torch.cuda.synchronize()),
                             STAGE_DEADLINE_S)
                    row["first_call_s"] = round(time.perf_counter() - began, 2)
                    row["finite"] = bool(torch.isfinite(out).all())
                    row["rel_error"] = _error(out, want)
                    row.update(_guarded(report, output, f"{name} timing", lambda: _time(call), STAGE_DEADLINE_S))
                    if tokens == max(DECODE_BATCHES):
                        load = (call, out, want)
                except Exception as exc:                                # noqa: BLE001
                    report["unavailable"][name] = f"{type(exc).__name__}: {exc}"[:300]
            report["cases"][name] = row
            print(json.dumps({name: row}), flush=True)
        if load is not None:
            report["load"] = run_load(report, output, *load)
            print(json.dumps({"load": report["load"]}), flush=True)
    print(_write(output, report), flush=True)
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
