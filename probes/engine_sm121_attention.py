"""GQA attention for engine/SM121_INTAKE.md U7 (paged KV with a sliding window, attention sinks and a logit soft cap:
the gpt-oss, Gemma, Mistral and Qwen3 families) and U8 (FP8 and NVFP4 KV cache), on the seed image's FlashInfer
0.6.18.dev20260819 -- built without sm_121 in torch's arch list, so its kernels are JIT-compiled here
(flashinfer/compilation_context.py maps capability 12.1 to compute_121a) -- against a plain fp32 torch reference (probe,
single-GPU lane).

Per rank of a TP=4 model, 8 query / 2 KV heads at head_dim 128 and gpt-oss's 16 / 2 at 64; bf16 q over an NHD cache of
16-token pages handed out in shuffled order, every slot no request owns holding GARBAGE:

    decode   1 request at KV 1,000 and at 33,000, and 4 requests mixing the two; variants causal, window127 (window_left
             127: the query and the 127 before it, HF sliding_window 128), sinks (a per-head fp32 logit in the softmax
             denominator, as gpt-oss), window127+sinks (gpt-oss's sliding layers), softcap50 (50 tanh(s / 50) on the
             scaled logits, q scaled up so they reach the cap), fp8_kv (e4m3, one scale per tensor), nvfp4_kv
             (FlashInfer's fp4_quantize: 16-element e4m3 block scales in the linear layout, one global scale per tensor)
    prefill  2,048 new tokens over a 2,048-token prefix already in the cache: causal, window127, sinks,
             window127+sinks

Routes: cuda-core and tensor-core are BatchDecodeWithPagedKVCacheWrapper with use_tensor_cores false (plan() keeps
backend "auto" and builds the CUDA-core decode kernel) and true (plan() resolves "auto" to fa2); "tensor-core no-split"
adds disable_split_kv, which prefill.py forces for NVFP4 KV (split-KV corrupted it when a short query reads a long KV)
and decode.py does not; auto is BatchPrefillWithPagedKVCacheWrapper's default; sink-jit is the AttentionSink JIT variant
through that wrapper, as flashinfer.BatchAttentionWithAttentionSinkWrapper builds it; xqa is
xqa_batch_decode_with_kv_cache (its own check: SM90/100/12x, NVFP4 KV on 12x only); cute-dsl runs last -- its decode
kernel is tcgen05/TMEM and sm_121a has neither (measurements/sm121a_architecture_20260911), so its refusal is the
record. At f0922749 the fa2 and CUDA-core paths accept sinks= in run() and never hand it to the kernel (prefill.py's
paged_run, decode.py's run), so a row with a feature carries vs_ref_without_<feature> beside vs_ref, and the case's
"effects" says how far that feature moves the reference.

Per arm: vs_ref (max |got - ref| / max |ref|, the reference reading the dequantized cache the kernel read), finite,
first_call_s (construct + plan + first call, JIT compile included), median_us / best_us over 9 calls after 3 warmups,
and what FlashInfer picked; quantized arms add vs_bf16_kv_ref. A keyword the installed API lacks (inspect.signature)
files the arm under "unsupported" rather than guessing; an arm that raises records its error and the rest still run,
unless the CUDA context is lost, which stops the lane under "aborted". The JSON is rewritten after every arm.

    python3 probes/engine_kernel_check.py --lanes sm121_attention --output /cache/sm121-attention.json

Numbers, not a verdict: a kernel that wins here is bound by a pull request that says so.
"""
from __future__ import annotations

import functools
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.engine_sm121_candidates import REPEATS, WARMUP, _device, _error, _time, _write  # noqa: E402

GEOMETRIES = {"8q/2kv d128": (8, 2, 128), "gpt-oss 16q/2kv d64": (16, 2, 64)}   # (query heads, KV heads, head dim)
DECODE_CASES = {"b1 kv1000": (1000,), "b1 kv33000": (33000,), "b4 kv1000/33000": (1000, 33000, 1000, 33000)}
PREFIX, NEW = 2048, 2048
PAGE = 16
SPARE_PAGES = 16
GARBAGE = 4.0                        # K holds +GARBAGE and V -GARBAGE where no request owns the slot
SINK_RANGE = (-2.0, 8.0)             # sink logits: linspace over the query heads
SOFTCAP_GAIN = 40.0                  # q scale for softcap50: logits ~ N(0, 40), so 50 tanh(s / 50) bends them
FP8_MAX, FP4_MAX = 448.0, 6.0
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
REF_ROWS = 256                       # query rows per reference chunk
SEED = 0
MEMORY_CAP_GIB = 6
WORKSPACE_MIB = {"decode": 128, "prefill": 128, "xqa": 256}

DECODE_VARIANTS = {
    "causal": dict(routes=("cuda-core", "tensor-core", "xqa")),
    "window127": dict(window=127, routes=("cuda-core", "tensor-core", "xqa")),
    "sinks": dict(sinks=True, routes=("cuda-core", "tensor-core", "sink-jit", "xqa")),
    "window127+sinks": dict(window=127, sinks=True, routes=("cuda-core", "tensor-core", "sink-jit", "xqa")),
    "softcap50": dict(softcap=50.0, routes=("cuda-core", "tensor-core", "xqa")),
    "fp8_kv": dict(kv="fp8", routes=("cuda-core", "tensor-core", "xqa")),
    "nvfp4_kv": dict(kv="nvfp4", routes=("tensor-core", "tensor-core no-split", "xqa")),
}
PREFILL_VARIANTS = {
    "causal": dict(routes=("auto",)),
    "window127": dict(window=127, routes=("auto",)),
    "sinks": dict(sinks=True, routes=("auto", "sink-jit")),
    "window127+sinks": dict(window=127, sinks=True, routes=("auto", "sink-jit")),
}
# run last, on the first decode case and the prefill; the paged cute-dsl prefill's run() refuses sinks=
CUTE_DSL = {"decode": ("causal", "window127+sinks"), "prefill": ("causal", "window127")}
DECODE_INIT = {"cuda-core": {}, "tensor-core": {"use_tensor_cores": True},
               "tensor-core no-split": {"use_tensor_cores": True}, "cute-dsl": {"backend": "cute-dsl"}}
DECODE_PLAN = {"tensor-core no-split": {"disable_split_kv": True}}
PREFILL_INIT = {"auto": {}, "cute-dsl": {"backend": "cute-dsl"}}
NOT_TRIED = {
    "trtllm-gen": "jit/attention/modules.py gen_trtllm_gen_fmha_module fetches its cubins and header with "
                  "get_artifact, a download when they are not cached -- not beside production",
}


# -- the oracle --------------------------------------------------------------------------------------------------------
def reference_attention(q, k, v, *, causal_offset, window=-1, sinks=None, softcap=None, scale):
    """fp32 attention of one request's queries over its contiguous K/V, the oracle every arm is held to.

    q [Tq, Hq, D]; k, v [Tk, Hkv, D]; query head h reads KV head h // (Hq / Hkv). Query row i sits at position
    causal_offset + i and sees key j when j <= causal_offset + i and, with window >= 0, j >= causal_offset + i - window
    (FlashInfer's window_left: the query and the `window` positions before it). softcap: s -> softcap * tanh(s /
    softcap) on the scaled logits before the softmax. sinks [Hq]: a per-head logit that joins each row's softmax
    denominator and carries no value (softmax over [scores, sink], sink column dropped). Returns [Tq, Hq, Dv] fp32."""
    import torch
    tq, hq, _ = q.shape
    tk, hkv, _ = k.shape
    group = hq // hkv
    kf = k.float().repeat_interleave(group, dim=1)
    vf = v.float().repeat_interleave(group, dim=1)
    cols = torch.arange(tk, device=q.device)
    out = torch.empty(tq, hq, v.shape[-1], dtype=torch.float32, device=q.device)
    for start in range(0, tq, REF_ROWS):
        qf = q[start:start + REF_ROWS].float()
        s = torch.einsum("qhd,khd->hqk", qf, kf) * scale
        if softcap:
            s = softcap * torch.tanh(s / softcap)
        pos = causal_offset + start + torch.arange(qf.shape[0], device=q.device)
        seen = cols[None, :] <= pos[:, None]
        if window is not None and window >= 0:
            seen &= cols[None, :] >= pos[:, None] - window
        s = s.masked_fill(~seen[None], float("-inf"))
        if sinks is not None:
            sink = sinks.float().to(q.device)[:, None, None].expand(hq, s.shape[1], 1)
            p = torch.softmax(torch.cat((s, sink), dim=-1), dim=-1)[..., :-1]
        else:
            p = torch.softmax(s, dim=-1)
        out[start:start + qf.shape[0]] = torch.einsum("hqk,khd->qhd", p, vf)
    return out


def case_reference(cache, pools, q, rows, feats, sinks, scale):
    """reference_attention for every request of a paged case, each over its tokens gathered from `pools` (K, V) through
    the case's slots; `rows` query rows per request, the last `rows` positions of its KV."""
    import torch
    k_pool, v_pool = pools
    out, at = [], 0
    for n, slot in zip(cache["lens"], cache["slots"]):
        k = k_pool.reshape(-1, *k_pool.shape[-2:])[slot]
        v = v_pool.reshape(-1, *v_pool.shape[-2:])[slot]
        out.append(reference_attention(q[at:at + rows], k, v, causal_offset=n - rows, window=feats["window"],
                                       sinks=sinks if feats["sinks"] else None, softcap=feats["softcap"],
                                       scale=scale))
        at += rows
    return torch.cat(out)


# -- the cache ---------------------------------------------------------------------------------------------------------
def paged_cache(lens, kv_heads, dim, device) -> dict:
    """Requests of the given KV lengths in one pool of 16-token pages, handed out from a random permutation (with spare
    pages left in it) so no request's pages are contiguous; K/V [pages, 16, kv_heads, dim] bf16 N(0, 1), GARBAGE in
    every slot no request owns. Returns the pools, each request's token slots into the flattened pool, FlashInfer's
    page table (indptr, indices, last_page_len) and the same pages as xqa's block table and seq_lens."""
    import torch
    counts = [-(-n // PAGE) for n in lens]
    total = sum(counts) + SPARE_PAGES
    order = torch.randperm(total).tolist()
    shape = (total, PAGE, kv_heads, dim)
    k = torch.full(shape, GARBAGE, dtype=torch.bfloat16, device=device)
    v = torch.full(shape, -GARBAGE, dtype=torch.bfloat16, device=device)
    slots, pages, at = [], [], 0
    for n, c in zip(lens, counts):
        ids = order[at:at + c]
        at += c
        page = torch.tensor(ids, dtype=torch.long, device=device)
        slot = (page[:, None] * PAGE + torch.arange(PAGE, device=device)).flatten()[:n]
        k.view(-1, kv_heads, dim)[slot] = torch.randn(n, kv_heads, dim, device=device).to(torch.bfloat16)
        v.view(-1, kv_heads, dim)[slot] = torch.randn(n, kv_heads, dim, device=device).to(torch.bfloat16)
        slots.append(slot)
        pages.append(ids)
    i32 = dict(dtype=torch.int32, device=device)
    table = torch.zeros(len(lens), max(counts), **i32)
    for b, ids in enumerate(pages):
        table[b, :len(ids)] = torch.tensor(ids, **i32)
    return dict(k=k, v=v, lens=list(lens), slots=slots,
                indptr=torch.tensor([0, *itertools.accumulate(counts)], **i32),
                indices=torch.tensor([p for ids in pages for p in ids], **i32),
                last_page_len=torch.tensor([n - PAGE * (c - 1) for n, c in zip(lens, counts)], **i32),
                block_tables=table, seq_lens=torch.tensor(list(lens), **i32))


def fp8_kv(cache):
    """e4m3 K and V, one scale per tensor (amax / 448): (what the kernel reads, the dequantized pools the reference
    reads, facts)."""
    import torch
    kernel, ref = {}, []
    for name in ("k", "v"):
        x = cache[name].float()
        scale = float(x.abs().amax()) / FP8_MAX
        q8 = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        kernel[name], kernel[f"{name}_scale"] = q8, scale
        ref.append(q8.float() * scale)
    return kernel, tuple(ref), {"k_scale": kernel["k_scale"], "v_scale": kernel["v_scale"]}


def nvfp4_dequant(packed, sf, scale, low_first=True):
    """fp32 values of packed e2m1 pairs [..., D/2] (uint8) with e4m3 block scales [..., D/16] (uint8) and a global
    dequant scale; with low_first the even element of each pair sits in the low nibble."""
    import torch
    magnitude = torch.tensor(E2M1, device=packed.device)

    def nibble(x):
        return magnitude[(x & 7).long()] * (1.0 - 2.0 * (x >> 3).float())

    lo, hi = nibble(packed & 15), nibble(packed >> 4)
    pairs = torch.stack((lo, hi) if low_first else (hi, lo), dim=-1).flatten(-2)
    return pairs * sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1) * scale


def nvfp4_kv(cache):
    """NVFP4 K and V through FlashInfer's fp4_quantize (16-element blocks, e4m3 block scales in the linear layout the
    fa2 and XQA paths read, global scale 448 * 6 / amax per tensor): (what the kernel reads -- packed pools, uint8
    block scales, global dequant scales amax / (448 * 6) --, the dequantized pools the reference reads, facts). Each
    tensor's round trip is recorded under both nibble orders and the reference decodes with the one that reproduces
    it."""
    import torch
    from flashinfer.fp4_quantization import fp4_quantize
    kernel, ref, facts = {}, [], {}
    for name in ("k", "v"):
        x = cache[name]
        d = x.shape[-1]
        amax = float(x.float().abs().amax())
        global_sf = torch.tensor([FP8_MAX * FP4_MAX / amax], dtype=torch.float32, device=x.device)
        packed, sf = fp4_quantize(x.reshape(-1, d).contiguous(), global_sf, sf_vec_size=16, sf_use_ue8m0=False,
                                  is_sf_swizzled_layout=False)
        packed = packed.view(torch.uint8).reshape(*x.shape[:-1], d // 2)
        sf = sf.view(torch.uint8).reshape(*x.shape[:-1], d // 16)
        scale = amax / (FP8_MAX * FP4_MAX)
        trips = {order: _error(nvfp4_dequant(packed, sf, scale, order == "low_first"), x)
                 for order in ("low_first", "high_first")}
        order = min(trips, key=trips.get)
        kernel[name], kernel[f"{name}_sf"], kernel[f"{name}_scale"] = packed, sf, scale
        ref.append(nvfp4_dequant(packed, sf, scale, order == "low_first"))
        facts[name] = {"global_scale": scale, "roundtrip": trips, "nibble_order": order}
    return kernel, tuple(ref), facts


# -- FlashInfer -------------------------------------------------------------------------------------------------------
def _keywords(fn):
    """Named parameters of fn, or None when its signature cannot be read or takes **kwargs it does not name (no keyword
    is then ruled out)."""
    import inspect
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params if p.kind is not p.VAR_POSITIONAL}


def api_keywords(report) -> dict:
    """The installed signatures of every call the arms make; report["api"] lists them."""
    apis = {}
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper as decode
        apis["decode.__init__"] = _keywords(decode.__init__)
        apis["decode.plan"] = _keywords(getattr(decode, "_plan_impl", decode.plan))   # plan() forwards **kwargs there
        apis["decode.run"] = _keywords(decode.run)
    except Exception as exc:                                                            # noqa: BLE001
        report["unavailable"]["flashinfer.decode"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper as prefill
        apis["prefill.__init__"] = _keywords(prefill.__init__)
        apis["prefill.plan"] = _keywords(prefill.plan)
        apis["prefill.run"] = _keywords(prefill.run)
    except Exception as exc:                                                            # noqa: BLE001
        report["unavailable"]["flashinfer.prefill"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        apis["xqa"] = _keywords(xqa_batch_decode_with_kv_cache)
    except Exception as exc:                                                            # noqa: BLE001
        report["unavailable"]["flashinfer xqa"] = f"{type(exc).__name__}: {exc}"[:300]
    report["api"] = {name: sorted(kw) if kw is not None else "unreadable" for name, kw in apis.items()}
    return apis


def flashinfer_facts() -> dict:
    import flashinfer
    facts = {"version": getattr(flashinfer, "__version__", None), "file": flashinfer.__file__,
             "FLASHINFER_CUDA_ARCH_LIST": os.environ.get("FLASHINFER_CUDA_ARCH_LIST"),
             "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST")}
    try:
        from flashinfer.compilation_context import CompilationContext
        archs = CompilationContext().TARGET_CUDA_ARCHS
        facts["jit_target_archs"] = sorted(f"{major}.{minor}" for major, minor in archs)
    except Exception as exc:                                                            # noqa: BLE001
        facts["jit_target_archs"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        from flashinfer.jit import env
        facts["jit_dir"] = str(env.FLASHINFER_JIT_DIR)
        facts["aot_jit_cache"] = env.has_flashinfer_jit_cache()
    except Exception as exc:                                                            # noqa: BLE001
        facts["jit_dir"] = f"{type(exc).__name__}: {exc}"[:300]
    return facts


def sink_jit_wrapper(workspace, head_dim, window):
    """BatchPrefillWithPagedKVCacheWrapper with the AttentionSink JIT variant, the arguments
    flashinfer.BatchAttentionWithAttentionSinkWrapper passes (attention/_core.py) with the head dim added to the module
    name: the wrapper's own name (batch_prefill_attention_sink_<dtype>_swa_<bool>_<backend>) leaves it out, and the JIT
    keys the module's generated sources by that name, so this probe's two head dims would share one. The sink and
    sm_scale go to run() positionally, as the wrapper's tests pass them."""
    import torch
    from flashinfer.jit.attention.variants import attention_sink_decl
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    uri = f"batch_prefill_attention_sink_bf16_hd{head_dim}_swa_{window >= 0}_fa2"
    jit_args = [uri, torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, head_dim, head_dim,
                ["sink"], ["float"], ["sm_scale"], ["double"], "AttentionSink", attention_sink_decl["fa2"]]
    jit_kwargs = {"use_sliding_window": window >= 0, "use_fp16_qk_reduction": False, "pos_encoding_mode": 0}
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2", jit_args=jit_args,
                                                  jit_kwargs=jit_kwargs)
    return wrapper, uri


def build_call(section, route, geometry, cache, kv, q, qo_indptr, feats, workspaces, sinks, scale):
    """Construct and plan one route over the paged cache (the JIT compile happens here or on the first call); returns
    (call, picked). kv is what the kernel reads: pools "k"/"v", for quantized caches the global dequant scales
    "k_scale"/"v_scale", for NVFP4 the block scales "k_sf"/"v_sf" (uint8)."""
    import torch
    hq, hkv, d = geometry
    pools, window, softcap = (kv["k"], kv["v"]), feats["window"], feats["softcap"]
    scales = {"k_scale": kv["k_scale"], "v_scale": kv["v_scale"]} if "k_scale" in kv else {}
    if route == "xqa":
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        extra = {"window_left": window} if window >= 0 else {}
        if sinks is not None:
            extra["sinks"] = sinks
        if "k_sf" in kv:
            extra["kv_cache_sf"] = (kv["k_sf"], kv["v_sf"])
        bmm1, bmm2 = scale * scales.get("k_scale", 1.0), scales.get("v_scale", 1.0)
        longest = max(cache["lens"])

        def call():
            return xqa_batch_decode_with_kv_cache(q, pools, workspaces["xqa"], cache["block_tables"], cache["seq_lens"],
                                                  longest, bmm1, bmm2, **extra)
        return call, {"backend": "xqa"}
    plan = {"q_data_type": q.dtype, "kv_data_type": kv["k"].dtype, "sm_scale": scale}
    if window >= 0:
        plan["window_left"] = window
    if softcap:
        plan["logits_soft_cap"] = softcap
    run = dict(scales)
    if "k_sf" in kv:
        run["kv_cache_sf"] = (kv["k_sf"].view(torch.float8_e4m3fn), kv["v_sf"].view(torch.float8_e4m3fn))
    if section == "decode" and route in DECODE_INIT:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspaces["decode"], "NHD", **DECODE_INIT[route])
        wrapper.plan(cache["indptr"], cache["indices"], cache["last_page_len"], hq, hkv, d, PAGE, **plan,
                     **DECODE_PLAN.get(route, {}))
        picked = {"backend": getattr(wrapper, "_backend", None),
                  "use_tensor_cores": getattr(wrapper, "use_tensor_cores", None)}
    else:
        if route == "sink-jit":
            wrapper, uri = sink_jit_wrapper(workspaces["prefill"], d, window)
        else:
            from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
            wrapper = BatchPrefillWithPagedKVCacheWrapper(workspaces["prefill"], "NHD", **PREFILL_INIT[route])
            uri = None
        wrapper.plan(qo_indptr, cache["indptr"], cache["indices"], cache["last_page_len"], hq, hkv, d, PAGE,
                     causal=True, **plan)
        picked = {"backend": getattr(wrapper, "_backend", None)}
        if route == "sink-jit":
            picked["jit_module"] = uri
            return (lambda: wrapper.run(q, pools, sinks, scale)), picked
    if sinks is not None:
        run["sinks"] = sinks
    return (lambda: wrapper.run(q, pools, **run)), picked


def _needs(section, route, feats) -> list:
    """(api, keyword) pairs the arm passes, held to the installed signatures before it runs."""
    if route == "xqa":
        out = [("xqa", "window_left")] if feats["window"] >= 0 else []
        if feats["sinks"]:
            out.append(("xqa", "sinks"))
        if feats["softcap"]:
            out.append(("xqa", "logits_soft_cap"))
        if feats["kv"] == "nvfp4":
            out.append(("xqa", "kv_cache_sf"))
        return out
    api = "decode" if section == "decode" and route in DECODE_INIT else "prefill"
    out = [(f"{api}.plan", kw) for kw in ("q_data_type", "kv_data_type", "sm_scale")]
    out += [(f"{api}.__init__", kw) for kw in (DECODE_INIT if api == "decode" else PREFILL_INIT).get(route, {})]
    out += [(f"{api}.plan", kw) for kw in DECODE_PLAN.get(route, {})]
    if api == "prefill":
        out.append(("prefill.plan", "causal"))
    if route == "sink-jit":
        out += [("prefill.__init__", kw) for kw in ("backend", "jit_args", "jit_kwargs")]
    if feats["window"] >= 0:
        out.append((f"{api}.plan", "window_left"))
    if feats["softcap"]:
        out.append((f"{api}.plan", "logits_soft_cap"))
    if feats["sinks"] and route != "sink-jit":
        out.append((f"{api}.run", "sinks"))
    if feats["kv"] != "bf16":
        out += [(f"{api}.run", "k_scale"), (f"{api}.run", "v_scale")]
    if feats["kv"] == "nvfp4":
        out.append((f"{api}.run", "kv_cache_sf"))
    return out


def _features(variant) -> dict:
    softcap = variant.get("softcap")
    return {"window": variant.get("window", -1), "sinks": variant.get("sinks", False), "softcap": softcap,
            "kv": variant.get("kv", "bf16"), "gain": SOFTCAP_GAIN if softcap else 1.0}


def _without(feats):
    """(label, features) of the reference with one of the variant's features taken away; q's gain stays."""
    if feats["window"] >= 0:
        yield "ref_without_window", {**feats, "window": -1}
    if feats["sinks"]:
        yield "ref_without_sinks", {**feats, "sinks": False}
    if feats["softcap"]:
        yield "ref_without_softcap", {**feats, "softcap": None}
    if feats["kv"] != "bf16":
        yield "bf16_kv_ref", {**feats, "kv": "bf16"}


def _sinks(q_heads, device):
    import torch
    return torch.linspace(*SINK_RANGE, q_heads, dtype=torch.float32, device=device)


def _cuda_lost():
    """The error text when the CUDA context no longer takes work (an illegal instruction or address is sticky), else
    None."""
    import torch
    try:
        torch.cuda.synchronize()
        torch.ones(1, device="cuda").add_(1).item()
        return None
    except Exception as exc:                                                            # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:300]


class _Arms:
    """Runs arms one at a time -- keyword check, build and first call, numerics, timing -- and records each; a (route,
    geometry, variant) whose first attempt raises is not tried again at the later cases."""

    def __init__(self, report, output, apis, workspaces):
        self.report, self.output, self.apis, self.ws = report, output, apis, workspaces
        self.tried, self.refused, self.aborted = set(), {}, False

    def run(self, rows, route, name, key, needs, build, want, others):
        import torch
        if self.aborted:
            return
        missing = [f"{api} has no keyword {kw}" for api, kw in needs
                   if self.apis.get(api) is not None and kw not in self.apis[api]]
        if missing:
            self.report["unsupported"][name] = "; ".join(missing)
            _write(self.output, self.report)
            return
        if key in self.refused:
            self.report["unavailable"][name] = f"not tried: {self.refused[key]} raised"
            return
        first = key not in self.tried
        self.tried.add(key)
        row = {}
        try:
            began = time.perf_counter()
            call, row["picked"] = build()
            got = call()
            torch.cuda.synchronize()
            row["first_call_s"] = round(time.perf_counter() - began, 2)
            row["vs_ref"] = _error(got, want)
            row["finite"] = bool(torch.isfinite(got.float()).all())
            for label, other in others.items():
                row[f"vs_{label}"] = _error(got, other)
            del got
            row.update(_time(call))
        except Exception as exc:                                                        # noqa: BLE001
            self.report["unavailable"][name] = f"{type(exc).__name__}: {exc}"[:300]
            if first and "vs_ref" not in row:
                self.refused[key] = name
            lost = _cuda_lost()
            if lost:
                self.aborted = True
                self.report["aborted"] = {"after": name, "cuda": lost}
        if row:
            rows[route] = row
            print(json.dumps({name: row}), flush=True)
        _write(self.output, self.report)


def _variants(arms, section, gname, cname, entry, geometry, cache, q, rows, qo_indptr, kvs, sinks, variants, specs):
    """Every variant's references (memoized per feature set), then its routes as arms."""
    memo = {}
    scale = geometry[2] ** -0.5

    def q_for(feats):
        return q if feats["gain"] == 1.0 else (q.float() * feats["gain"]).to(q.dtype)

    def ref(feats):
        key = tuple(sorted(feats.items()))
        if key not in memo:
            memo[key] = case_reference(cache, kvs[feats["kv"]][1], q_for(feats), rows, feats, sinks, scale)
        return memo[key]

    for vname, routes in variants.items():
        feats = _features(specs[vname])
        label = " ".join(part for part in (section, gname, cname, vname) if part)
        if feats["kv"] not in kvs:
            continue
        try:
            want = ref(feats)
            others = {tag: ref(f) for tag, f in _without(feats)}
        except Exception as exc:                                                        # noqa: BLE001
            arms.report["unavailable"][f"{label} reference"] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        entry["effects"][vname] = {tag: _error(other, want) for tag, other in others.items()}
        out = entry["rows"].setdefault(vname, {})
        for route in routes:
            build = functools.partial(build_call, section, route, geometry, cache, kvs[feats["kv"]][0], q_for(feats),
                                      qo_indptr, feats, arms.ws, sinks if feats["sinks"] else None, scale)
            arms.run(out, route, f"{label} {route}", (section, route, gname, vname), _needs(section, route, feats),
                     build, want, others)


def decode_section(arms, gname, geometry, cases, variants, device):
    import torch
    hq, hkv, d = geometry
    kinds = {_features(DECODE_VARIANTS[v])["kv"] for v in variants}
    for cname, lens in cases.items():
        if arms.aborted:
            return
        torch.manual_seed(SEED)
        cache = paged_cache(lens, hkv, d, device)
        q = torch.randn(len(lens), hq, d, device=device).to(torch.bfloat16)
        entry = arms.report["decode"].setdefault(gname, {}).setdefault(cname, {
            "kv_lens": list(lens), "pages": int(cache["indices"].numel()), "pool_pages": int(cache["k"].shape[0]),
            "effects": {}, "rows": {}})
        kvs = {"bf16": ({"k": cache["k"], "v": cache["v"]}, (cache["k"], cache["v"]))}
        for kind, quantize in (("fp8", fp8_kv), ("nvfp4", nvfp4_kv)):
            if kind not in kinds:
                continue
            try:
                kernel, ref_pools, entry[kind] = quantize(cache)
                kvs[kind] = (kernel, ref_pools)
            except Exception as exc:                                                    # noqa: BLE001
                failed = f"{type(exc).__name__}: {exc}"[:300]
                arms.report["unavailable"][f"decode {gname} {cname} {kind} cache"] = failed
        qo_indptr = torch.arange(len(lens) + 1, dtype=torch.int32, device=device)
        _variants(arms, "decode", gname, cname, entry, geometry, cache, q, 1, qo_indptr, kvs, _sinks(hq, device),
                  variants, DECODE_VARIANTS)
        del cache, q, kvs
        torch.cuda.empty_cache()


def prefill_section(arms, gname, geometry, variants, device):
    import torch
    if arms.aborted:
        return
    hq, hkv, d = geometry
    torch.manual_seed(SEED)
    cache = paged_cache((PREFIX + NEW,), hkv, d, device)
    q = torch.randn(NEW, hq, d, device=device).to(torch.bfloat16)
    entry = arms.report["prefill"].setdefault(gname, {
        "new": NEW, "prefix": PREFIX, "pages": int(cache["indices"].numel()), "effects": {}, "rows": {}})
    kvs = {"bf16": ({"k": cache["k"], "v": cache["v"]}, (cache["k"], cache["v"]))}
    qo_indptr = torch.tensor([0, NEW], dtype=torch.int32, device=device)
    _variants(arms, "prefill", gname, "", entry, geometry, cache, q, NEW, qo_indptr, kvs, _sinks(hq, device), variants,
              PREFILL_VARIANTS)
    del cache, q, kvs
    torch.cuda.empty_cache()


def run(output=None) -> dict:
    import torch
    began = time.perf_counter()
    report = {"lane": "sm121_attention",
              "geometries": {g: dict(zip(("q_heads", "kv_heads", "head_dim"), s)) for g, s in GEOMETRIES.items()},
              "settings": {"page": PAGE, "layout": "NHD", "spare_pages": SPARE_PAGES, "garbage": GARBAGE,
                           "sinks": f"linspace{SINK_RANGE} over the query heads, fp32", "softcap_q_gain": SOFTCAP_GAIN,
                           "seed": SEED, "warmup": WARMUP, "repeats": REPEATS},
              "not_tried": NOT_TRIED, "unavailable": {}, "unsupported": {}, "decode": {}, "prefill": {}}
    _device(report, MEMORY_CAP_GIB)
    try:
        report["flashinfer"] = flashinfer_facts()
    except Exception as exc:                                                            # noqa: BLE001 -- the answer
        report["unavailable"]["flashinfer"] = f"{type(exc).__name__}: {exc}"[:300]
        print(_write(output, report), flush=True)
        return report
    apis = api_keywords(report)
    device = torch.device("cuda")
    workspaces = {name: torch.zeros(mib << 20, dtype=torch.uint8, device=device) for name, mib in WORKSPACE_MIB.items()}
    arms = _Arms(report, output, apis, workspaces)
    with torch.inference_mode():
        for gname, geometry in GEOMETRIES.items():
            decode_section(arms, gname, geometry, DECODE_CASES,
                           {v: spec["routes"] for v, spec in DECODE_VARIANTS.items()}, device)
            prefill_section(arms, gname, geometry, {v: spec["routes"] for v, spec in PREFILL_VARIANTS.items()}, device)
        first = next(iter(DECODE_CASES))
        for gname, geometry in GEOMETRIES.items():
            decode_section(arms, gname, geometry, {first: DECODE_CASES[first]},
                           dict.fromkeys(CUTE_DSL["decode"], ("cute-dsl",)), device)
            prefill_section(arms, gname, geometry, dict.fromkeys(CUTE_DSL["prefill"], ("cute-dsl",)), device)
    report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report["seconds"] = round(time.perf_counter() - began, 1)
    print(_write(output, report), flush=True)
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
