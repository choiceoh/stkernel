# SPDX-License-Identifier: Apache-2.0
"""ST sparse MLA with GB10 warp MMA and cluster-local split reduction."""
import hashlib
import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)
MLA_D = 512
MLA_H = 16
MLA_SPLITS_MAX = 64
MLA_MAX_SPLIT_ROWS = 64
MLA_WS_ROWS = 3 * MLA_MAX_SPLIT_ROWS
_MLA_WS = None
_WS = None
_EXT = None
_MLA_CLUSTER_MAX = 0
_ARMED = {"mla": False}
# Large-M prefill candidates (39차). tile32 is the qualified production default
# after GPU numerical/graph/sanitizer and matched serving brackets; stock remains
# available as a baseline and pair/pair4 remain comparison candidates. D11
# (2026-09-12, ST): the profile's STK_mla_prefill knob selects one through
# configure_prefill() before the lane arms; nothing in this module reads the
# environment except the build root (a cache path, like TRITON_CACHE_DIR).
PREFILL_MODES = ("stock", "tile32", "pair", "pair4")
# Probe hooks (never env, never serving): mla_decode(splits=, probe=) overrides the
# split rule and selects the kernel's roofline mode (1 = streams only, 2 = + the dot,
# probes/mk_mla_bench.py); PAIR_STATS logs the adjacent-row selection overlap that the
# pair candidate's forecast rests on (39차: never recorded in vLLM -- an env trap).
PAIR_STATS = False
# GB10 cluster-local split reduction (measurements/st_gb10_mla_20260911: wins for
# 32 <= T <= 64 at split 2/3, adopted as the default dispatch). D11: the served form,
# not an env switch -- an A/B flips this module attribute before maybe_arm().
ENABLE_MLA_CLUSTER = True
ENABLE_MLA_PREFILL32 = False
ENABLE_MLA_PREFILL_PAIR = False
MLA_PREFILL_GROUP = 2


def configure_prefill(mode: str) -> None:
    """Select the served large-M prefill path once, before maybe_arm()."""
    global ENABLE_MLA_PREFILL32, ENABLE_MLA_PREFILL_PAIR, MLA_PREFILL_GROUP
    if mode not in PREFILL_MODES:
        raise ValueError(f"STK_mla_prefill must be one of {PREFILL_MODES}, got {mode!r}")
    want = (mode == "tile32", mode in ("pair", "pair4"), 4 if mode == "pair4" else 2)
    if want != (ENABLE_MLA_PREFILL32, ENABLE_MLA_PREFILL_PAIR, MLA_PREFILL_GROUP) and _ARMED["mla"]:
        raise RuntimeError("configure_prefill: the MLA lane is already armed with another prefill mode")
    ENABLE_MLA_PREFILL32, ENABLE_MLA_PREFILL_PAIR, MLA_PREFILL_GROUP = want


def _build():
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils.cpp_extension import load
    src = Path(__file__).with_name("glm53_megakernel.cu")
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a"]
    key = hashlib.sha256(src.read_bytes() + repr((flags, torch.__version__, torch.version.cuda)).encode()).hexdigest()[:16]
    root = Path(os.environ.get("ST_MLA_BUILD_ROOT", str(Path.home() / ".cache/st/mla")))
    build = root / key
    build.mkdir(parents=True, exist_ok=True)
    _EXT = load(name="st_mla_" + key, sources=[str(src)], extra_cuda_cflags=flags,
                build_directory=str(build), verbose=False)
    return _EXT


def _ensure_workspace(device):
    global _WS
    import torch
    if _WS is None:
        _WS = {name: torch.zeros(8, dtype=torch.int32, device=device)
               for name in ("barrier", "barrier_mla")}
    return _WS


def maybe_arm():
    """Compile and judge the mandatory MLA lane once, before graph capture."""
    global _MLA_CLUSTER_MAX
    if _ARMED["mla"]:
        return
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("ST MLA must be warmed before CUDA graph capture")
    ext = _build()
    major, minor, sms, _ = ext.probe_device()
    if (major, minor, sms) != (12, 1, 48):
        raise RuntimeError(f"ST MLA requires GB10 SM121/48 SMs, got {major}.{minor}/{sms}")
    _MLA_CLUSTER_MAX = int(ext.mla_cluster_max()) if ENABLE_MLA_CLUSTER else 0
    if not _selftest_mla():
        raise RuntimeError("ST MLA numerical self-test failed")
    _ARMED["mla"] = True


def _rel_err(a, b) -> float:
    import torch

    d = float((a.float() - b.float()).norm())
    den = float(b.float().norm())
    if not math.isfinite(d) or not math.isfinite(den):
        return math.inf
    error = float(d / den) if den > 0 else float(d)
    # Every caller must fail closed: max(0.0, NaN) is 0.0 and NaN > tol
    # is false. Neither an invalid output nor an invalid oracle can arm a
    # segment, including replay/aggregate gates that use those idioms.
    return error if math.isfinite(error) else math.inf



def _mla_workspace(device, T: int, splits: int):
    """Split partials + (m, l): allocated once at MLA_WS_ROWS, never grown."""
    global _MLA_WS
    import torch

    if _MLA_WS is None:
        _MLA_WS = {
            "cap": MLA_WS_ROWS,
            "part": torch.zeros(MLA_WS_ROWS * MLA_H * MLA_D, dtype=torch.float32, device=device),
            "pml": torch.zeros(MLA_WS_ROWS * MLA_H * 2, dtype=torch.float32, device=device),
        }
    need = T * splits
    if need > _MLA_WS["cap"]:
        raise RuntimeError(
            f"mla: T={T} x splits={splits} = {need} rows exceed the fixed "
            f"workspace of {MLA_WS_ROWS}; mla_splits() must bound T*splits")
    return _MLA_WS



def mla_splits(T: int, forced: "int | None" = None) -> int:
    """Slot-axis splits for this row count.

    Prefer the smallest s making T*s a multiple of the measured resident
    grid. If the fixed scratch budget prevents that, use the nearest split
    count within the budget. Rows above MLA_MAX_SPLIT_ROWS are unsplit
    prefill. Cluster reduction uses this same plan and numerical order.
    """
    if _EXT is None or T <= 0:
        return 1
    if T > MLA_MAX_SPLIT_ROWS:
        # prefill: every row is its own item, the kernel normalises in place
        # and no [T][splits][H][D] fp32 scratch exists (268 MB at T=8192)
        return 1
    grid = int(_EXT.mla_grid())
    budget = max(1, min(MLA_SPLITS_MAX, MLA_WS_ROWS // T))
    if forced:                                           # probe hook
        return max(1, min(budget, int(forced)))
    for s in range(1, budget + 1):
        if (T * s) % grid == 0:
            return s
    return max(1, min(budget, round(grid / T)))



def _mla_uses_cluster(T: int, W: int, splits: int) -> bool:
    """Use only the small clusters that win on GB10, preserving split order.

    Larger clusters lose SM occupancy/packing efficiency. The existing split
    planner remains authoritative: this changes the reduction's storage and
    synchronization, not attention selection or floating point summation order.
    """
    return (ENABLE_MLA_CLUSTER and 32 <= T <= MLA_MAX_SPLIT_ROWS
            and 1 <= W <= 2176 and 2 <= splits <= min(3, _MLA_CLUSTER_MAX))


def mla_decode(q_nope, ckv, slots, lens, sm_scale: float, ckv_scale: float,
               out=None, *, splits: "int | None" = None, probe: int = 0):
    """Sparse MLA decode over the indexer's top-k slots.

    q_nope [T, H, D] bf16 (never quantised -- the sparse backend forbids it);
    ckv    e4m3 [num_slots, D] flat latent cache; slots [T, W] int32 global
    slot ids with the valid prefix first; lens [T] int32 valid counts ON THE
    DEVICE (that is what keeps this launch inside the captured graph)."""
    import torch

    T, H, D = q_nope.shape
    assert (H, D) == (MLA_H, MLA_D), f"mla: shape {(H, D)} != {(MLA_H, MLA_D)}"
    assert q_nope.is_contiguous() and slots.is_contiguous()
    assert slots.dtype == torch.int32 and lens.dtype == torch.int32
    if (ENABLE_MLA_PREFILL32 and not ENABLE_MLA_PREFILL_PAIR
            and 4096 <= T <= 8192 and 1 <= slots.shape[1] <= 2176
            and q_nope.dtype == torch.bfloat16 and ckv.is_contiguous()
            and ckv.element_size() == 1 and lens.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()):
        result = _mla_prefill32(q_nope, ckv, slots, lens, sm_scale, ckv_scale, out)
        if not getattr(_mla_prefill32, "_announced", False):
            _mla_prefill32._announced = True
            logger.warning("[megakernel] mla prefill32 LAUNCHED T=%d W=%d register-Q tile=32",
                           T, slots.shape[1])
        return result
    if (ENABLE_MLA_PREFILL_PAIR and 128 <= T <= 8192
            and 1 <= slots.shape[1] <= 2176
            and q_nope.dtype == torch.bfloat16 and ckv.is_contiguous()
            and ckv.element_size() == 1 and lens.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()):
        return _mla_prefill_pair(q_nope, ckv, slots, lens, sm_scale, ckv_scale, out)
    splits = mla_splits(T, splits)
    if (probe == 0 and _mla_uses_cluster(T, slots.shape[1], splits)
            and q_nope.dtype == torch.bfloat16 and ckv.is_contiguous()
            and ckv.element_size() == 1 and lens.is_contiguous()):
        if out is None:
            out = torch.empty_like(q_nope)
        _EXT.run_mla_cluster(
            [q_nope.data_ptr(), ckv.data_ptr(), slots.data_ptr(), lens.data_ptr(), out.data_ptr()],
            [float(sm_scale), float(ckv_scale)], [int(T), int(slots.shape[1]), int(splits)],
        )
        return out
    ws = _ensure_workspace(q_nope.device)
    assert splits == 1 or T * splits <= MLA_WS_ROWS, (T, splits)
    mw = (_mla_workspace(q_nope.device, T, splits) if splits > 1
          else {"part": ws["barrier"], "pml": ws["barrier"]})   # unused when splits == 1
    if out is None:
        out = torch.empty_like(q_nope)
    _EXT.run_mla(
        [q_nope.data_ptr(), ckv.data_ptr(), slots.data_ptr(), lens.data_ptr(),
         out.data_ptr(), mw["part"].data_ptr(), mw["pml"].data_ptr(),
         ws["barrier_mla"].data_ptr()],
        [float(sm_scale), float(ckv_scale)],
        [int(T), int(slots.shape[1]), int(splits), int(probe)],
    )
    return out



def _mla_prefill32(q_nope, ckv, slots, lens, sm_scale, ckv_scale, out=None):
    import torch

    if out is None:
        out = torch.empty_like(q_nope)
    _EXT.run_mla_prefill32(
        [q_nope.data_ptr(), ckv.data_ptr(), slots.data_ptr(), lens.data_ptr(),
         out.data_ptr()], [float(sm_scale), float(ckv_scale)],
        [int(q_nope.shape[0]), int(slots.shape[1])],
    )
    return out



def _mla_prefill_pair(q_nope, ckv, slots, lens, sm_scale, ckv_scale, out=None):
    """Two or four adjacent rows reuse FP8 loads with independent membership.

    The schedule holds only slot IDs and one membership bit per row, never gathered
    KV. Its size is bounded by T*W, independent of the physical cache span.
    Per-call storage is deliberately eager-only: it cannot invalidate the
    fixed workspace used by captured decode graphs. Duplicate input slots
    remain duplicate schedule entries, with each row's multiplicity intact.
    Ordering can differ from the indexer, so this is not a bit-exact claim.
    """
    import torch

    T, _, _ = q_nope.shape
    W = slots.shape[1]
    width = MLA_PREFILL_GROUP
    _fn = _mla_prefill_pair
    if not getattr(_fn, "_announced", False):
        # 39차: the bracket's proof that the optional path ran (P1 had none);
        # self-contained so a def exec'd alone in a stub namespace still runs
        _fn._announced = True
        _lg = globals().get("logger")
        if _lg is not None:
            _lg.warning("[megakernel] mla prefill pair engaged (T=%d, W=%d, group=%d)", T, W, width)
    groups = (T + width - 1) // width
    if PAIR_STATS and getattr(_fn, "_stats_calls", 0) < 6:
        # 39차 diagnostic: how much of the sparse selection adjacent rows really
        # share. The forecast assumed 75 % common selection; the pair kernel's work
        # is the UNION of the group's selections, so union/W is the traffic ratio.
        _fn._stats_calls = getattr(_fn, "_stats_calls", 0) + 1
        try:
            with torch.no_grad():
                valid = torch.arange(W, device=slots.device)[None, :] < lens[:, None]
                s = torch.where(valid, slots.long(), torch.full_like(slots.long(), -1))
                rows = (T // width) * width
                g = s[:rows].view(-1, width, W)                       # [groups, width, W]
                a = g[:, 0]; b = g[:, 1]
                inter = ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)).any(-1).sum(-1).float()
                la = (a >= 0).sum(-1).float(); lb = (b >= 0).sum(-1).float()
                union = la + lb - inter
                jac = (inter / union.clamp_min(1)).mean().item()
                flat = g.reshape(g.shape[0], -1)
                srt, _ = flat.sort(-1)
                uniq = ((srt[:, 1:] != srt[:, :-1]) & (srt[:, 1:] >= 0)).sum(-1).float() + (srt[:, :1] >= 0).sum(-1).float()
                lsum = (flat >= 0).sum(-1).float()
                ratio = (uniq / lsum.clamp_min(1)).mean().item()
                logger.warning("[megakernel] mla pair stats: T=%d W=%d group=%d mean len=%.0f | adjacent-row jaccard=%.3f "
                               "| group union/sum-of-lengths=%.3f (forecast assumed 0.4375 traffic at 75%% common)",
                               T, W, width, lsum.mean().item() / width, jac, ratio)
        except Exception as e:  # noqa: BLE001 -- a diagnostic must never take the lane down
            logger.warning("[megakernel] mla pair stats failed: %r", e)
    schedule = torch.empty((groups, width * W), dtype=torch.int32, device=q_nope.device)
    membership = torch.empty_like(schedule)
    lengths = torch.empty(groups, dtype=torch.int32, device=q_nope.device)
    if out is None:
        out = torch.empty_like(q_nope)
    launch = (_EXT.run_mla_prefill_group4 if width == 4
              else _EXT.run_mla_prefill_pair)
    launch(
        [q_nope.data_ptr(), ckv.data_ptr(), slots.data_ptr(), lens.data_ptr(),
         out.data_ptr(), schedule.data_ptr(), membership.data_ptr(), lengths.data_ptr()],
        [float(sm_scale), float(ckv_scale)], [int(T), int(W)],
    )
    return out



def mla_decode_ref(q_nope, ckv, slots, lens, sm_scale: float, ckv_scale: float):
    """Pure-torch twin of the kernel, in fp32. Same contract, no pipelining."""
    import torch

    T, H, D = q_nope.shape
    out = torch.zeros(T, H, D, dtype=torch.float32, device=q_nope.device)
    q = q_nope.float()
    for t in range(T):
        n = int(lens[t].item())
        if n <= 0:
            continue
        idx = slots[t, :n].long()
        c = ckv.view(-1, D)[idx].to(torch.float32) * ckv_scale   # [n, D]
        s = (q[t] @ c.T) * sm_scale                              # [H, n]
        p = torch.softmax(s, dim=-1)
        out[t] = p @ c
    return out.to(q_nope.dtype)



def _selftest_mla() -> bool:
    """Diff the kernel against the torch twin on the serving geometry.

    Gates on the ranking-safe band: the twin sums in fp32 in slot order while
    the kernel runs an online softmax over split partials, so the two differ
    in summation order only. bf16 output rounding is 2^-8 relative, which is
    the floor here."""
    import torch

    torch.manual_seed(0)
    dev = "cuda"
    worst = 0.0
    # Include awkward split grids and the unsplit prefill store.
    cases = [(8, 2048, False), (16, 2048, True), (32, 512, True), (1, 64, False),
             (40, 2048, True), (100, 2048, True)]
    if _MLA_CLUSTER_MAX:
        cases += [(32, 1, True), (48, 33, True), (64, 2176, True)]
    if ENABLE_MLA_PREFILL32 and not ENABLE_MLA_PREFILL_PAIR:
        cases += [(128, 1, True), (129, 33, True), (131, 2176, True)]
    if ENABLE_MLA_PREFILL_PAIR:
        # The optional path begins at T=128. Its boot gate must exercise
        # shared selections, repeated slots, low overlap, odd T and empty
        # rows before the existing MLA segment is allowed to arm.
        cases += [(128, 2048, False), (129, 2176, True), (130, 64, True)]
        if MLA_PREFILL_GROUP == 4:
            cases += [(131, 2176, True)]
    for T, W, ragged in cases:
        num_slots = (max(4096, 4 * W)
                     if ENABLE_MLA_PREFILL_PAIR and MLA_PREFILL_GROUP == 4 and T >= 128
                     else 4096)
        q = torch.randn(T, MLA_H, MLA_D, dtype=torch.bfloat16, device=dev) * 0.3
        cache = (torch.randn(num_slots, MLA_D, device=dev) * 0.5).to(torch.float8_e4m3fn)
        slots = torch.randint(0, num_slots, (T, W), dtype=torch.int32, device=dev)
        if ragged:
            lens = torch.randint(1, W + 1, (T,), dtype=torch.int32, device=dev)
        else:
            lens = torch.full((T,), W, dtype=torch.int32, device=dev)
        if _mla_uses_cluster(T, W, mla_splits(T)):
            lens[0] = 0
            slots[0].fill_(-1)
        if ENABLE_MLA_PREFILL32 and T >= 128:
            lens[0] = 0
            slots[0].fill_(-1)
            lens[1] = W
            slots[1].fill_(0)  # repeated selections preserve multiplicity
        if ENABLE_MLA_PREFILL_PAIR and T >= 128:
            slots[1].copy_(slots[0])
            lens[1].copy_(lens[0])
            if MLA_PREFILL_GROUP == 4 and not ragged:
                slots[2].copy_(slots[0])
                slots[3].copy_(slots[0])
                lens[2:4].copy_(lens[0].expand(2))
            if ragged:
                lens[2:4].zero_()
            if MLA_PREFILL_GROUP == 4:
                # Four disjoint full lists force weak-reuse fallback; at
                # W2176 their 8704 unique keys also exceed the hash capacity.
                # The following group covers unequal per-row membership and
                # multiplicity on the shared route instead of only identical
                # lists. These fixtures run only with the explicit group knob.
                col = torch.arange(W, dtype=torch.int32, device=dev)
                common = 3 * W // 4
                for row in range(4):
                    slots[4 + row].copy_(col + row * W)
                    slots[8 + row, :common].copy_(col[:common])
                    slots[8 + row, common:].copy_(col[common:] + row * (W - common))
                lens[4:12].fill_(W)
                slots[9, common].copy_(slots[8, 0])
        sm, ks = MLA_D ** -0.5, 0.7
        # Small edge fixtures exercise the new device kernel directly; the
        # serving selector keeps actual chunks below 4096 on the old kernel.
        # This does not emit the serving-path engagement marker.
        mla_call = (_mla_prefill32 if ENABLE_MLA_PREFILL32 and not ENABLE_MLA_PREFILL_PAIR
                    and T >= 128 else mla_decode)
        got = mla_call(q, cache.view(torch.uint8), slots, lens, sm, ks)
        ref = mla_decode_ref(q, cache, slots, lens, sm, ks)
        torch.cuda.synchronize()
        error = _rel_err(got.float(), ref.float())
        if not error <= 2e-2:
            logger.warning("[megakernel] selftest mla T=%d W=%d rel=%.2e -> DISARM",
                           T, W, error)
            return False
        worst = max(worst, error)
    logger.warning("[megakernel] selftest mla rel=%.2e pair_prefill=%s group=%d prefill32=%s -> ARM",
                   worst, ENABLE_MLA_PREFILL_PAIR, MLA_PREFILL_GROUP, ENABLE_MLA_PREFILL32)
    return True
