# SPDX-License-Identifier: Apache-2.0
"""ST sparse MLA with GB10 warp MMA and cluster-local split reduction."""
import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)
# The compiled cell, stated once in engine/kernels/cells.py (the wizard's table refuses against the same numbers).
from engine.kernels.cells import MLA_HEADS as MLA_H, MLA_LATENT as MLA_D   # noqa: E402
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
# available as a baseline. The pair/pair4 candidates read the UNION of a group's
# selections once, betting that adjacent queries share most of theirs -- measured
# at last (45차 §23 조사 16차) and retired: with a 32K context's latent, tile32 is
# flat at 18.3 ms whatever the overlap while pair is 42~61 ms and pair4 47~111,
# and even at a 100% shared selection tile32 still wins. D11 (2026-09-12, ST): the
# profile's STK_mla_prefill knob selects one through configure_prefill() before the
# lane arms; nothing in this module reads the environment except the build root
# (a cache path, like TRITON_CACHE_DIR).
PREFILL_MODES = ("stock", "tile32")
# Probe hook (never env, never serving): mla_decode(splits=, probe=) overrides the
# split rule and selects the kernel's roofline mode (1 = streams only, 2 = + the dot,
# probes/mk_mla_bench.py).
# GB10 cluster-local split reduction (measurements/st_gb10_mla_20260911: wins for
# 32 <= T <= 64 at split 2/3, adopted as the default dispatch). D11: the served form,
# not an env switch -- an A/B flips this module attribute before maybe_arm().
ENABLE_MLA_CLUSTER = True
ENABLE_MLA_PREFILL32 = False


def configure_prefill(mode: str) -> None:
    """Select the served large-M prefill path once, before maybe_arm()."""
    global ENABLE_MLA_PREFILL32
    if mode not in PREFILL_MODES:
        raise ValueError(f"STK_mla_prefill must be one of {PREFILL_MODES}, got {mode!r}")
    want = mode == "tile32"
    if want != ENABLE_MLA_PREFILL32 and _ARMED["mla"]:
        raise RuntimeError("configure_prefill: the MLA lane is already armed with another prefill mode")
    ENABLE_MLA_PREFILL32 = want


def _bound():
    from engine.base.kernel_shape import bound
    return bound()


def _check_cell():
    """The kernel is compiled for one attention cell (MLA_H heads over an MLA_D latent per rank);
    a bound kernel shape that differs is refused by name before anything is armed (D3)."""
    a = _bound().attention
    if (a.kind, a.heads, a.head_dim) != ("mla", MLA_H, MLA_D):
        raise RuntimeError(f"ST MLA is compiled for the {MLA_H} heads x {MLA_D} latent MLA cell; "
                           f"the bound kernel shape asks for {a}")


def _build():
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils.cpp_extension import load
    from engine.kernels.native_cache import prepare_sources
    src = Path(__file__).with_name("glm53_megakernel.cu")
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a"]
    root = Path(os.environ.get("ST_MLA_BUILD_ROOT", str(Path.home() / ".cache/st/mla")))
    key, build, sources = prepare_sources(root, [src], (flags, torch.__version__, torch.version.cuda))
    _EXT = load(name="st_mla_" + key, sources=list(sources), extra_cuda_cflags=flags,
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
    _check_cell()
    ext = _build()
    major, minor, sms, _ = ext.probe_device()
    device = _bound().device
    if (major, minor) != device.capability or sms != device.sms:
        raise RuntimeError(f"ST MLA requires GB10 SM{device.capability[0]}{device.capability[1]}/{device.sms} SMs, "
                           f"got {major}.{minor}/{sms}")
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
    if (ENABLE_MLA_PREFILL32
            and 4096 <= T <= 32768 and 1 <= slots.shape[1] <= 2176
            and q_nope.dtype == torch.bfloat16 and ckv.is_contiguous()
            and ckv.element_size() == 1 and lens.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()):
        result = _mla_prefill32(q_nope, ckv, slots, lens, sm_scale, ckv_scale, out)
        if not getattr(_mla_prefill32, "_announced", False):
            _mla_prefill32._announced = True
            logger.warning("[megakernel] mla prefill32 LAUNCHED T=%d W=%d register-Q tile=32",
                           T, slots.shape[1])
        return result
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
    if ENABLE_MLA_PREFILL32:
        cases += [(128, 1, True), (129, 33, True), (131, 2176, True)]
    for T, W, ragged in cases:
        num_slots = 4096
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
        sm, ks = MLA_D ** -0.5, 0.7
        # Small edge fixtures exercise the new device kernel directly; the
        # serving selector keeps actual chunks below 4096 on the old kernel.
        # This does not emit the serving-path engagement marker.
        mla_call = _mla_prefill32 if ENABLE_MLA_PREFILL32 and T >= 128 else mla_decode
        got = mla_call(q, cache.view(torch.uint8), slots, lens, sm, ks)
        ref = mla_decode_ref(q, cache, slots, lens, sm, ks)
        torch.cuda.synchronize()
        error = _rel_err(got.float(), ref.float())
        if not error <= 2e-2:
            logger.warning("[megakernel] selftest mla T=%d W=%d rel=%.2e -> DISARM",
                           T, W, error)
            return False
        worst = max(worst, error)
    logger.warning("[megakernel] selftest mla rel=%.2e prefill32=%s -> ARM", worst, ENABLE_MLA_PREFILL32)
    return True
