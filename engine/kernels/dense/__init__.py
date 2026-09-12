"""Explicit GB10 dense lanes: W4A8 decode, FP8 and NVFP4 prefill.

Packs and the native module belong to the bound layer, before graph capture.
There is no launch-time fallback or vLLM import. Production may explicitly
retire BF16 storage into prepared packs; standalone numerical tests retain it.
"""
from dataclasses import dataclass
from functools import cache
import os
from pathlib import Path

import torch


@cache
def extension():
    from torch.utils.cpp_extension import load
    from engine.kernels.native_cache import prepare_sources
    source = Path(__file__).with_name("kernels.cu")
    flags = ["-O2", "-gencode", "arch=compute_121a,code=sm_121a",
             "-DMK_GRID_DEF=96", "-DMK_MHC_GRID_DEF=144", "-DMK_NBUF2_DEF=3",
             "-DMK_FP8_PACK2_DEF=1", "-DMK_GEMM_TRANSPOSE_M8_DEF=1",
             "-DMK_GEMM_COMPACT_M8_DEF=1", "-DMK_M8_FASTPATH_DEF=1"]
    root = Path(os.environ.get("ST_DENSE_BUILD_ROOT", str(Path.home()/".cache/st/dense")))
    key, directory, sources = prepare_sources(root, [source], (flags, torch.__version__, torch.version.cuda))
    ext = load(name="st_dense_"+key, sources=list(sources), extra_cuda_cflags=flags,
               build_directory=str(directory), verbose=False)
    if tuple(ext.probe_device())[:3] != (12, 1, 48):
        raise RuntimeError("native dense lane requires GB10 SM121 with 48 SMs")
    return ext


@dataclass(frozen=True)
class W4Pack:
    data: torch.Tensor
    scale: torch.Tensor
    rowscale: torch.Tensor
    rows: int
    cols: int
    calibrated: bool = False    # GPTQ from a calibration Hessian (the store says), not round-to-nearest


GPTQ_ACT_ORDER = True       # columns in decreasing Hessian-diagonal order with static groups (45차 §23 GPU 판정 7차)
TILE = 4096                 # one GPTQ tile of K: a wider weight is packed as a sequence of them
KMAX = 20480                # the decode kernel's widest K (kernels.cu KBLK_LIMIT): the drafter's fc, whole


def packed_nbytes(rows, cols, *, prefill=True):
    """Aligned resident bound for W4 tiles (folded or not) plus optional FP8.

    Calibration may fold tiles and share their row scale, so declare the larger
    unmerged form. This does not reserve the BF16 source or packing scratch.
    """
    if rows <= 0 or cols <= 0 or cols % 128 or cols > KMAX:
        raise ValueError('dense storage requires positive rows and supported 128-aligned columns')
    padded, end = (rows + 127) // 128 * 128, 0
    def add(size):
        nonlocal end
        end = (end + 255) // 256 * 256 + size
    for start in range(0, cols, TILE):
        width = min(TILE, cols - start)
        add(padded * width // 2)  # two W4 values per byte
        add(padded * width // 16)  # one E4M3 scale per group
        add(padded * 4)  # FP32 row scales
    if prefill:
        add(padded * cols)
        add((padded // 128) * (cols // 128) * 4)
    return (end + 255) // 256 * 256


def _tile_pack(codes, scales, shift, n, k0, k1):
    """One W4Pack of columns [k0, k1) out of row-major codes/scales of a wider weight."""
    padded = shift.shape[0]
    k = k1 - k0
    pairs = codes[:, k0:k1].reshape(padded, k//2, 2)
    data = (pairs[..., 0] | (pairs[..., 1] << 4))
    data = data.view(padded//128, 128, k//128, 64).permute(0, 2, 1, 3).contiguous()
    sc = scales[:, k0//16:k1//16].reshape(padded//128, 128, k//128, 8).permute(0, 2, 1, 3).contiguous()
    return W4Pack(data, sc, torch.exp2(-shift).contiguous(), n, k)


@torch.inference_mode()
def pack_w4(weight, *, hessian=None, per_row=True, act_order=None, factor=None):
    from .packing import _E2M1_GRID, _E2M1_MIDS, _w4_row_shift, _w4_rtn_codes, _w4_gptq_codes
    if (not weight.is_cuda or weight.ndim != 2 or weight.dtype != torch.bfloat16
            or weight.shape[1] % 128 or not 0 < weight.shape[1] <= TILE):
        raise ValueError("W4 packing requires CUDA BF16 [N,K], K in 128..4096 aligned to 128")
    n, k = weight.shape
    padded = (n+127)//128*128
    mids = torch.tensor(_E2M1_MIDS, device=weight.device)
    grid = torch.tensor(_E2M1_GRID, device=weight.device)
    need, shift, _ = _w4_row_shift(weight, padded, k//16, per_row)
    if hessian is None:
        codes, scales = _w4_rtn_codes(weight, shift, need, mids, grid)
    else:
        if hessian.shape != (k, k) or not torch.isfinite(hessian).all():
            raise ValueError("GPTQ Hessian must be finite and match the input dimension")
        codes, scales = _w4_gptq_codes(weight, shift, need, hessian, mids, grid, factor=factor,
                                       act_order=GPTQ_ACT_ORDER if act_order is None else act_order)
    pack = _tile_pack(codes, scales, shift, n, 0, k)
    return W4Pack(pack.data, pack.scale, pack.rowscale, n, k, hessian is not None)


@torch.inference_mode()
def pack_w4_wide(weight, hessian, *, per_row=True, act_order=None, factor=None):
    """GPTQ over the WHOLE K of a weight wider than a tile (the drafter's fc: 20480 = five tiles), with the full
    Hessian: the error feedback crosses tile boundaries, which tile-by-tile packing cannot (its input, the target's
    hidden states of five layers, is correlated across the tiles). The fp64 factorisation runs on the CPU. Returns
    the tiles' W4Packs, each calibrated."""
    from .packing import _E2M1_GRID, _E2M1_MIDS, _w4_row_shift, _w4_gptq_codes
    if (not weight.is_cuda or weight.ndim != 2 or weight.dtype != torch.bfloat16
            or weight.shape[1] % TILE or weight.shape[1] <= TILE):
        raise ValueError("wide W4 packing takes a CUDA BF16 [N,K] with K a multiple of 4096 above 4096")
    n, k = weight.shape
    if hessian.shape != (k, k) or not torch.isfinite(hessian).all():
        raise ValueError("GPTQ Hessian must be finite and match the input dimension")
    padded = (n+127)//128*128
    mids = torch.tensor(_E2M1_MIDS, device=weight.device)
    grid = torch.tensor(_E2M1_GRID, device=weight.device)
    need, shift, _ = _w4_row_shift(weight, padded, k//16, per_row)
    codes, scales = _w4_gptq_codes(weight, shift, need, hessian, mids, grid, factor=factor,
                                   act_order=GPTQ_ACT_ORDER if act_order is None else act_order, factor_device="cpu")
    packs = []
    for k0 in range(0, k, TILE):
        t = _tile_pack(codes, scales, shift, n, k0, k0 + TILE)
        packs.append(W4Pack(t.data, t.scale, t.rowscale, n, TILE, True))
    return packs


def _fold(packs):
    """One pack out of a wide weight's K tiles, when they are tiles of one quantisation.

    `pack_wide` and `pack_w4_wide` take the row shift over the whole K and hand every tile the same one, so
    their tiles differ only in which columns they hold: concatenating the tile-major data and scales along the
    k-tile axis IS the wide pack the kernel wants. It matters because the call that used to walk them summed
    each tile's output AFTER the launch had already rounded it to bf16 -- five launches, five roundings, four
    adds and a cast for one product. Folded, the k blocks accumulate in fp32 inside the kernel and round once.

    The tiles a first, uncalibrated boot files take their shift per tile (kernels/dense/packing), so those do
    not fold and are left as they are; the next boot's store packs the weight whole.
    """
    packs = list(packs)
    if len(packs) < 2 or sum(p.cols for p in packs) > KMAX:
        return packs
    first = packs[0]
    if not all(p.rows == first.rows and p.calibrated == first.calibrated
               and p.rowscale.shape == first.rowscale.shape and torch.equal(p.rowscale, first.rowscale)
               for p in packs[1:]):
        return packs
    return [W4Pack(torch.cat([p.data for p in packs], 1).contiguous(),
                   torch.cat([p.scale for p in packs], 1).contiguous(),
                   first.rowscale, first.rows, sum(p.cols for p in packs), first.calibrated)]


def w4_gemm(x, pack):
    if (x.ndim != 2 or not 1 <= x.shape[0] <= 32 or x.shape[1] != pack.cols or x.shape[1] > KMAX
            or x.dtype != torch.bfloat16 or x.device != pack.data.device):
        raise ValueError("W4 decode requires 1..32 BF16 rows matching the bound pack, K at most 20480")
    out = torch.empty(x.shape[0], pack.rows, dtype=x.dtype, device=x.device)
    extension().run_gemm(x.contiguous(), pack.data, pack.scale, out, pack.rows,
                         1., 0, pack.rowscale.data_ptr(), 0, 0, 0)
    return out


class DenseLinear:
    """Immutable dispatch: <=32 rows W4A8, above that FP8.

    There was a third lane: NVFP4 (W4A4) above 1023 rows, adopted in 39차 on
    the vLLM stack's throughput alone. Measured against THIS engine's FP8
    lane on the shapes this rank serves, it is not faster -- +3.7% at 1K
    rows, -2.3% at 8K, a wash -- while its fp4 activations cost 3.65x the
    output error on every prefill row (45차 §23 GPU 판정 12차). A lane that
    buys nothing and spends the accuracy the GPTQ rounds bought is not a
    lane, so prefill is FP8 and there is one prefill form (D3).

    The K>4096 drafter projection is a fixed sequence of K tiles with FP32
    accumulation, matching the existing MK lane. Padding is weight-owned.
    """
    def __init__(self, weight, *, prefill=True, hessians=None, store=None, name=None, smooth=None):
        """`smooth` [K]: the factor `weight` was multiplied by, its input divided by (kernels/dense/smoothing) -- the
        store scales the calibration Hessian alike; the calibration files its sums in the unsmoothed domain."""
        if (weight.ndim != 2 or not weight.is_cuda or weight.dtype != torch.bfloat16
                or weight.shape[1] % 128):
            raise ValueError("dense weights must be CUDA BF16 with K aligned to 128")
        extension()
        self.rows, self.cols = weight.shape
        self.name = name
        self.smooth = smooth
        self.observer = None  # calibration.Calibration sums this layer's inputs through it (X^T X for the GPTQ packs)
        self.executed = 0  # boot proof: W4=1, FP8=2
        packs = []
        if self.cols > TILE and store is not None and store.calibrated(name):
            packs = list(store.pack_wide(weight, name, smooth=smooth))   # one GPTQ over the whole K, from the full Hessian
        elif self.cols > TILE and hessians is not None and hessians.shape == (self.cols, self.cols):
            packs = pack_w4_wide(weight, hessians)
        else:
            for start in range(0, self.cols, TILE):
                w = weight[:, start:start+TILE].contiguous()
                key = name if self.cols <= TILE else f'{name}.k{start//TILE}'
                packs.append(store.pack(w, key, smooth=None if smooth is None else smooth[start:start+TILE]) if store is not None else
                             pack_w4(w, hessian=None if hessians is None else hessians[start//TILE]))
        self.packs = tuple(_fold(packs))
        packs.clear()                     # the folded copy is the pack now; the tiles are 42 MiB of nothing
        self.calibrated = all(p.calibrated for p in self.packs)
        if prefill:
            # the FP8 lane's weights: GPTQ on the fp8 grid from the same calibration, else round-to-nearest
            fp8 = store.pack_fp8(weight, name, smooth=smooth) if (store is not None and store.calibrated(name)) else None
            self.fp8 = FP8Linear(weight, quantized=fp8, name=name)
        else:
            self.fp8 = None

    def consume_weight(self, storage):
        """Retire the source arena region into W4/FP8 views before capture."""
        from engine.modules.packed_storage import consume
        tensors=[t for p in self.packs for t in (p.data,p.scale,p.rowscale)]
        if self.fp8 is not None:
            tensors.extend(self.fp8.weight)
        owned=iter(consume(storage,tensors))
        self.packs=tuple(W4Pack(next(owned),next(owned),next(owned),p.rows,p.cols,p.calibrated) for p in self.packs)
        if self.fp8 is not None:
            self.fp8.weight=next(owned),next(owned)

    def __call__(self, x, rows_ok=None):
        """`rows_ok` [rows] bool: which rows are real -- only a calibration run reads it (the pipeline's ghost rows,
        a masked observation's positions past the committed count); the product itself covers every row."""
        if x.shape[-1] != self.cols or x.dtype != torch.bfloat16:
            raise ValueError("dense input does not match its bound weight")
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.cols)
        if self.observer is not None:
            self.observer(flat, rows_ok)
        if flat.shape[0] <= 32:
            self.executed |= 1
            if len(self.packs) == 1:
                out = w4_gemm(flat, self.packs[0])
            else:
                # tiles whose row shifts disagree cannot be one pack (see `_fold`), so they are still summed
                # here -- and each addend has already been rounded to bf16 by its own launch
                acc = None
                at = 0
                for pack in self.packs:
                    partial = w4_gemm(flat[:, at:at+pack.cols], pack).float()
                    acc = partial if acc is None else acc+partial
                    at += pack.cols
                out = acc.bfloat16()
        else:
            if self.fp8 is None:
                raise ValueError("large-M dense call without a prepared prefill lane")
            out = self.fp8(flat)
            self.executed |= 2
        return out.reshape(*shape, self.rows)


class FP8Linear:
    """Block-scaled FP8 for prefill and the accuracy-sensitive vocabulary head. `quantized`: (q, scale) prepared by
    the store -- GPTQ on the fp8 grid from the weight's calibration (packing.fp8_gptq) -- instead of round-to-nearest."""
    def __init__(self, weight, *, quantized=None, name=None):
        self.rows, self.cols = weight.shape
        self.name = name
        self.observer = None  # calibration sums this layer's inputs through it when it stands alone (the head)
        self.executed = False
        self.calibrated = quantized is not None
        padded_rows = (self.rows+127)//128*128
        if quantized is not None:
            q, scale = quantized
            if tuple(q.shape) != (padded_rows, self.cols) or tuple(scale.shape) != (padded_rows//128, self.cols//128):
                raise ValueError("prepared FP8 weights do not match the bound weight")
            self.weight = q.to(weight.device), scale.to(weight.device)
            return
        from deep_gemm import per_block_cast_to_fp8
        w = torch.nn.functional.pad(weight, (0, 0, 0, padded_rows-self.rows))
        qs, scales = [], []
        # Bound pack-time FP32 temporaries even for the vocabulary head.
        for chunk in w.split(1024):
            q, scale = per_block_cast_to_fp8(chunk.float(), use_ue8m0=True)
            qs.append(q); scales.append(scale)
        self.weight = torch.cat(qs), torch.cat(scales)

    def consume_weight(self, storage):
        from engine.modules.packed_storage import consume
        self.weight=consume(storage,self.weight)

    def __call__(self, x, rows_ok=None):
        from deep_gemm import fp8_gemm_nt
        if self.observer is not None:
            self.observer(x.reshape(-1, self.cols), rows_ok)
        from .fp8 import quantize
        from engine.kernels.deep_gemm import _initialize
        _initialize()
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.cols).contiguous()
        q, scale = quantize(flat)
        out = torch.empty((flat.shape[0], self.weight[0].shape[0]), device=x.device, dtype=torch.bfloat16)
        fp8_gemm_nt((q, scale), self.weight, out)
        self.executed = True
        return out[:, :self.rows].reshape(*shape, self.rows)
