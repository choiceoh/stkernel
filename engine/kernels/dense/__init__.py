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

from engine.kernels.cells import DENSE_ALIGN, DENSE_KMAX, dense_glue_refusal


def flags_for(target=None):
    """The dense module's nvcc flags for a (major, minor) capability, the fleet's when unset: what `build`
    compiles with, and what a compile gate without a device reports (probes/engine_decode_native_compile.py)."""
    from engine.kernels import arch
    return ["-O2", *arch.gencode(target or arch.FLEET),
            "-DMK_GRID_DEF=96", "-DMK_MHC_GRID_DEF=144", "-DMK_NBUF2_DEF=3",
            "-DMK_FP8_PACK2_DEF=1", "-DMK_GEMM_TRANSPOSE_M8_DEF=1",
            "-DMK_GEMM_COMPACT_M8_DEF=1", "-DMK_M8_FASTPATH_DEF=1"]


@cache
def build(target=None):
    """Compile the dense lane's module when its key is new, and load it. No device is touched, so the fleet boot
    builds it before its first collective (profiles/glm53/natives); `extension` probes the device at first use.

    `target` is the (major, minor) capability to compile for, the fleet's when unset. It is an ARGUMENT and not
    something read here on purpose: this function's contract, pinned by
    tests/test_engine_glm53_natives.py, is that it consults neither the device nor the bound shape. `extension`
    already reads the shape to judge what it probed, so the decision belongs there."""
    from torch.utils.cpp_extension import load
    from engine.kernels.common.native_cache import prepare_cuda_sources
    source = Path(__file__).with_name("kernels.cu")
    flags = flags_for(target)
    root = Path(os.environ.get("ST_DENSE_BUILD_ROOT", str(Path.home()/".cache/st/dense")))
    key, directory, sources = prepare_cuda_sources(root, [source], (flags, torch.__version__, torch.version.cuda))
    return load(name="st_dense_"+key, sources=list(sources), extra_cuda_cflags=flags,
                build_directory=str(directory), verbose=False)


@cache
def extension():
    from engine.base.kernel_shape import bound
    device = bound().device
    ext = build(tuple(device.capability))
    if tuple(ext.probe_device())[:3] != (*device.capability, device.sms):
        raise RuntimeError(f"native dense lane requires GB10 SM{device.capability[0]}{device.capability[1]} "
                           f"with {device.sms} SMs")
    return ext


@dataclass(frozen=True)
class W4Pack:
    data: torch.Tensor
    scale: torch.Tensor
    rowscale: torch.Tensor
    rows: int
    cols: int
    calibrated: bool = False    # GPTQ from a calibration Hessian (the store says), not round-to-nearest


def repack_w4(pack, tile_rows=16):
    """Reorder existing bytes before capture; never requantize or retain a second pack.

    Disk caches keep their canonical 128-row format. The tensor shape identifies
    the resident layout to every native reader, including small prefill calls.
    Padding, row scales and calibration identity survive the permutation.
    """
    if tile_rows not in (16, 128):
        raise ValueError('W4 resident tiles must have 16 or 128 rows')
    if pack.rows <= 0 or pack.cols <= 0 or pack.cols % 128:
        raise ValueError('W4 dimensions must be positive with 128-aligned columns')
    q, s = pack.data, pack.scale
    padded = (pack.rows + 127) // 128 * 128
    if (q.ndim != 4 or q.shape[2] not in (16, 128)
            or tuple(q.shape) != (padded // q.shape[2], pack.cols // 128, q.shape[2], 64)
            or tuple(s.shape) != (*q.shape[:3], 8)
            or q.dtype != torch.uint8 or s.dtype != torch.int8
            or q.device != s.device or not q.is_contiguous() or not s.is_contiguous()):
        raise ValueError('W4 bytes and scales must identify the same resident tile layout')
    if q.shape[2] == tile_rows:
        return pack
    def reorder(t):
        # Swap just the eight subtiles and K blocks: one copy, no row-major scratch.
        kb, width = pack.cols // 128, t.shape[3]
        shape = (padded // 128, kb, 8, 16, width) if tile_rows == 16 else (padded // 128, 8, kb, 16, width)
        return t.view(shape).permute(0, 2, 1, 3, 4).contiguous().view(padded // tile_rows, kb, tile_rows, width)
    return W4Pack(reorder(q), reorder(s), pack.rowscale, pack.rows, pack.cols, pack.calibrated)


GPTQ_ACT_ORDER = True       # columns in decreasing Hessian-diagonal order with static groups (45차 §23 GPU 판정 7차)
TILE = 4096                 # one GPTQ tile of K: a wider weight is packed as a sequence of them
KMAX = DENSE_KMAX           # the decode kernel's widest K (kernels.cu KBLK_LIMIT): the drafter's fc, whole


def padded_columns(cols: int) -> int:
    """The input width PaddedDenseLinear packs a `cols`-wide weight at: the next multiple of DENSE_ALIGN. The arena a
    preshard reserves for such a projection is `packed_nbytes(rows, padded_columns(cols))`."""
    why = dense_glue_refusal(cols)
    if why is not None:
        raise ValueError(f"PaddedDenseLinear: {why}")
    return -(-cols // DENSE_ALIGN) * DENSE_ALIGN


def packed_nbytes(rows, cols, *, prefill=True, decode_fp8=False, decode_w4=True):
    """Aligned resident bound for the selected W4 and FP8 readers.

    Calibration may fold tiles and share their row scale, so declare the larger
    unmerged form. This does not reserve the BF16 source or packing scratch.
    """
    if rows <= 0 or cols <= 0 or cols % 128 or cols > KMAX:
        raise ValueError('dense storage requires positive rows and supported 128-aligned columns')
    padded, end = (rows + 127) // 128 * 128, 0
    def add(size):
        nonlocal end
        end = (end + 255) // 256 * 256 + size
    for start in range(0, cols if decode_w4 else 0, TILE):
        width = min(TILE, cols - start)
        add(padded * width // 2)  # two W4 values per byte
        add(padded * width // 16)  # one E4M3 scale per group
        add(padded * 4)  # FP32 row scales
    for _ in range(int(prefill) + int(decode_fp8)):
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


def wide_input_cell(rows, n, k):
    """Same-pack GPU-qualified wide decode cells; keep M14's two regressions out.

    The pack is invocation-owned, and its MMA/reduction is the ordinary W4
    program. Private-workspace shared-expert overlap retains its own route.
    See measurements/st_decode_batch_20260913 for warm and evicted B/A/A/B.
    """
    return rows in (14, 21, 28) and (
        (n, k) in ((4096, 2048), (2048, 4096), (4096, 4096), (6144, 4096), (4096, 3072))
        or (rows in (21, 28) and (n, k) in ((6416, 4096), (4096, 1536))))


def bound_input_cell(rows, n, k):
    """Candidate K=7 input reuse, explicitly bound before graph capture.

    At 16 rows the KDA input, gate/up, qkv_a and the KDA/MLA/MLP-down TX outputs run
    sixteen-row CTAs (measurements/st_c2_dense_cells_20260915, judged on chains of
    distinct layers with L2 evicted -- the serving regime). K1536 queries share their
    pack through QueryPair; M14's rejected cells stay out.
    """
    if rows == 8:
        return ((k == 4096 and n in (4096, 6144, 6416))
                or (n == 4096 and k in (2048, 3072)))
    return rows in (16, 24, 32) and (
        (n, k) in ((4096, 2048), (2048, 4096), (4096, 4096), (6144, 4096), (4096, 3072), (6416, 4096))
        or (rows in (24, 32) and (n, k) == (4096, 1536)))


def producer_pack_nbytes(rows, cols):
    """The input pack a bound cell reads when its input's producer wrote it. Eight rows: the C1 cell's own
    layout (mk_input_pack_kernel), [k/128 x 1024] FP8 words then [k/128 x 8] row scales. Sixteen rows: the
    wide pack the sixteen-row CTA reads (mk_wide_input_pack_kernel), [k/128 x 32 x 128] FP8 bytes in natural
    order then [k/128 x 32] row scales; rows 16..31 of each K block are unused."""
    blocks = cols // 128
    if cols % 128 or not blocks:
        raise ValueError('an input pack needs a positive 128-aligned width')
    if rows == 8:
        return blocks * 1024 + blocks * 8 * 4
    if rows == 16:
        return blocks * 32 * 128 + blocks * 32 * 4
    raise ValueError('producer input packs exist at 8 and 16 rows only')


def w4_gemm(x, pack, workspace=None, *, bound_input=False, producer_pack=None):
    if producer_pack is not None and not bound_input:
        raise ValueError('producer input pack requires a bound input cell')
    if (x.ndim != 2 or not 1 <= x.shape[0] <= 32 or x.shape[1] != pack.cols or x.shape[1] > KMAX
            or x.dtype != torch.bfloat16 or x.device != pack.data.device):
        raise ValueError("W4 decode requires 1..32 BF16 rows matching the bound pack, K at most 20480")
    out = torch.empty(x.shape[0], pack.rows, dtype=x.dtype, device=x.device)
    # f_a/g_a are columns of the fused KDA projection: preserve their wider
    # row stride instead of launching a copy for each of the 68 products.
    if bound_input:
        extension().run_gemm_bound_input(x, pack.data, pack.scale, out, pack.rows,
                                         pack.rowscale.data_ptr(), workspace, None, producer_pack=producer_pack)
    elif workspace is None:
        ext = extension()
        run = ext.run_gemm_wide_input if wide_input_cell(x.shape[0], pack.rows, pack.cols) else ext.run_gemm
        run(x, pack.data, pack.scale, out, pack.rows, 1., 0, pack.rowscale.data_ptr(), 0, 0, 0)
    else:
        extension().run_gemm_private(x, pack.data, pack.scale, out, pack.rows,
                                    pack.rowscale.data_ptr(), workspace)
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
    input_dtype = torch.bfloat16  # __call__ enforces this before invoking its calibration observer

    def __init__(self, weight, *, prefill=True, hessians=None, store=None, name=None, smooth=None,
                 decode_precision='w4', decode_name=None, fp8_decode_rows=False):
        """`smooth` [K]: the factor `weight` was multiplied by, its input divided by (kernels/dense/smoothing) -- the
        store scales the calibration Hessian alike; the calibration files its sums in the unsmoothed domain.
        `fp8_decode_rows`: the FP8 lane's 1..16 rows on fp8_rows' one launch (FP8Linear `decode_rows`)."""
        if (weight.ndim != 2 or not weight.is_cuda or weight.dtype != torch.bfloat16
                or weight.shape[1] % 128):
            raise ValueError("dense weights must be CUDA BF16 with K aligned to 128")
        extension()
        self.rows, self.cols = weight.shape
        if decode_precision not in ('w4', 'fp8') or (decode_precision == 'fp8' and not prefill):
            raise ValueError('FP8 decode requires a prepared FP8 pack')
        self.decode_precision = decode_precision
        self.name = name
        w4_name = decode_name if decode_name and decode_precision == 'w4' else name
        self.smooth = smooth
        self.observer = None  # calibration.Calibration sums this layer's inputs through it (X^T X for the GPTQ packs)
        self.executed = 0  # boot proof: W4=1, FP8=2
        self.workspace = None  # optional private W4 scratch for independent execution
        self.decode_input_rows = ()  # immutable candidate cells bound before capture
        self.bound_input_executed = set()
        self.producer_pack_executed = set()
        packs = []
        # The lanes below look their packs up by this weight's bytes: one hash of them serves every lane, taken beside
        # the first lane's calibration load (kernels/dense/store.WeightDigest) instead of once per lane.
        digest = None
        if store is not None and ((decode_precision != 'fp8' and (self.cols <= TILE or store.calibrated(w4_name)))
                                  or (prefill and store.calibrated(name))
                                  or (decode_name is not None and decode_precision == 'fp8')):
            digest = store.weight_digest(weight)
        if decode_precision == 'fp8':
            pass  # No W4 invocation exists: skip its packing, factorisation and resident bytes.
        elif self.cols > TILE and store is not None and store.calibrated(w4_name):
            packs = list(store.pack_wide(weight, w4_name, smooth=smooth, digest=digest))   # one GPTQ over the whole K, from the full Hessian
        elif self.cols > TILE and hessians is not None and hessians.shape == (self.cols, self.cols):
            packs = pack_w4_wide(weight, hessians)
        else:
            for start in range(0, self.cols, TILE):
                w = weight[:, start:start+TILE].contiguous()
                key = w4_name if self.cols <= TILE else f'{w4_name}.k{start//TILE}'
                shared = digest if digest is not None and digest.covers(w) else None   # a tile copy is other bytes
                packs.append(store.pack(w, key, smooth=None if smooth is None else smooth[start:start+TILE], digest=shared)
                             if store is not None else
                             pack_w4(w, hessian=None if hessians is None else hessians[start//TILE]))
        self.packs = tuple(_fold(packs))
        packs.clear()                     # the folded copy is the pack now; the tiles are 42 MiB of nothing
        self.calibrated = bool(self.packs) and all(p.calibrated for p in self.packs)
        if prefill:
            # the FP8 lane's weights: GPTQ on the fp8 grid from the same calibration, else round-to-nearest
            fp8 = (store.pack_fp8(weight, name, smooth=smooth, digest=digest)
                   if (store is not None and store.calibrated(name)) else None)
            self.fp8 = FP8Linear(weight, quantized=fp8, name=name, decode_rows=fp8_decode_rows)
        else:
            self.fp8 = None
        self.decode_fp8 = None
        if decode_name is not None and decode_precision == 'fp8':
            # Decode GPTQ must affect the executed FP8 grid, while even a
            # one-token prefill retains the shared pack. No raw BF16 reader.
            fp8 = store.pack_fp8(weight, decode_name, smooth=smooth, digest=digest) if store is not None else None
            if fp8 is None:
                raise ValueError('FP8 decode requires its completed decode calibration')
            self.decode_fp8 = FP8Linear(weight, quantized=fp8, name=decode_name)

    def consume_weight(self, storage):
        """Retire the source arena region into W4/FP8 views before capture."""
        from engine.modules.packed_storage import consume
        tensors=[t for p in self.packs for t in (p.data,p.scale,p.rowscale)]
        if self.fp8 is not None:
            tensors.extend(self.fp8.weight)
        if getattr(self, 'decode_fp8', None) is not None:
            tensors.extend(self.decode_fp8.weight)
        owned=iter(consume(storage,tensors))
        self.packs=tuple(W4Pack(next(owned),next(owned),next(owned),p.rows,p.cols,p.calibrated) for p in self.packs)
        if self.fp8 is not None:
            self.fp8.weight=next(owned),next(owned)
        if getattr(self, 'decode_fp8', None) is not None:
            self.decode_fp8.weight=next(owned),next(owned)

    def prepare_cta_layout(self):
        """Target KDA input: replace the resident W4 pack before arena relocation."""
        if (self.rows, self.cols) != (6416, 4096) or len(self.packs) != 1:
            raise ValueError('CTA weight layout is declared for the target KDA input only')
        if self.executed:
            raise RuntimeError('repack weights before execution or graph capture')
        self.packs = (repack_w4(self.packs[0]),)

    def isolate_workspace(self):
        """Before capture: permit this layer to overlap another W4 GEMM.

        One owner, one stream at a time. Counter words start at zero and are
        rearmed by every completed GEMM. Packs and arithmetic are unchanged.
        """
        if not self.packs:
            raise ValueError('a private W4 workspace requires a prepared W4 lane')
        if self.workspace is None:
            self.workspace = torch.zeros(extension().gemm_workspace_elements(), dtype=torch.float32,
                                         device=self.packs[0].data.device)
        return self.workspace.numel() * self.workspace.element_size()

    def __call__(self, x, rows_ok=None, *, observe=True, decode=False, producer_pack=None, normalization=None):
        """`rows_ok` [rows] bool: which rows are real -- only a calibration run reads it (the pipeline's ghost rows,
        a masked observation's positions past the committed count); the product itself covers every row."""
        if x.shape[-1] != self.cols or x.dtype != torch.bfloat16:
            raise ValueError("dense input does not match its bound weight")
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.cols)
        if normalization is not None and (not decode or self.decode_precision != 'fp8'):
            raise ValueError('fused dense normalization requires explicit FP8 decode')
        if producer_pack is not None and not self.input_pack_rows(flat.shape[0]):
            raise ValueError("producer input pack requires an unobserved single bound W4 cell")
        if observe and self.observer is not None:
            self.observer(flat, rows_ok)
        if flat.shape[0] <= 32 and getattr(self, 'decode_precision', 'w4') == 'w4':
            self.executed |= 1
            if len(self.packs) == 1:
                bound_input = self._bound_input(flat.shape[0], self.packs[0])
                out = w4_gemm(flat, self.packs[0], self.workspace, bound_input=bound_input,
                              **({"producer_pack": producer_pack} if producer_pack is not None else {}))
                if producer_pack is not None:
                    self.producer_pack_executed.add(flat.shape[0])
                if bound_input:
                    self.bound_input_executed.add(flat.shape[0])
            else:
                # tiles whose row shifts disagree cannot be one pack (see `_fold`), so they are still summed
                # here -- and each addend has already been rounded to bf16 by its own launch
                acc = None
                at = 0
                for pack in self.packs:
                    partial = w4_gemm(flat[:, at:at+pack.cols], pack, self.workspace).float()
                    acc = partial if acc is None else acc+partial
                    at += pack.cols
                out = acc.bfloat16()
        else:
            if self.fp8 is None:
                raise ValueError("large-M dense call without a prepared prefill lane")
            lane = self.decode_fp8 if decode and getattr(self, 'decode_fp8', None) is not None else self.fp8
            if normalization is not None:
                out = lane(flat, decode=decode, normalization=normalization)
            else:
                out = lane(flat, decode=decode) if getattr(lane, 'cublas', None) is not None else lane(flat)
            self.executed |= 2
        return out.reshape(*shape, self.rows)

    def _bound_input(self, rows, pack):
        return rows in getattr(self, 'decode_input_rows', ()) and bound_input_cell(rows, pack.rows, pack.cols)

    def input_pack_rows(self, rows):
        return (rows == 8 and self.observer is None and len(self.packs) == 1
                and getattr(self, 'decode_precision', 'w4') == 'w4'
                and self._bound_input(rows, self.packs[0]))

    def packet_projector(self):
        """The prefill transport may bypass BF16 storage only without observers."""
        if (self.cols != 4096 or self.fp8 is None or self.observer is not None
                or self.fp8.observer is not None):
            return None
        return self._project_packets

    def slot_writer(self, rows):
        """Only a single existing W4 product can write its final BF16 result."""
        if not 1 <= rows <= 32 or self.rows != 4096 or len(self.packs) != 1 or self.observer is not None:
            return None
        return self._write_slot

    def producer_pack_rows(self, rows):
        """Rows at which this direct writer reads a pack its input's producer wrote: a bound C1 cell, or at 16
        rows the KDA output's sixteen-row CTA (4096x2048), the one bound TX cell with a pack-writing producer."""
        return ((rows == 8 or (rows == 16 and self.cols == 2048)) and self.slot_writer(rows) is not None
                and self._bound_input(rows, self.packs[0]))

    def _write_slot(self, x, address, pack=None):
        if self.slot_writer(x.shape[0]) is None or x.ndim != 2 or x.shape[1] != self.cols:
            raise ValueError("unsupported direct W4 producer")
        p = self.packs[0]
        if pack is not None and not self.producer_pack_rows(x.shape[0]):
            raise ValueError("a producer pack is only a bound C1 cell's or the sixteen-row KDA output's input")
        if self._bound_input(x.shape[0], p):
            extension().run_gemm_bound_input(x, p.data, p.scale, address, p.rows,
                                             p.rowscale.data_ptr(), self.workspace, address,
                                             producer_pack=pack)
            self.bound_input_executed.add(x.shape[0])
            if pack is not None:
                self.producer_pack_executed.add(x.shape[0])
        else:
            extension().run_gemm_to_slot(x, p.data, p.scale, address, p.rows, p.rowscale.data_ptr(), self.workspace)
        self.executed |= 1

    def _project_packets(self, received, local_rows, *, real_rows=None, routed=False):
        if self.packet_projector() is None or local_rows * 4 <= 32:
            raise ValueError("packet projection requires the unobserved FP8 prefill lane")
        from engine.kernels.prefill_collectives.consumer import quantize_gather
        options = dict(routed=True) if routed else {}
        out = self.fp8.project_quantized(*quantize_gather(received, local_rows, real_rows=real_rows, **options))
        self.executed |= 2
        return out


class PaddedDenseLinear(DenseLinear):
    """DenseLinear for a weight whose input width is not DENSE_ALIGN-aligned: glue (cells.GLUE, cells.dense_glue_refusal).

    The weight gains zero columns up to the next multiple of DENSE_ALIGN before it is packed, and the input gains zeros
    at the call; rows are padded inside the pack already. Exact: a zero column adds nothing to a row's product, the W4
    row shift and the real columns' E4M3 group scales are maxima a zero never raises, and both activation quantizers
    (the W4A8 kernel's and fp8.quantize) scale by the amax of their groups. A calibration observer sees the padded input,
    so the Hessians it sums are the padded weight's -- unpadded `hessians` are refused, and a `store` calibrated for
    `name` at the unpadded width packs round-to-nearest (its `pack` keeps only a Hessian of the weight's width; the
    layer's `calibrated` says which). `consume_weight` takes the padded weight's storage, `packed_nbytes(rows,
    padded_columns(cols))` wide. The direct producers (the packet projector, the slot writer) read an input the caller
    has not widened, so they are not offered."""

    def __init__(self, weight, *, prefill=True, hessians=None, store=None, name=None, smooth=None,
                 decode_precision='w4', fp8_decode_rows=False):
        if weight.ndim != 2:
            raise ValueError("PaddedDenseLinear: a dense weight is [N, K]")
        if hessians is not None:
            raise ValueError("PaddedDenseLinear takes no unpadded Hessians: calibrate at the padded width "
                             "(its observer sees the padded input)")
        self.input_cols = weight.shape[1]
        self.pad = padded_columns(self.input_cols) - self.input_cols
        if self.pad:
            weight = torch.nn.functional.pad(weight, (0, self.pad))
            if smooth is not None:
                smooth = torch.nn.functional.pad(smooth, (0, self.pad), value=1.0)
        super().__init__(weight, prefill=prefill, store=store, name=name, smooth=smooth,
                         decode_precision=decode_precision, fp8_decode_rows=fp8_decode_rows)

    def __call__(self, x, rows_ok=None, *, observe=True):
        """x at the weight's width, or already at the padded width with zero columns (common.swiglu's `pad_to` writes
        it so: the pad is then the producer's launch, not a separate one here)."""
        if x.shape[-1] not in (self.input_cols, self.input_cols + self.pad) or x.dtype != torch.bfloat16:
            raise ValueError("dense input does not match its bound weight")
        widen = self.pad and x.shape[-1] == self.input_cols
        return super().__call__(torch.nn.functional.pad(x, (0, self.pad)) if widen else x, rows_ok, observe=observe)

    def packet_projector(self):
        return None

    def slot_writer(self, rows):
        return None


class FP8Linear:
    """Block-scaled FP8 for prefill and the accuracy-sensitive vocabulary head. `quantized`: (q, scale) prepared by
    the store -- GPTQ on the fp8 grid from the weight's calibration (packing.fp8_gptq) -- instead of round-to-nearest.

    `decode_rows`, what 1..16 rows of a decode step take (a profile opts in where it measured the shape):
      False      the reader: deep_gemm, or the cuBLASLt reader when one is prepared
      True       fp8_rows.project: one launch over block-128 FP8 rows, the same quantized inputs as deep_gemm
      "w8a16"    fp8_rows.project_bf16: one launch over the BF16 rows NOT quantised -- the weight's bytes, and the
                 product the weight's alone (no activation rounding). It serves these rows ahead of a prepared
                 cuBLASLt reader, which keeps every larger batch; `decode_rows_executed` records that it served."""
    DECODE_ROWS = (False, True, "w8a16")

    def __init__(self, weight, *, quantized=None, name=None, decode_rows=False):
        if decode_rows not in self.DECODE_ROWS:
            raise ValueError(f"FP8Linear decode_rows is one of {self.DECODE_ROWS}, not {decode_rows!r}")
        self.rows, self.cols = weight.shape
        self.name = name
        self.decode_rows = decode_rows
        self.decode_rows_executed = False
        self.observer = None  # calibration sums this layer's inputs through it when it stands alone (the head)
        self.executed = False
        self.calibrated = quantized is not None
        self.cublas = None
        padded_rows = (self.rows+127)//128*128
        if quantized is not None:
            q, scale = quantized
            if tuple(q.shape) != (padded_rows, self.cols) or tuple(scale.shape) != (padded_rows//128, self.cols//128):
                raise ValueError("prepared FP8 weights do not match the bound weight")
            # UE8M0: every reader takes these scales as powers of two (the MX32 weight's E8M0 exponents, DeepGEMM's
            # packed scales). A checkpoint's own FP32 block scales are not, and on sm_121 DeepGEMM faults or asserts
            # on them (vllm#54125, sglang#39482) -- requantize with packing.fp8_block_scales before binding.
            if not bool((torch.frexp(scale.float()).mantissa == 0.5).all()):
                raise ValueError(f"{name or 'FP8Linear'}: prepared FP8 block scales must be powers of two (UE8M0)")
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
        if self.cublas is not None:
            raise RuntimeError('relocate FP8 weights before preparing cuBLAS')
        from engine.modules.packed_storage import consume
        self.weight=consume(storage,self.weight)

    def prepare_cublas(self, *, split_decode=False, storage=None):
        from .cublaslt_serving import Reader
        if self.cublas is not None:
            raise RuntimeError('cuBLAS reader was already prepared')
        self.cublas = Reader(self.weight, split_decode=split_decode, storage=storage)

    def __call__(self, x, rows_ok=None, *, out=None, decode=False, normalization=None):
        if self.observer is not None:
            self.observer(x.reshape(-1, self.cols), rows_ok)
        from .fp8 import quantize
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.cols).contiguous()
        if normalization is not None and (self.cublas is None or self.rows != self.weight[0].shape[0]):
            raise ValueError('fused FP8 normalization requires an unpadded cuBLAS reader')
        if normalization is None and self.bf16_rows(flat):
            return self.project_bf16(flat, out=out).reshape(*shape, self.rows)
        if self.cublas is not None:
            options = {} if normalization is None else dict(normalization=normalization)
            result = self.cublas(flat, out=out, decode=decode, **options)
            self.executed = True
            return result[:, :self.rows].reshape(*shape, self.rows)
        q, scale = quantize(flat)
        return self.project_quantized(q, scale, out=out).reshape(*shape, self.rows)

    def bf16_rows(self, x) -> bool:
        """Do these rows take the W8A16 lane? `decode_rows="w8a16"`, 1..16 contiguous BF16 rows of the weight's width
        on its device, and a weight fp8_rows can read (both dimensions whole 128-blocks)."""
        if getattr(self, "decode_rows", False) != "w8a16" or x.ndim != 2 or x.dtype != torch.bfloat16                 or not x.is_contiguous():
            return False
        from . import fp8_rows
        return (1 <= x.shape[0] <= fp8_rows.MAX_ROWS and x.shape[1] == self.cols and self.cols % 128 == 0
                and self.weight[0].shape[0] % 128 == 0 and x.device == self.weight[0].device)

    def project_bf16(self, x, *, out=None):
        """The W8A16 lane: BF16 rows by the FP8 weight, one launch (`out` owns the full padded width, as the readers')."""
        from . import fp8_rows
        result = fp8_rows.project_bf16(x, self.weight, out=out)
        self.executed = self.decode_rows_executed = True
        return result[:, :self.rows]

    def qualify_decode_rows(self, *, rows=(1, 7, 16), columns=1024, producer=False, reader_band=2.0 ** -3) -> dict:
        """D3 for the W8A16 lane before a boot serves, on this layer's own weight -> {rows: {exact, reader[, mx]}}.

        exact    the lane against the FP32 product of its dequantized weight on the first `columns` outputs, as a
                 fraction of the largest magnitude; past 2^-7 (one BF16 step) raises -- a wrong scale, block or tile
                 misses by orders
        reader   against the prepared cuBLASLt reader on the same rows (and `mx`: its producer form, MX32 rows):
                 the two lanes round the rows differently, so this is a sanity band (2^-3), not a tie; it is also
                 what executes the reader's paths where the served batch never exceeds the lane's rows

        The lane is called as the kernel, not through `project_bf16`, so `decode_rows_executed` still records
        only what served."""
        if getattr(self, "decode_rows", False) != "w8a16":
            return {}
        from . import fp8_rows, mxfp8
        wq, ws = self.weight
        n = min(columns, wq.shape[0]) // 128 * 128
        exact_weight = wq[:n].float() * ws[:n // 128].repeat_interleave(128, 0).repeat_interleave(128, 1)
        gen = torch.Generator(device="cpu").manual_seed(0)
        report = {}
        for m in rows:
            x = torch.randn(m, self.cols, generator=gen).to(torch.bfloat16).to(wq.device)
            got = fp8_rows.project_bf16(x, self.weight).float()
            ref = x.float() @ exact_weight.t()
            exact = float((got[:, :n] - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
            if not exact <= 2.0 ** -7:
                raise RuntimeError(f"{self.name}: the W8A16 lane at {m} rows is {exact:.2e} of the largest magnitude "
                                   f"from its exact product")
            row = {"exact": round(exact, 6)}
            if self.cublas is not None:
                forms = {"reader": lambda: self.cublas(x)}
                if producer:
                    forms["mx"] = lambda: self.cublas.project_mx(*mxfp8.quantize(x, num_warps=1))
                for form, run in forms.items():
                    other = run().float()
                    drift = float((got - other).abs().max() / other.abs().max().clamp_min(1e-30))
                    if not drift <= reader_band:
                        raise RuntimeError(f"{self.name}: the W8A16 lane at {m} rows is {drift:.2e} of the largest "
                                           f"magnitude from the cuBLASLt {form} -- beyond rounding")
                    row[form] = round(drift, 6)
            report[m] = row
        return report

    def project_mx(self, hidden, q, scale, *, out=None):
        """Consume the head producer, retaining the BF16 calibration boundary. Rows the W8A16 lane takes read
        `hidden` -- the producer's BF16 rows -- and leave its MX rows unread."""
        if self.bf16_rows(hidden):
            if self.observer is not None:
                self.observer(hidden, None)
            return self.project_bf16(hidden, out=out)
        if self.cublas is None:
            raise RuntimeError('native MX input requires a prepared cuBLAS reader')
        if (hidden.shape != q.shape or hidden.ndim != 2 or hidden.shape[1] != self.cols
                or hidden.dtype != torch.bfloat16 or hidden.device != q.device
                or not hidden.is_contiguous()):
            raise ValueError('head producer hidden rows must match its FP8 input')
        if self.observer is not None:
            self.observer(hidden, None)
        result = self.cublas.project_mx(q, scale, out=out)
        self.executed = True
        return result[:, :self.rows]

    def project_quantized(self, q, scale, *, out=None):
        """Consume the existing FP8 recipe; `out` owns the full padded GEMM output."""
        if self.cublas is not None:
            result = self.cublas.project_quantized(q, scale, out=out)
            self.executed = True
            return result[:, :self.rows]
        if (q.ndim != 2 or q.shape[1] != self.cols or q.dtype != torch.float8_e4m3fn
                or scale.shape != (q.shape[0], self.cols // 128) or scale.dtype != torch.float32
                or not q.is_contiguous() or not scale.is_contiguous()
                or q.device != self.weight[0].device or scale.device != q.device):
            raise ValueError("FP8 activation bytes/scales must match the bound weight")
        shape = (q.shape[0], self.weight[0].shape[0])
        if out is None:
            out = torch.empty(shape, device=q.device, dtype=torch.bfloat16)
        elif (tuple(out.shape) != shape or out.dtype != torch.bfloat16 or out.device != q.device
              or not out.is_contiguous() or out.data_ptr() % 16):
            raise ValueError("FP8 output must be aligned contiguous BF16 with the full padded weight width")
        from . import fp8_rows
        if self.decode_rows and q.shape[0] <= fp8_rows.MAX_ROWS:
            fp8_rows.project(q, scale, self.weight, out=out)
        else:
            from deep_gemm import fp8_gemm_nt
            from engine.kernels.deep_gemm import _initialize
            _initialize()
            fp8_gemm_nt((q, scale), self.weight, out)
        self.executed = True
        return out[:, :self.rows]
