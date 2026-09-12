"""DFlash block attention directly over its circular, grouped-query KV cache.

One query/head per CTA was the first form: it never expands KV heads, never gathers the ring and never
materialises scores, and at TP=1 its 6 x 32 CTAs filled GB10. At TP=4 the rank keeps eight query heads over two
KV heads, so the grid is 6 x 8 = 48 CTAs -- and twenty-four of them scan the SAME ring. A layer read 50.5 MiB
of a 2.1 MiB cache and took 145 us for it (45차 §84).

`_attend` reads each KV head once instead: one CTA per (KV head, slice of the window), holding every query of
that head -- rows and the group together -- as one tile. The window is cut so that KV heads x slices fills the
machine, and `_combine` folds the slices' partial softmaxes into the answer. The block's own keys, which are
not in the ring yet, ride in the last slice exactly as they rode in the tail of the old loop.

Position is a device scalar so the same graph handles ring wrap and startup.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attend(Q, K, V, R, P, Slot, ACC, MAX, DEN, SLOT_STRIDE: tl.constexpr, LAYER_OFFSET: tl.constexpr,
            B: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, RHK: tl.constexpr, D: tl.constexpr,
            W: tl.constexpr, RS: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr,
            SPAN: tl.constexpr, TILES: tl.constexpr, BQ: tl.constexpr):
    """One row's KV head, one slice of the window, every query of that head at once.

    The mask is the same for every query -- the block attends over its own keys without a causal step -- so a
    slice's keys are read once and hit a whole tile of dot products. The tile is bounded: B * (H // HK) is 28
    at TP=4 but 1024 where one KV head serves them all, and an accumulator that wide does not fit in a CTA.

    The step's rows ride the grid. They used to be a python loop -- one launch a (layer, row) and a `cat` to
    put the answers back together -- which is the same disease `write_draft_kv_rows` cured for the ring."""
    row = tl.program_id(0)
    kh = tl.program_id(1) // TILES
    tile = tl.program_id(1) % TILES
    part = tl.program_id(2)
    if SLOT_STRIDE:
        R += tl.load(Slot + row).to(tl.int64) * SLOT_STRIDE + LAYER_OFFSET
    Q += row * B * H * D
    K += row * B * HK * D
    V += row * B * HK * D
    group = H // HK
    qi = tile * BQ + tl.arange(0, BQ)
    d = tl.arange(0, D)
    live = qi < B * group
    q = tl.load(Q + ((qi // group) * H + kh * group + qi % group)[:, None] * D + d[None, :],
                live[:, None], other=0.0)                                        # [BQ, D]
    position = tl.load(P + row)
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    denominator = tl.zeros((BQ,), tl.float32)
    accumulator = tl.zeros((BQ, D), tl.float32)
    for start in range(SPAN // BN):
        n = part * SPAN + start * BN + tl.arange(0, BN)
        context = n < W
        absolute = position - W + n
        valid = (n < W + B) & (~context | (absolute >= 0))
        slot = (absolute + W) % W
        kr = tl.load(R + (slot[:, None] * RHK + kh) * D + d[None, :], context[:, None] & valid[:, None], other=0)
        kb = tl.load(K + ((n[:, None] - W) * HK + kh) * D + d[None, :], ~context[:, None] & valid[:, None], other=0)
        key = tl.where(context[:, None], kr, kb)
        score = tl.dot(q, tl.trans(key), out_dtype=tl.float32) * SCALE           # [BQ, BN]
        score = tl.where(valid[None, :], score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        # A slice that is entirely past the end must not turn -inf - -inf into NaN.
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        correction = tl.exp(maximum - safe_max)
        probability = tl.exp(score - safe_max[:, None])
        vr = tl.load(R + RS + (slot[:, None] * RHK + kh) * D + d[None, :], context[:, None] & valid[:, None], other=0)
        vb = tl.load(V + ((n[:, None] - W) * HK + kh) * D + d[None, :], ~context[:, None] & valid[:, None], other=0)
        value = tl.where(context[:, None], vr, vb).to(tl.float32)
        # the weights stay in fp32 through the value product, as they did when this was a per-query sum: the
        # drafter's acceptance is read off these, and bf16 weights cost a percent of the output vector
        accumulator = accumulator * correction[:, None] + tl.dot(probability, value, input_precision="ieee")
        denominator = denominator * correction + tl.sum(probability, 1)
        maximum = new_max
    at = ((row * tl.num_programs(1) + tl.program_id(1)) * tl.num_programs(2) + part) * BQ + tl.arange(0, BQ)
    tl.store(ACC + at[:, None] * D + d[None, :], accumulator, live[:, None])
    tl.store(MAX + at, maximum, live)
    tl.store(DEN + at, denominator, live)


@triton.jit
def _combine(ACC, MAX, DEN, O, B: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             PARTS: tl.constexpr, BP: tl.constexpr, TILES: tl.constexpr, BQ: tl.constexpr):
    """The slices' partial softmaxes into one answer, one program per (row, block position, head)."""
    row, pos, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group = H // HK
    kh = head // group
    which = pos * group + head % group
    part = tl.arange(0, BP)
    live = part < PARTS
    at = ((row * HK * TILES + kh * TILES + which // BQ) * PARTS + part) * BQ + which % BQ
    maxima = tl.load(MAX + at, live, other=-float("inf"))
    whole = tl.max(maxima, 0)
    whole = tl.where(whole == -float("inf"), 0.0, whole)
    weight = tl.where(live, tl.exp(maxima - whole), 0.0)
    d = tl.arange(0, D)
    total = tl.sum(tl.load(ACC + at[:, None] * D + d[None, :], live[:, None], other=0.0) * weight[:, None], 0)
    tl.store(O + ((row * B + pos) * H + head) * D + d, total / tl.sum(tl.load(DEN + at, live, other=0.0) * weight, 0))


def draft_attention(q, k, v, ring, position, *, slot=None, layer=0):
    """One block: BF16 [B,H,128], [B,HK,128], ring [2,W,HK,128] or the arena field with a slot -> [B,H,128]."""
    if isinstance(position, int):
        if position < 0:
            raise ValueError("negative DFlash position")
        position = torch.tensor(position, device=q.device, dtype=torch.int64)
    if q.ndim != 3:
        raise ValueError("one block is [B, H, 128]; a step's rows go through attend_rows")
    return attend_rows(q[None], k[None], v[None], ring, position.reshape(1),
                       slot=slot, layer=layer)[0]


def attend_rows(q, k, v, ring, positions, *, slot=None, layer=0):
    """Every row of a step at once: q [n, B, H, 128], k/v [n, B, HK, 128], positions [n], slots [n].

    One launch a layer, not one a (layer, row). The rows are independent -- each reads its own slot's ring at
    its own context length -- so they are a grid dimension, and the answers do not have to be concatenated
    back together afterwards."""
    if slot is not None:
        if (ring.ndim != 6 or not 0 <= layer < ring.shape[1] or slot.ndim != 1 or slot.numel() != q.shape[0]
                or slot.dtype != torch.int64 or slot.device != q.device):
            raise ValueError("DFlash arena requires a device slot per row and an in-range layer")
        geometry = ring.shape[2:]
        stride, offset = ring.stride(0), layer * ring.stride(1)
    else:
        if q.shape[0] != 1:
            raise ValueError("a bare ring belongs to one row; the arena field carries the rest")
        geometry = ring.shape
        stride, offset = 0, 0
    if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or len(geometry) != 4
            or q.shape[:2] != k.shape[:2] or q.shape[3] != 128 or k.shape[3] != 128
            or geometry[0] != 2 or geometry[2] < k.shape[2] or geometry[3] != k.shape[3]
            or q.shape[2] % k.shape[2] or not 1 <= q.shape[1] <= 32
            or not all(t.is_cuda and t.device == q.device and t.dtype == torch.bfloat16
                       for t in (q, k, v, ring))
            or not all(t.is_contiguous() for t in (q, k, v))
            or not (ring[0].is_contiguous() if slot is not None else ring.is_contiguous())):
        raise ValueError("DFlash attention requires contiguous CUDA BF16 blocks and KV ring")
    # the values are not read here: a context length is a device number and looking at it would synchronise,
    # which is not allowed while a graph is capturing
    if positions.device != q.device or positions.dtype != torch.int64 or positions.numel() != q.shape[0]:
        raise ValueError("DFlash positions must be one CUDA int64 context length per row")
    out = torch.empty_like(q)
    n, b, h, d = q.shape
    hk, cells = k.shape[2], geometry[1]
    BN, SMS = 32, 48
    # Cut the window so that (rows x KV heads x slices) fills the machine: below that the slices are wider,
    # above it they are one block each and the combine grows for nothing.
    parts = max(1, min(triton.cdiv(cells + b, BN), SMS // (n * hk)))
    span = triton.cdiv(triton.cdiv(cells + b, BN), parts) * BN
    parts = triton.cdiv(cells + b, span)
    BQ = 32                                     # a CTA's query tile: wider spills the fp32 accumulator
    tiles = triton.cdiv(b * (h // hk), BQ)
    held = n * hk * tiles * parts * BQ
    acc = torch.empty(held, d, device=q.device, dtype=torch.float32)
    scale = torch.empty(2, held, device=q.device, dtype=torch.float32)
    _attend[(n, hk * tiles, parts)](q, k, v, ring, positions,
                                    slot if slot is not None else positions,
                                    acc, scale[0], scale[1], stride, offset, b, h, hk, geometry[2], d,
                                    cells, cells*geometry[2]*geometry[3], d**-.5, BN, span, tiles, BQ,
                                    num_warps=4, enable_fp_fusion=False)
    _combine[(n, b, h)](acc, scale[0], scale[1], out, b, h, hk, d,
                        parts, triton.next_power_of_2(parts), tiles, BQ, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _write_kv(K,V,R,Slot,Pos,Valid,HAS_VALID:tl.constexpr,N:tl.constexpr,WIDTH:tl.constexpr,ROW:tl.constexpr,W:tl.constexpr,
              STRIDE:tl.constexpr,OFFSET:tl.constexpr,BLOCK:tl.constexpr):
    token=tl.program_id(0)
    d=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    slot=tl.load(Slot).to(tl.int64)
    pos=tl.load(Pos+token).to(tl.int64)%W
    base=R+slot*STRIDE+OFFSET+pos*ROW+d
    accepted = token < tl.load(Valid) if HAS_VALID else True
    mask=(d<WIDTH)&accepted
    tl.store(base,tl.load(K+token*WIDTH+d,mask,other=0),mask)
    tl.store(base+W*ROW,tl.load(V+token*WIDTH+d,mask,other=0),mask)


@triton.jit
def _write_kv_rows(K,V,R,Slot,Pos,Valid,sKr,sKt,sVr,sVt,T:tl.constexpr,WIDTH:tl.constexpr,ROW:tl.constexpr,
                   W:tl.constexpr,STRIDE:tl.constexpr,OFFSET:tl.constexpr,BLOCK:tl.constexpr):
    """`_write_kv` for every row of the step at once. Same store, one program per (row, token) instead of one
    LAUNCH per row: the single-slot form made `observe_rows` issue layers x rows kernels a step, which is the
    same disease the commit and the block verifier had.

    K and V are read where they lie. The values arrive as one layer of a [n, t, layers, 2, kv, D] projection,
    so making them contiguous first is a copy of the whole block for every layer -- and the kernel reads each
    element once anyway."""
    row=tl.program_id(0)
    token=tl.program_id(1)
    d=tl.program_id(2)*BLOCK+tl.arange(0,BLOCK)
    slot=tl.load(Slot+row).to(tl.int64)
    pos=tl.load(Pos+row*T+token).to(tl.int64)%W
    base=R+slot*STRIDE+OFFSET+pos*ROW+d
    mask=(d<WIDTH)&(token<tl.load(Valid+row))
    tl.store(base,tl.load(K+row*sKr+token*sKt+d,mask,other=0),mask)
    tl.store(base+W*ROW,tl.load(V+row*sVr+token*sVt+d,mask,other=0),mask)


def write_draft_kv_rows(field,slots,layer,positions,k,v,*,valid):
    """The step's whole block into the rings: slots [n], positions [n, t], k/v [n, t, kv, D], valid [n].

    One launch a layer instead of one a (layer, row). `observe_rows` is 21% of a decode step in production
    (45차: forward 63.5%, observe 21.4%, propose 15.1%), and its fast path was looping rows because the
    single-slot kernel required `slot.numel() == 1`.
    """
    n,t=positions.shape
    if (field.ndim!=6 or k.shape!=v.shape or k.shape[:2]!=(n,t) or k.shape[-1]!=field.shape[-1]
            or k.shape[-2]>field.shape[-2] or t>field.shape[3] or slots.numel()!=n
            or slots.dtype!=torch.int64 or positions.dtype!=torch.int64 or valid.numel()!=n
            or valid.dtype!=torch.int64 or not 0<=layer<field.shape[1]):
        raise ValueError("invalid batched DFlash ring write")
    width=k.shape[-1]*k.shape[-2]
    if k.stride(-1)!=1 or v.stride(-1)!=1 or k.stride(-2)!=k.shape[-1] or v.stride(-2)!=v.shape[-1]:
        k,v=k.contiguous(),v.contiguous()                 # the heads of a cell must be one run to be one read
    _write_kv_rows[(n,t,triton.cdiv(width,256))](
        k,v,field,slots.contiguous(),positions.contiguous(),valid.contiguous(),
        k.stride(0),k.stride(1),v.stride(0),v.stride(1),
        t,width,field.shape[-1]*field.shape[-2],field.shape[3],
        field.stride(0),layer*field.stride(1),256)


def write_draft_kv(field,slot,layer,positions,k,v,*,valid=None):
    """Write only accepted positions in the arena, without copying a whole ring."""
    if (field.ndim!=6 or k.shape!=v.shape or k.shape[-1]!=field.shape[-1] or k.shape[-2]>field.shape[-2]
            or positions.numel()!=k.shape[0] or k.shape[0]>field.shape[3]
            or slot.numel()!=1 or slot.dtype!=torch.int64 or positions.dtype!=torch.int64
            or not 0<=layer<field.shape[1]):
        raise ValueError("invalid direct DFlash ring write")
    if valid is not None and (valid.numel()!=1 or valid.dtype!=torch.int64 or valid.device!=field.device):
        raise ValueError("accepted DFlash count must be an int64 device scalar")
    width=k.shape[-1]*k.shape[-2]
    _write_kv[(k.shape[0],triton.cdiv(width,256))](
        k.contiguous(),v.contiguous(),field,slot,positions,valid if valid is not None else slot,valid is not None,
        k.shape[0],width,field.shape[-1]*field.shape[-2],field.shape[3],
        field.stride(0),layer*field.stride(1),256)
