"""Explicit SM121 packet-load and BF16 MMA layouts for the long-prefill router."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def _packet_bf16x4(values, scales):
    # Four contiguous bytes enter one register; two BF16 pairs leave in two.
    # Reuse two FP32 temporaries and preserve the transport's multiply + RN,
    # including subnormal BF16 values. This is not exponent-only arithmetic.
    return gl.inline_asm_elementwise("""{
        .reg .b16 lo, hi, h0, h1;
        .reg .b32 halfs;
        .reg .f32 f0, f1;
        mov.b32 {lo, hi}, $2;
        cvt.rn.f16x2.e4m3x2 halfs, lo;
        mov.b32 {h0, h1}, halfs;
        cvt.f32.f16 f0, h0;
        cvt.f32.f16 f1, h1;
        mul.f32 f0, f0, $3;
        mul.f32 f1, f1, $4;
        cvt.rn.bf16x2.f32 $0, f1, f0;
        cvt.rn.f16x2.e4m3x2 halfs, hi;
        mov.b32 {h0, h1}, halfs;
        cvt.f32.f16 f0, h0;
        cvt.f32.f16 f1, h1;
        mul.f32 f0, f0, $5;
        mul.f32 f1, f1, $6;
        cvt.rn.bf16x2.f32 $1, f1, f0;
    }""", constraints='=r,=r,r,f,f,f,f', args=[values, scales],
        dtype=gl.bfloat16, is_pure=True, pack=4)


@gluon.jit(do_not_specialize=['M', 'LOCAL_ROWS', 'PACKET_BYTES'])
def _router_packet_gemm(Packed, Scales, W, Out, M, LOCAL_ROWS, PACKET_BYTES,
                        BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, NATIVE: gl.constexpr=False,
                        PACKED_CONVERT: gl.constexpr=False, PREFETCH: gl.constexpr=False):
    # Every lane reads adjacent K values. Transport-scale addressing must not
    # turn a coalesced activation tile into scalar loads along the row axis.
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0],
        warps_per_cta=[2, 2], instr_shape=[16, 8])
    if NATIVE:
        memory_a: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
        memory_b: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    else:
        memory_a: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
        memory_b: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [1, 4], [0, 1])
    rows = (gl.program_id(0)//gl.cdiv(288, BN))*BM + gl.arange(0, BM, layout=gl.SliceLayout(1, memory_a))
    cols = (gl.program_id(0)%gl.cdiv(288, BN))*BN + gl.arange(0, BN, layout=gl.SliceLayout(0, memory_b))
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, memory_a))
    kb = gl.arange(0, BK, layout=gl.SliceLayout(1, memory_b))
    rank, local_row = rows//LOCAL_ROWS, rows%LOCAL_ROWS
    scale_ptr = Scales + rank*(PACKET_BYTES//4) + LOCAL_ROWS*1024 + local_row*2
    scale0 = gl.load(scale_ptr, rows < M, other=0.)
    scale1 = gl.load(scale_ptr + 1, rows < M, other=0.)
    input_ptr = Packed + rank[:,None]*PACKET_BYTES + local_row[:,None]*4096 + ka[None,:]
    weight_ptr = W + cols[None,:]*4096 + kb[:,None]
    acc = gl.full((BM, BN), 0., gl.float32, mma)
    if PREFETCH:
        raw = gl.load(input_ptr, rows[:,None] < M, other=0.)
        weights = gl.load(weight_ptr, cols[None,:] < 288, other=0.)
    for block in range(4096//BK):
        if PREFETCH:
            next_raw = gl.load(input_ptr + (block+1)*BK,
                (rows[:,None] < M) & (block+1 < 4096//BK), other=0.)
            next_weights = gl.load(weight_ptr + (block+1)*BK,
                (cols[None,:] < 288) & (block+1 < 4096//BK), other=0.)
        else:
            raw = gl.load(input_ptr + block*BK, rows[:,None] < M, other=0.)
            weights = gl.load(weight_ptr + block*BK, cols[None,:] < 288, other=0.)
        scale = gl.where(block < 2048//BK, scale0, scale1)
        if PACKED_CONVERT:
            a = _packet_bf16x4(raw, scale[:,None])
        else:
            a = (raw.to(gl.float32)*scale[:,None]).to(gl.bfloat16)
        # Pin both operands to the ordinary router's kWidth=2 and K order.
        # FP8 source width cannot select a different accumulation here.
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 2))
        b = gl.convert_layout(weights, gl.DotOperandLayout(1, mma, 2))
        acc = mma_v2(a, b, acc)
        if PREFETCH:
            raw, weights = next_raw, next_weights
    output: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    rr = (gl.program_id(0)//gl.cdiv(288, BN))*BM + gl.arange(0, BM, layout=gl.SliceLayout(1, output))
    cc = (gl.program_id(0)%gl.cdiv(288, BN))*BN + gl.arange(0, BN, layout=gl.SliceLayout(0, output))
    gl.store(Out + rr[:,None]*288 + cc[None,:], gl.convert_layout(acc, output),
             (rr[:,None] < M) & (cc[None,:] < 288))
