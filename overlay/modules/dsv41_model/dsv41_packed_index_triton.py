"""Opt-in direct packed-key indexer scoring, with no full BF16 key expansion.

Dense mode shares one decoded key tile across four queries (at most 128 MMA
rows for H32). Compact mode uses one query's candidate positions. Width,
candidate/output columns, query count and strides are unspecialized runtime
integers; a growing decode context must not trigger per-token compilation.
No device is queried on import or by offline_compile. Code generation alone
does not validate actual CUDA numerics, collective equivalence or performance.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_RUNTIME_INTS = [
    "WIDTH", "OUT_COLS", "QUERIES", "Q_B", "Q_Q", "Q_H", "Q_D",
    "P_B", "P_S", "S_B", "S_S", "W_B", "W_Q", "W_H", "I_B", "I_Q", "I_C",
]


@triton.jit
def _decode_bf16_bits(code, scale):
    code = code.to(tl.int32)
    scale = scale.to(tl.int32)
    magnitude = code & 7
    exponent = scale + (magnitude >> 1) - 1
    mantissa = tl.where(magnitude > 1, (magnitude & 1) << 6, 0)
    normal = (exponent << 7) | mantissa
    shift = tl.minimum(8, tl.maximum(0, 1 - exponent))
    subnormal = (128 | mantissa) >> shift
    bits = tl.where(exponent > 0, normal, subnormal)
    bits = tl.where(exponent >= 255, 0x7F80, bits)
    bits = tl.where(magnitude == 0, 0, bits) | ((code & 8) << 12)
    bits = tl.where(scale == 255, 0x7FC0, bits)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def _packed_scores_kernel(
    Q, PACKED, SCALES, W, IDS, OUT,
    WIDTH, OUT_COLS, QUERIES,
    Q_B, Q_Q, Q_H, Q_D, P_B, P_S, S_B, S_S, W_B, W_Q, W_H, I_B, I_Q, I_C,
    HEADS: tl.constexpr, COMPACT: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    query_base = tl.program_id(1).to(tl.int64) * BLOCK_Q
    column = tl.program_id(2).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C)
    row = tl.arange(0, BLOCK_M)
    local_query = row // HEADS
    head = row % HEADS
    query = query_base + local_query
    valid_row = (local_query < BLOCK_Q) & (query < QUERIES)
    dim = tl.arange(0, 128)
    if COMPACT:
        # Compact BLOCK_Q is exactly one: different queries have different IDs.
        selected = tl.load(
            IDS + batch * I_B + query_base * I_Q + column * I_C,
            (query_base < QUERIES) & (column < OUT_COLS), other=-1,
        ).to(tl.int64)
    else:
        selected = column
    valid_key = (column < OUT_COLS) & (selected >= 0) & (selected < WIDTH)
    q = tl.load(
        Q + batch * Q_B + query[:, None] * Q_Q + head[:, None] * Q_H + dim[None, :] * Q_D,
        valid_row[:, None], other=0,
    )
    payload = tl.load(
        PACKED + batch * P_B + selected[None, :] * P_S + (dim[:, None] // 2),
        valid_key[None, :], other=0,
    )
    scale = tl.load(
        SCALES + batch * S_B + selected[None, :] * S_S + (dim[:, None] // 32),
        valid_key[None, :], other=127,
    )
    code = (payload >> ((dim[:, None] & 1) * 4)) & 15
    keys = _decode_bf16_bits(code, scale)
    dot = tl.dot(q, keys, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
    weights = tl.load(
        W + batch * W_B + query * W_Q + head * W_H,
        valid_row, other=0,
    ).to(tl.float32)
    relu = tl.where(dot < 0.0, 0.0, dot)
    product = (relu * weights[:, None]).to(tl.bfloat16).to(tl.float32)
    product = tl.where(valid_row[:, None], product, 0.0)
    grouped = product.reshape((BLOCK_M // HEADS, HEADS, BLOCK_C))
    reduced = tl.sum(grouped, axis=1).to(tl.bfloat16)
    query_out = query_base + tl.arange(0, BLOCK_M // HEADS)
    valid_out = ((tl.arange(0, BLOCK_M // HEADS) < BLOCK_Q) & (query_out < QUERIES))[:, None]
    values = tl.where(valid_key[None, :], reduced, 0.0)
    tl.store(
        OUT + (batch * QUERIES + query_out[:, None]) * OUT_COLS + column[None, :],
        values, valid_out & (column[None, :] < OUT_COLS),
    )


def packed_scores_triton(q, cache, weights, *, width, ids=None):
    """Standalone guarded entry; packed owners never leave uint8 storage."""
    try:
        from .dsv41_packed_index import _score_contract
    except ImportError:
        from dsv41_packed_index import _score_contract
    batch, queries, heads, columns = _score_contract(q, cache, weights, width, ids, 4, 256)
    if not q.is_cuda:
        raise ValueError("packed Triton scoring requires CUDA tensors")
    if heads not in (1, 2, 4, 8, 16, 32):
        raise ValueError("packed Triton scoring requires H in {1,2,4,8,16,32}")
    compact = ids is not None
    block_q = 1 if compact else 4
    block_m = max(16, block_q * heads)
    output = torch.empty((batch, queries, columns), dtype=torch.bfloat16, device=q.device)
    if columns:
        # Dense mode never dereferences IDS; reuse Q as a typed dummy pointer.
        ids_arg = ids if compact else q
        ids_strides = ids.stride() if compact else (0, 0, 0)
        with torch.cuda.device(q.device):
            _packed_scores_kernel[(batch, triton.cdiv(queries, block_q), triton.cdiv(columns, 32))](
                q, cache.packed, cache.scales, weights, ids_arg, output,
                width, columns, queries,
                *q.stride(), cache.packed.stride(0), cache.packed.stride(1),
                cache.scales.stride(0), cache.scales.stride(1), *weights.stride(), *ids_strides,
                heads, compact, block_q, block_m, 32,
                num_warps=4, enable_fp_fusion=False,
            )
    cache.validate()
    return output


def offline_compile(output_dir):
    """Emit H8/H32 x dense/compact SM121 PTX+cubin without CUDA initialization."""
    import hashlib
    from pathlib import Path
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if torch.cuda.is_initialized():
        raise RuntimeError("offline compile requires CUDA to remain uninitialized")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    options = {"num_warps": 4, "enable_fp_fusion": False}
    variants = []
    for heads in (8, 32):
        for compact in (False, True):
            label = f"H{heads}-{'compact' if compact else 'dense'}"
            signature = {"Q": "*bf16", "PACKED": "*u8", "SCALES": "*u8", "W": "*bf16",
                         "IDS": "*i32" if compact else "*bf16", "OUT": "*bf16"}
            signature.update({name: "i32" for name in _RUNTIME_INTS})
            block_q = 1 if compact else 4
            constants = {"HEADS": heads, "COMPACT": compact, "BLOCK_Q": block_q,
                         "BLOCK_M": max(16, block_q * heads), "BLOCK_C": 32}
            queries = 1 if compact else 7
            width, columns, capacity = 131073, 16384 if compact else 131073, 1048576
            example = dict(
                WIDTH=width, OUT_COLS=columns, QUERIES=queries,
                Q_B=queries * heads * 128, Q_Q=heads * 128, Q_H=128, Q_D=1,
                P_B=capacity * 64, P_S=64, S_B=capacity * 4, S_S=4,
                W_B=queries * heads, W_Q=heads, W_H=1,
                I_B=queries * columns if compact else 0,
                I_Q=columns if compact else 0, I_C=1 if compact else 0,
            )
            compiled = triton.compile(
                ASTSource(_packed_scores_kernel, signature, constexprs=constants),
                target=GPUTarget("cuda", 121, 32), options=options,
            )
            directory = root / label
            directory.mkdir(exist_ok=False)
            artifacts = []
            for kind in ("ptx", "cubin"):
                data = compiled.asm[kind]
                if isinstance(data, str):
                    data = data.encode()
                if not data:
                    raise RuntimeError(f"empty {kind} artifact")
                destination = directory / f"kernel.{kind}"
                destination.write_bytes(data)
                artifacts.append({"path": str(destination.relative_to(root)), "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()})
            variants.append(dict(
                label=label, heads=heads, compact=compact, entry="_packed_scores_kernel",
                signature=signature, constants=constants, runtime_example=example,
                options=options, target={"backend": "cuda", "arch": 121, "warp_size": 32},
                kernel_hash=compiled.hash, shared_bytes=compiled.metadata.shared, artifacts=artifacts,
            ))
    if torch.cuda.is_initialized():
        raise RuntimeError("offline compilation initialized CUDA")
    return variants
