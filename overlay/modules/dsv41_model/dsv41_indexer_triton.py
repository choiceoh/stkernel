"""Opt-in compact V4.1 indexer score kernel; no quantizer or top-k replacement.

Importing this module does not initialize CUDA. `_compact_scores_kernel` is the
explicit entry for no-device Triton ASTSource compilation. It accepts strided
BF16 q/key/weights and int32 candidate IDs, and writes contiguous BF16 scores.
The default core backend remains Torch until real device differential evidence
exists. Tensor-core dot and head reduction may round differently from cuBLAS.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["WIDTH"])
def _compact_scores_kernel(
    Q, K, W, IDS, OUT,
    Q_QUERIES: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
    WIDTH, CAPACITY: tl.constexpr,
    Q_B: tl.constexpr, Q_Q: tl.constexpr, Q_H: tl.constexpr, Q_D: tl.constexpr,
    K_B: tl.constexpr, K_S: tl.constexpr, K_D: tl.constexpr,
    W_B: tl.constexpr, W_Q: tl.constexpr, W_H: tl.constexpr,
    I_B: tl.constexpr, I_Q: tl.constexpr, I_C: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    batch = (row // Q_QUERIES).to(tl.int64)
    query = (row % Q_QUERIES).to(tl.int64)
    candidates = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    selected = tl.load(IDS + batch * I_B + query * I_Q + candidates * I_C, candidates < CAPACITY, other=-1)
    valid = (candidates < CAPACITY) & (selected >= 0) & (selected < WIDTH)
    q = tl.load(
        Q + batch * Q_B + query * Q_Q + heads[:, None] * Q_H + dims[None, :] * Q_D,
        (heads[:, None] < HEADS) & (dims[None, :] < DIM), other=0,
    )
    keys = tl.load(
        K + batch * K_B + selected[None, :].to(tl.int64) * K_S + dims[:, None] * K_D,
        valid[None, :] & (dims[:, None] < DIM), other=0,
    )
    dot_fp32 = tl.dot(q, keys, out_dtype=tl.float32)
    # Mandatory eager BF16 boundaries: GEMM output, weighted product, head sum.
    dots = dot_fp32.to(tl.bfloat16).to(tl.float32)
    weights = tl.load(W + batch * W_B + query * W_Q + heads * W_H, heads < HEADS, other=0).to(tl.float32)
    # Preserve NaNs and signed zero just as the reference eager ReLU does.
    relu = tl.where(dots < 0.0, 0.0, dots)
    products = (relu * weights[:, None]).to(tl.bfloat16).to(tl.float32)
    products = tl.where(heads[:, None] < HEADS, products, 0.0)
    values = tl.sum(products, axis=0).to(tl.bfloat16)
    values = tl.where(valid, values, 0.0)
    tl.store(OUT + row.to(tl.int64) * CAPACITY + candidates, values, candidates < CAPACITY)


def compact_scores_triton(q, index_k, weights, ids):
    """Explicit device backend with the same admission as the Torch core."""
    try:
        from .dsv41_indexer import _score_contract
    except ImportError:
        from dsv41_indexer import _score_contract
    _score_contract(q, index_k, weights, ids, 1)
    if not q.is_cuda:
        raise ValueError("the Triton compact indexer requires CUDA tensors")
    batch, queries, heads, dim = q.shape
    if dim != 128 or heads not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Triton score backend requires D=128 and H in {1,2,4,8,16,32}")
    capacity = ids.shape[-1]
    output = torch.empty((batch, queries, capacity), dtype=torch.bfloat16, device=q.device)
    if capacity:
        with torch.cuda.device(q.device):
            _compact_scores_kernel[(batch * queries, triton.cdiv(capacity, 32))](
                q, index_k, weights, ids, output,
                queries, heads, dim, index_k.shape[1], capacity,
                *q.stride(), *index_k.stride(), *weights.stride(), *ids.stride(),
                max(16, triton.next_power_of_2(heads)), 128, 32,
                num_warps=4, enable_fp_fusion=False,
            )
    return output


def offline_compile(output_dir):
    """Compile H8/H32 on SM121 without querying a device or creating a context.

    The two variants use the runtime launch options and a strided two-batch KV
    slice: visible width 131073, allocation 1048576. Artifacts are the original
    PTX/cubin, with hashes and exact specialization metadata. This proves code
    generation only, never numerical correctness or performance.
    """
    import hashlib
    from pathlib import Path
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if torch.cuda.is_initialized():
        raise RuntimeError("offline_compile requires CUDA to remain uninitialized")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    signature = {"Q": "*bf16", "K": "*bf16", "W": "*bf16", "IDS": "*i32", "OUT": "*bf16", "WIDTH": "i32"}
    options = {"num_warps": 4, "enable_fp_fusion": False}
    result = []
    for heads in (8, 32):
        constants = dict(
            Q_QUERIES=1, HEADS=heads, DIM=128, CAPACITY=16384,
            Q_B=heads * 128, Q_Q=heads * 128, Q_H=128, Q_D=1,
            K_B=1048576 * 128, K_S=128, K_D=1,
            W_B=heads, W_Q=heads, W_H=1, I_B=16384, I_Q=16384, I_C=1,
            BLOCK_H=max(16, heads), BLOCK_D=128, BLOCK_C=32,
        )
        compiled = triton.compile(
            ASTSource(_compact_scores_kernel, signature, constexprs=constants),
            target=GPUTarget("cuda", 121, 32), options=options,
        )
        directory = root / f"H{heads}"
        directory.mkdir(exist_ok=False)
        artifacts = []
        for kind in ("ptx", "cubin"):
            data = compiled.asm[kind]
            if isinstance(data, str):
                data = data.encode()
            if not data:
                raise RuntimeError(f"empty {kind} compiler artifact")
            destination = directory / f"kernel.{kind}"
            destination.write_bytes(data)
            artifacts.append(dict(path=str(destination.relative_to(root)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        result.append(dict(
            entry="_compact_scores_kernel", heads=heads, signature=signature,
            constants=constants, runtime_example={"WIDTH": 131073}, options=options,
            target={"backend": "cuda", "arch": 121, "warp_size": 32},
            kernel_hash=compiled.hash, shared_bytes=compiled.metadata.shared,
            artifacts=artifacts,
        ))
    if torch.cuda.is_initialized():
        raise RuntimeError("offline compilation unexpectedly initialized CUDA")
    return result
