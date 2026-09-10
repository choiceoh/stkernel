"""Opt-in dual-pool sparse attention; no prefix concatenation or cache ownership.

Each CTA handles one query and sixteen independent heads. The reference's
64-slot index order and online softmax are retained across both KV pointers.
No CUDA context is queried at import or by offline_compile. AOT is codegen
evidence only: TileLang/GPU numerical equivalence and speed need device tests.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_RUNTIME_DIMS = ["WIDTH_WINDOW", "WIDTH_COMP", "QUERIES", "TOPK"]
_RUNTIME_STRIDES = [
    "Q_B", "Q_Q", "Q_H", "Q_D", "W_B", "W_S", "W_D",
    "C_B", "C_S", "C_D", "S_H", "I_B", "I_Q", "I_K",
]
_RUNTIME_INTS = _RUNTIME_DIMS + _RUNTIME_STRIDES
_OPTIONS = {"num_warps": 8, "num_stages": 1, "enable_fp_fusion": False}
# Internal admission budget, not a claim about every SM121 device's limit.
_SHARED_BUDGET = 96 * 1024


@triton.jit(do_not_specialize=_RUNTIME_INTS + ["SCALE"])
def _dual_sparse_kernel(
    Q, WINDOW, COMPRESSED, SINK, IDS, OUT, SCALE: tl.float32,
    WIDTH_WINDOW: tl.int32, WIDTH_COMP: tl.int32, QUERIES: tl.int32, TOPK: tl.int32,
    Q_B: tl.int64, Q_Q: tl.int64, Q_H: tl.int64, Q_D: tl.int64,
    W_B: tl.int64, W_S: tl.int64, W_D: tl.int64,
    C_B: tl.int64, C_S: tl.int64, C_D: tl.int64, S_H: tl.int64,
    I_B: tl.int64, I_Q: tl.int64, I_K: tl.int64,
    HEADS: tl.constexpr, DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // QUERIES
    query = row % QUERIES
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    dim = tl.arange(0, DIM)
    lane = tl.arange(0, 64)
    q = tl.load(
        Q + batch * Q_B + query * Q_Q + heads[:, None].to(tl.int64) * Q_H
        + dim[None, :].to(tl.int64) * Q_D,
        heads[:, None] < HEADS, other=0,
    )
    maximum = tl.full((16,), -1e30, tl.float32)
    denominator = tl.full((16,), 0.0, tl.float32)
    accumulated = tl.full((16, DIM), 0.0, tl.float32)
    for start in range(0, TOPK, 64):
        slot = start + lane
        selected = tl.load(
            IDS + batch * I_B + query * I_Q + slot.to(tl.int64) * I_K,
            slot < TOPK, other=-1,
        ).to(tl.int64)
        in_window = (selected >= 0) & (selected < WIDTH_WINDOW)
        in_comp = (selected >= WIDTH_WINDOW) & (selected < WIDTH_WINDOW + WIDTH_COMP)
        # Clamp pointer arithmetic as well as masking loads. No invalid slot
        # addresses a poisoned tail, and empty pools are never dereferenced.
        window_row = tl.where(in_window, selected, 0)
        comp_row = tl.where(in_comp, selected - WIDTH_WINDOW, 0)
        window = tl.load(
            WINDOW + batch * W_B + window_row[:, None] * W_S
            + dim[None, :].to(tl.int64) * W_D,
            in_window[:, None], other=0,
        )
        compressed = tl.load(
            COMPRESSED + batch * C_B + comp_row[:, None] * C_S
            + dim[None, :].to(tl.int64) * C_D,
            in_comp[:, None], other=0,
        )
        kv = tl.where(in_window[:, None], window, compressed)
        initial = tl.where((in_window | in_comp)[None, :], 0.0, -float("inf"))
        initial = tl.broadcast_to(initial, (16, 64))
        scores = tl.dot(q, tl.trans(kv), initial, out_dtype=tl.float32) * SCALE
        previous = maximum
        maximum = tl.maximum(previous, tl.max(scores, axis=1), propagate_nan=tl.PropagateNan.ALL)
        correction = tl.exp(previous - maximum)
        probability = tl.exp(scores - maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, axis=1)
        accumulated = accumulated * correction[:, None]
        accumulated = tl.dot(probability.to(tl.bfloat16), kv, accumulated, out_dtype=tl.float32)
    sink = tl.load(SINK + heads.to(tl.int64) * S_H, heads < HEADS, other=0)
    denominator = denominator + tl.exp(sink - maximum)
    output = tl.div_rn(accumulated, denominator[:, None]).to(tl.bfloat16)
    tl.store(
        OUT + ((batch * QUERIES + query) * HEADS + heads[:, None]) * DIM + dim[None, :],
        output, heads[:, None] < HEADS,
    )


def _resource_contract(compiled):
    shared = int(compiled.metadata.shared)
    if shared < 0 or shared > _SHARED_BUDGET:
        raise RuntimeError("dual sparse compiled shared memory exceeds the 96 KiB admission budget")
    return shared


def dual_sparse_triton(q, window_kv, compressed_kv, attn_sink, topk_idxs, softmax_scale):
    """Guarded standalone CUDA entry; runtime arguments use actual pool strides."""
    try:
        from .dsv41_dual_sparse import _contract
    except ImportError:
        from dsv41_dual_sparse import _contract
    batch, queries, heads, width_window, width_comp, scale = _contract(
        q, window_kv, compressed_kv, attn_sink, topk_idxs, softmax_scale, 1,
    )
    if not q.is_cuda:
        raise ValueError("Triton dual sparse attention requires CUDA tensors")
    if batch * queries > 2**31 - 1:
        raise ValueError("flattened query count exceeds the CUDA grid-x limit")
    output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    compressed = window_kv if compressed_kv is None else compressed_kv
    args = (
        q, window_kv, compressed, attn_sink, topk_idxs, output, scale,
        width_window, width_comp, queries, topk_idxs.shape[-1],
        *q.stride(), *window_kv.stride(), *compressed.stride(), attn_sink.stride(0),
        *topk_idxs.stride(), heads, 512,
    )
    # Query count can exceed CUDA's grid-y limit during long prefill.
    grid = (batch * queries, triton.cdiv(heads, 16))
    with torch.cuda.device(q.device):
        if torch.cuda.get_device_capability(q.device) != (12, 1):
            raise ValueError("dual sparse Triton path is admitted only for SM121")
        # Compile/cache lookup without launch, so the budget is checked before
        # executing the kernel. The following launch reuses that same artifact.
        compiled = _dual_sparse_kernel.warmup(*args, grid=grid, **_OPTIONS)
        _resource_contract(compiled)
        _dual_sparse_kernel[grid](*args, **_OPTIONS)
    return output


def offline_compile(output_dir):
    """Compile H8/H16/H32/H64 for SM121 with runtime dimensions and no CUDA init."""
    import hashlib
    from pathlib import Path
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if torch.cuda.is_initialized():
        raise RuntimeError("offline compile requires CUDA to remain uninitialized")
    annotation_types = {param.name: param.annotation_type for param in _dual_sparse_kernel.params}
    expected_types = {name: "i32" for name in _RUNTIME_DIMS}
    expected_types.update({name: "i64" for name in _RUNTIME_STRIDES})
    expected_types["SCALE"] = "fp32"
    if any(annotation_types.get(name) != dtype for name, dtype in expected_types.items()):
        raise RuntimeError("Triton runtime scalar annotations do not match the offline signature")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    variants = []
    for heads in (8, 16, 32, 64):
        label = f"H{heads}"
        signature = {"Q": "*bf16", "WINDOW": "*bf16", "COMPRESSED": "*bf16",
                     "SINK": "*fp32", "IDS": "*i32", "OUT": "*bf16", "SCALE": "fp32"}
        signature.update({name: "i32" for name in _RUNTIME_DIMS})
        signature.update({name: "i64" for name in _RUNTIME_STRIDES})
        constants = {"HEADS": heads, "DIM": 512}
        capacity, queries, topk = 1048576, 131072, 640
        example = dict(
            WIDTH_WINDOW=128, WIDTH_COMP=131073, QUERIES=queries, TOPK=topk,
            Q_B=queries * heads * 512, Q_Q=heads * 512, Q_H=512, Q_D=1,
            W_B=128 * 512, W_S=512, W_D=1,
            C_B=capacity * 512, C_S=512, C_D=1, S_H=1,
            I_B=queries * topk, I_Q=topk, I_K=1,
        )
        compiled = triton.compile(
            ASTSource(_dual_sparse_kernel, signature, constexprs=constants),
            target=GPUTarget("cuda", 121, 32), options=dict(_OPTIONS),
        )
        shared = _resource_contract(compiled)
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
        # Access only already-present metadata; loading a binary just to ask
        # the CUDA driver for n_regs/spills would violate this offline gate.
        metadata = compiled.metadata._asdict()
        registers = metadata.get("n_regs", metadata.get("num_regs"))
        spills = metadata.get("n_spills", metadata.get("num_spills"))
        variants.append(dict(
            label=label, heads=heads, entry="_dual_sparse_kernel", signature=signature,
            constants=constants, runtime_example=example, options=dict(_OPTIONS),
            runtime_scalar_annotation_types=expected_types,
            grid_contract="(batch*queries, ceildiv(heads,16)); grid-x <= 2**31-1",
            target={"backend": "cuda", "arch": 121, "warp_size": 32},
            kernel_hash=compiled.hash, shared_bytes=shared, shared_budget_bytes=_SHARED_BUDGET,
            n_regs=registers, n_spills=spills,
            register_metadata_status="available" if registers is not None else "not provided without device initialization",
            spill_metadata_status="available" if spills is not None else "not provided without device initialization",
            artifacts=artifacts,
        ))
    if torch.cuda.is_initialized():
        raise RuntimeError("offline compilation initialized CUDA")
    return variants
