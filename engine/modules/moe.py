"""NVFP4 (W4A4, group 16) expert weights: what the bytes mean, and a reference
GEMM that consumes them (module).

Qwen3.8's routed experts are the only NVFP4 tensors in its checkpoint
(hf_quant_config: quant_algo NVFP4, group_size 16, everything else excluded).
Per expert projection the checkpoint holds four tensors:

    weight          U8   [out, in/2]     two e2m1 values per byte
    weight_scale    E4M3 [out, in/16]    one scale per 16 elements along K
    weight_scale_2  F32  []              one global scale for the tensor
    input_scale     F32  []              the activation side's global scale

so a weight element is  e2m1(nibble) * weight_scale[row, k//16] * weight_scale_2.
The nibble order is the same as DSv4.1's fp4 (modules/quant.py): even element
low, odd element high -- and unlike DSv4.1 the scale is per ROW x 16-group,
not per [32, 32] block, which is exactly the shape difference the b12x lane's
dispatch has to get right (its IMA lives there).

This is the oracle for the MoE lane (D4/D14), slow on purpose.
"""
from __future__ import annotations

import torch

from engine.modules.quant import FP4_TABLE, _unpack_fp4, _fp4_encode, _pow2_round

GROUP = 16
FP4_MAX = 6.0
FP8_MAX = 448.0


def dequant_nvfp4(weight_u8: torch.Tensor, weight_scale: torch.Tensor,
                  weight_scale_2: torch.Tensor) -> torch.Tensor:
    """[out, in] float32 from the four-tensor NVFP4 layout (three of them)."""
    out, half = weight_u8.shape
    vals = _unpack_fp4(weight_u8.view(torch.float4_e2m1fn_x2))            # [out, in]
    scales = weight_scale.float().repeat_interleave(GROUP, dim=1)[:, : half * 2]
    return vals * scales * weight_scale_2.float()


def quant_nvfp4_act(x: torch.Tensor, input_scale: torch.Tensor):
    """W4A4's activation side: per-16 e4m3 scales under one global scale.

    Returns (packed e2m1 [.., K/2], scales e4m3 [.., K/16]). Dequantise as
    e2m1 * scale * input_scale.
    """
    k = x.size(-1)
    assert k % GROUP == 0
    z = x.float() / input_scale.float()
    blocks = z.unflatten(-1, (k // GROUP, GROUP))
    scale = (blocks.abs().amax(dim=-1) / FP4_MAX).clamp(max=FP8_MAX)
    scale = scale.to(torch.float8_e4m3fn).float().clamp_min(torch.finfo(torch.float32).tiny)
    nib = _fp4_encode((blocks / scale.unsqueeze(-1)).flatten(-2)).unflatten(-1, (k // 2, 2))
    packed = (nib[..., 0] | (nib[..., 1] << 4)).view(torch.float4_e2m1fn_x2)
    return packed, scale.to(torch.float8_e4m3fn)


def dequant_nvfp4_act(packed, scale, input_scale) -> torch.Tensor:
    vals = _unpack_fp4(packed)
    return vals * scale.float().repeat_interleave(GROUP, dim=-1) * input_scale.float()


def expert_gemm(x: torch.Tensor, weight_u8, weight_scale, weight_scale_2, input_scale=None,
                quantize_act: bool = False) -> torch.Tensor:
    """y[M, out] = x[M, in] @ W^T with W dequantised; optionally with x quantised
    the way the W4A4 kernel sees it (that is the fidelity a kernel is held to)."""
    w = dequant_nvfp4(weight_u8, weight_scale, weight_scale_2)
    if quantize_act:
        packed, s = quant_nvfp4_act(x, input_scale)
        xq = dequant_nvfp4_act(packed, s, input_scale)
    else:
        xq = x.float()
    return (xq @ w.T).to(x.dtype)


def _selfcheck() -> None:
    import json, struct
    from pathlib import Path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path("/home/choiceoh/models/qwen38-flash-next-nvfp4")
    shard = ck / "layer-00000-experts-0000-0127.safetensors"
    with shard.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    def load(name, dtype):
        e = hdr[name]; lo, hi = e["data_offsets"]
        with shard.open("rb") as f:
            f.seek(base + lo); raw = f.read(hi - lo)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(e["shape"]).to(dev)
    p = "model.language_model.layers.0.mlp.experts.0.gate_proj."
    w = load(p + "weight", torch.uint8); ws = load(p + "weight_scale", torch.float8_e4m3fn)
    ws2 = load(p + "weight_scale_2", torch.float32); xs = load(p + "input_scale", torch.float32)
    W = dequant_nvfp4(w, ws, ws2)
    ok = torch.isfinite(W).all().item() and W.shape == (640, 2560)
    print(f"  real expert 0 gate_proj: dequant {tuple(W.shape)} finite={torch.isfinite(W).all().item()} "
          f"std {W.std().item():.4f} |max| {W.abs().max().item():.4f} scale_2 {ws2.item():.3e} input_scale {xs.item():.3e}")
    assert ok and 1e-3 < W.std().item() < 1.0, "a projection's weights should be O(1e-2) and finite"
    # unique e2m1 magnitudes: a real NVFP4 tensor uses the whole table, not a corner of it
    mags = _unpack_fp4(w.view(torch.float4_e2m1fn_x2)).abs().unique().tolist()
    assert set(mags) == {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, mags
    # activation round trip inside fp4's own resolution relative to the block max
    x = torch.randn(8, 2560, device=dev) * 0.5
    packed, s = quant_nvfp4_act(x, xs)
    xq = dequant_nvfp4_act(packed, s, xs)
    rel = ((xq - x).abs() / x.abs().clamp_min(1e-2)).median().item()
    assert rel < 0.25, rel                      # e2m1 has 1 mantissa bit: coarse by design
    y = expert_gemm(x.bfloat16(), w, ws, ws2, xs, quantize_act=True)
    assert torch.isfinite(y).all() and y.shape == (8, 640)
    print(f"  nvfp4 act round trip median rel err {rel:.3f} (1 mantissa bit), W4A4 gemm finite {tuple(y.shape)} OK")


if __name__ == "__main__":
    _selfcheck()
