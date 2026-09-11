"""NVFP4 as the engine's native weight type (module). Held as on disk, sharded
as packed bytes, never unpacked at rest.

The operator's scope: Spark x4, NVFP4 only, no legacy. So the weight of a
projection IS the checkpoint's four tensors --

    weight_packed   U8      [out, in/2]     two e2m1 per byte, even element low
    weight_scale    E4M3    [out, in/16]    one scale per 16 along K
    weight_global   F32     []              one scale for the tensor
    input_global    F32     []              the activation side (W4A4)

-- and the arena holds exactly those bytes (loader views). Two checkpoint
name families spell them: compressed-tensors (GLM: weight_packed /
weight_scale / weight_global_scale / input_global_scale) and modelopt
(Qwen: weight / weight_scale / weight_scale_2 / input_scale). Same packing,
same group 16 -- and OPPOSITE global scales: modelopt's weight_scale_2 is a
multiplier (Qwen: 2.08e-4), compressed-tensors' weight_global_scale is a
DIVISOR (GLM: 1.728e4 = 448*6/amax). Read as a multiplier it makes a
projection 3e8 too large and nothing errors -- the column check even
passed, because both sides shared the mistake. `load()` inverts per family
and the self-check asserts the magnitude, not just the agreement.

TP on packed bytes: a column split takes rows of every tensor; a row split
takes columns -- in/2 of the packed weight and in/16 of the scale -- and
both are exact because 16 divides every shard boundary this fleet uses.

`forward` is the torch reference (modules/moe: dequant then GEMM). The b12x
CuTe-DSL kernel is the served lane and reports itself through proof; this
class is what that lane is judged against.
"""
from __future__ import annotations

import torch
from torch import nn

from engine.modules.linear import _Identity
from engine.modules.moe import dequant_nvfp4, quant_nvfp4_act, dequant_nvfp4_act, GROUP

NAMES = {                                    # canonical -> (compressed-tensors, modelopt)
    "packed": ("weight_packed", "weight"),
    "scale": ("weight_scale", "weight_scale"),
    "global": ("weight_global_scale", "weight_scale_2"),
    "input": ("input_global_scale", "input_scale"),
}
DIVISOR_FAMILY = {"weight_global_scale", "input_global_scale"}   # compressed-tensors: 1/x is the multiplier


class NVFP4Linear(nn.Module):
    """One projection, column- or row-parallel, as packed NVFP4."""

    def __init__(self, in_features, out_features, parallel="column", prefix="", comm=None):
        super().__init__()
        self.comm = comm or _Identity(); self.tp, self.rank = self.comm.world_size, self.comm.rank
        self.parallel, self.prefix = parallel, prefix
        if parallel == "column":
            self.out_local, self.in_local = out_features // self.tp, in_features
        elif parallel == "row":
            self.out_local, self.in_local = out_features, in_features // self.tp
        else:
            self.out_local, self.in_local = out_features, in_features
        if self.in_local % GROUP or in_features % 2:
            raise ValueError(f"{prefix}: K {in_features} must split into groups of {GROUP}")
        u8 = torch.uint8; e4 = torch.float8_e4m3fn
        self.packed = nn.Parameter(torch.empty(self.out_local, self.in_local // 2, dtype=u8), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(self.out_local, self.in_local // GROUP, dtype=e4), requires_grad=False)
        self.weight_global = nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)
        self.input_global = nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)

    def load(self, tensors: dict):
        """tensors: the four, keyed by either name family, FULL (unsharded)."""
        def pick(key):
            for name in NAMES[key]:
                if name in tensors:
                    t = tensors[name]
                    if name in DIVISOR_FAMILY:
                        t = 1.0 / t.float()
                    return t
            raise KeyError(f"{self.prefix}: no {key} among {sorted(tensors)}")
        packed, scale = pick("packed"), pick("scale")
        if self.parallel == "column":
            packed = packed.narrow(0, self.rank * self.out_local, self.out_local)
            scale = scale.narrow(0, self.rank * self.out_local, self.out_local)
        elif self.parallel == "row":
            packed = packed.narrow(1, self.rank * (self.in_local // 2), self.in_local // 2)
            scale = scale.narrow(1, self.rank * (self.in_local // GROUP), self.in_local // GROUP)
        self.packed.data.copy_(packed.view(torch.uint8)); self.scale.data.copy_(scale.view(torch.float8_e4m3fn))
        self.weight_global.data.copy_(pick("global").float().reshape(()))
        self.input_global.data.copy_(pick("input").float().reshape(()))

    def dequant(self) -> torch.Tensor:
        return dequant_nvfp4(self.packed, self.scale, self.weight_global)

    def forward(self, x, quantize_act: bool = True):
        w = self.dequant()
        if quantize_act:                        # W4A4: what the served kernel sees
            p, s = quant_nvfp4_act(x, self.input_global)
            xq = dequant_nvfp4_act(p, s, self.input_global)
        else:
            xq = x.float()
        out = (xq @ w.T).to(x.dtype)
        if self.parallel == "row" and self.tp > 1:
            out = self.comm.all_reduce(out)
        return out, None


def _selfcheck() -> None:
    import json, struct
    from pathlib import Path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path("/home/choiceoh/models/glm53-redhat-nvfp4")
    wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
    pre = "model.language_model.layers.3.mlp.experts.0.gate_proj."
    names = {k[len(pre):]: k for k in wm if k.startswith(pre)}
    tensors = {}
    for short, full in names.items():
        sh = ck / wm[full]
        with sh.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        e = h[full]; lo, hi = e["data_offsets"]
        with sh.open("rb") as f:
            f.seek(base + lo); raw = f.read(hi - lo)
        dt = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}[e["dtype"]]
        tensors[short] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt).reshape(e["shape"]).to(dev)
    out_f, in_f = tensors["weight_packed"].shape[0], tensors["weight_packed"].shape[1] * 2
    class FakeComm:
        def __init__(self, r): self.world_size, self.rank = 2, r
        def all_reduce(self, t): return t
        def all_gather(self, t, dim=-1): return t
    torch.manual_seed(0); x = torch.randn(4, in_f, device=dev, dtype=torch.bfloat16)
    with torch.device(dev):
        full = NVFP4Linear(in_f, out_f, "replicated"); full.load(tensors)
        cols = [NVFP4Linear(in_f, out_f, "column", comm=FakeComm(r)) for r in (0, 1)]
        rows = [NVFP4Linear(in_f, out_f, "row", comm=FakeComm(r)) for r in (0, 1)]
    for m in cols + rows: m.load(tensors)
    W = full.dequant()
    assert 1e-3 < W.std().item() < 0.2 and W.abs().max().item() < 2.0, f"weights must be O(1e-2): std {W.std().item():.3e}, max {W.abs().max().item():.3e}"
    ref, _ = full(x, quantize_act=False)
    col = torch.cat([c(x, quantize_act=False)[0] for c in cols], -1)
    assert torch.allclose(col.float(), ref.float(), atol=1e-2, rtol=1e-2), "column shards of packed bytes concat to the full projection"
    row = sum(r(x.narrow(-1, i * in_f // 2, in_f // 2), quantize_act=False)[0].float() for i, r in enumerate(rows))
    scale = ref.float().abs().max().item()
    assert torch.allclose(row, ref.float(), atol=2e-2 * scale, rtol=2e-2), "row shards of packed bytes + scales sum to the full projection"
    assert cols[0].packed.shape == (out_f // 2, in_f // 2) and rows[0].scale.shape == (out_f, in_f // 2 // GROUP)
    y, _ = full(x)                            # W4A4 path, finite
    assert torch.isfinite(y).all()
    print(f"  nvfp4_linear: real GLM expert gate_proj [{out_f},{in_f}] held packed; column/row TP=2 on packed bytes == full; "
          f"W4A4 finite; |W| std {full.dequant().std().item():.4f} OK")


if __name__ == "__main__":
    _selfcheck()
