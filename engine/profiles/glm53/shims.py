"""The 17 names glm5next_* import from vLLM that are ours under another name
(profile). Each is a few lines or a rename; none is generality. With these
and engine.modules, the served model file's imports resolve without vLLM --
which is what makes the re-hosting an import swap (hosting.py is the meter).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from engine.modules.linear import ColumnParallelLinear, ReplicatedLinear
from engine.modules.logits import LogitsProcessor
from engine.modules.norm import RMSNorm


# --- model_executor.models.utils -------------------------------------------
def maybe_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def extract_layer_index(prefix: str) -> int:
    for part in reversed(prefix.split(".")):
        if part.isdigit():
            return int(part)
    raise ValueError(f"no layer index in {prefix!r}")


# --- layers.activation ------------------------------------------------------
class SiluAndMul(nn.Module):
    """[.., 2d] -> silu(x[:d]) * x[d:] -- the gated MLP's act, fused in vLLM."""
    def forward(self, x):
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


class SiluAndMulWithClamp(SiluAndMul):
    def __init__(self, limit: float = 0.0):
        super().__init__(); self.limit = limit

    def forward(self, x):
        d = x.shape[-1] // 2
        g, u = x[..., :d], x[..., d:]
        if self.limit:
            g = g.clamp(max=self.limit); u = u.clamp(-self.limit, self.limit)
        return torch.nn.functional.silu(g) * u


# --- layers.layernorm --------------------------------------------------------
LayerNorm = nn.LayerNorm


# --- quantization.utils.quant_utils -----------------------------------------
@dataclass(frozen=True)
class GroupShape:
    row: int
    col: int


def scaled_dequantize(x, x_s, group_shape: "GroupShape | None" = None, out_dtype=torch.bfloat16):
    """Blocked-scale dequant: x [M, N] fp8, x_s [M/r, N/c] -> out_dtype."""
    if group_shape is None:
        return (x.float() * x_s.float()).to(out_dtype)
    r, c = group_shape.row, group_shape.col
    s = x_s.float().repeat_interleave(r, 0)[: x.shape[0]].repeat_interleave(c, 1)[:, : x.shape[1]]
    return (x.float() * s).to(out_dtype)


# --- models.deepseek_v2 -----------------------------------------------------
def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    import math
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _get_moe_router_dtype(config) -> "torch.dtype | None":
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(getattr(config, "moe_router_dtype", None))


# --- mamba_utils --------------------------------------------------------------
def is_conv_state_dim_first() -> bool:
    return True             # the engine stores conv state as (dim, width-1): what the kernels want


class MambaStateCopyFunc:
    """A state slot copy is `base.tiered_kv`'s job; vLLM's calculator returns a callable."""
    @staticmethod
    def copy(src, dst): dst.copy_(src)


class MambaStateCopyFuncCalculator:
    @classmethod
    def gated_delta_net_state_copy_func(cls): return MambaStateCopyFunc.copy
    @classmethod
    def short_conv_state_copy_func(cls): return MambaStateCopyFunc.copy


# --- fused_moe ---------------------------------------------------------------
def fused_moe_make_expert_params_mapping(ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name,
                                         num_experts, num_redundant_experts=0):
    """(param_name, weight_name, expert_id, shard_id) for every expert projection --
    the checkpoint->parameter map the loader walks."""
    out = []
    for e in range(num_experts + num_redundant_experts):
        for shard, name in (("w1", ckpt_gate_proj_name), ("w2", ckpt_down_proj_name), ("w3", ckpt_up_proj_name)):
            out.append(("experts.w13_" if shard in ("w1", "w3") else "experts.w2_", f"experts.{e}.{name}.", e, shard))
    return out


class GateLinear(ReplicatedLinear):
    """The router: a replicated [E, hidden] projection in the router dtype."""


DenebGateLinear = GateLinear     # moe_gate_sm121 is the SM121 gate kernel lane; the module is this


# --- fp8_lm_head -------------------------------------------------------------
def decodable_vocab_size(config) -> int:
    return int(getattr(config, "vocab_size"))


class Fp8HeadLogitsProcessor(LogitsProcessor):
    """GLM's head: optionally an fp8-quantised lm_head. The fp8 path is the
    modules.quant reference; this keeps the constructor signature."""
    def __init__(self, vocab_size: int, scale: float = 1.0, fp8_env: str = "", **_):
        super().__init__(vocab_size, scale)


SHIMS = {n: globals()[n] for n in (
    "maybe_prefix", "extract_layer_index", "LayerNorm", "SiluAndMul", "SiluAndMulWithClamp",
    "GroupShape", "scaled_dequantize", "yarn_get_mscale", "is_conv_state_dim_first",
    "MambaStateCopyFunc", "MambaStateCopyFuncCalculator", "fused_moe_make_expert_params_mapping",
    "Fp8HeadLogitsProcessor", "decodable_vocab_size", "GateLinear", "DenebGateLinear", "_get_moe_router_dtype")}


def _selfcheck() -> None:
    from engine.profiles.glm53.hosting import KIND
    missing = sorted(KIND["shim"] - set(SHIMS))
    assert not missing, f"meter lists shims this file does not provide: {missing}"
    x = torch.randn(3, 8)
    assert torch.allclose(SiluAndMul()(x), torch.nn.functional.silu(x[:, :4]) * x[:, 4:])
    assert extract_layer_index("model.layers.17.self_attn") == 17 and maybe_prefix("", "x") == "x"
    fp8 = torch.randn(64, 64).to(torch.float8_e4m3fn); s = torch.rand(2, 2)
    assert scaled_dequantize(fp8, s, GroupShape(32, 32)).shape == (64, 64)
    assert abs(yarn_get_mscale(40.0, 1.0) - (0.1 * 3.6888794541139363 + 1.0)) < 1e-6
    print(f"  shims: all {len(KIND['shim'])} meter shims provided and behave OK")


if __name__ == "__main__":
    _selfcheck()
