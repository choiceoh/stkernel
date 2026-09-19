"""ModelOpt NVFP4's dequantisation multipliers bound to the b12x quantisation contract (module).

For each FC, ModelOpt dequantises x as q*s*a and W as q*s*w. The MMA quantisers b12x imports DIVIDE by a; their GEMM
epilogues need a*w. The direct micro backend uses reciprocal quantiser scales and normalises them in dispatch. Derived
FP32 alphas are made once, before CUDA graph capture; packed weights and E4M3 block scales remain untouched. Every
ModelOpt NVFP4 checkpoint's experts read this way (GLM-5.3's, Qwen3.8's): it is a feature of the quantisation, not of
a model, and both profiles bind through this one class (engine/profiles/glm53/modelopt_scales.py re-exports it).
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelOptScales:
    weight13: torch.Tensor
    input13: torch.Tensor
    weight2: torch.Tensor
    input2: torch.Tensor
    alpha13: torch.Tensor
    alpha2: torch.Tensor

    @classmethod
    def bind(cls, weight13, input13, weight2, input2, *, experts, device):
        values = (weight13, input13, weight2, input2)
        for value in values:
            if (value.shape != (experts,) or value.dtype != torch.float32
                    or value.device != device or not value.is_contiguous()):
                raise ValueError("ModelOpt requires contiguous per-expert FP32 scales on the weight device")
            if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise ValueError("ModelOpt scales must be positive and finite")
        derived = (weight13 * input13, weight2 * input2)
        checked = (*derived, input13.reciprocal(), input2.reciprocal())
        if any(not bool(torch.isfinite(v).all()) or not bool((v > 0).all()) for v in checked):
            raise ValueError("ModelOpt scale products/reciprocals must remain positive finite FP32")
        return cls(*values, *derived)


__all__ = ["ModelOptScales"]
