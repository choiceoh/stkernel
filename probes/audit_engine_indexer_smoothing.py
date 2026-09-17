"""CPU check of indexer head-gate invariance using rank weights and calibration.

Inputs are synthetic, never incident tokens. The old incomplete smoothing group
is reconstructed separately; the fixed path calls the target's actual method.
This measures a mathematical defect, not end-to-end generation quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from engine.kernels.dense.smoothing import fold, scales
from engine.profiles.glm53.net import Glm53Net, rmsnorm


def relative_l2(actual, expected):
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


def audit(rank_file, cache_root, rank):
    results = []
    with safe_open(str(rank_file), framework="pt", device="cpu") as weights:
        layers = sorted(int(k.split(".")[0][1:]) for k in weights.keys() if k.endswith(".idx.w_heads"))
        for layer in layers:
            prefix = f"L{layer}."
            name = f"Glm5NextForCausalLM/model.layers.{layer}.self_attn.fused_qkv_a_proj"
            path = cache_root / "mkcalib" / f"rank{rank}" / (name + ".pt")
            blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            amax = blob.get("amax")
            if amax is None:
                raise ValueError(f"missing channel calibration: {path}")
            net = Glm53Net.__new__(Glm53Net)
            net.F = SimpleNamespace(is_dsa=lambda _: True, is_moe=lambda _: True)
            net.layers = [layer]
            net.p = {prefix + key: weights.get_tensor(prefix + key).clone()
                     for key in ("in_norm", "mla.qkv_a", "idx.wk", "idx.gate", "idx.w_heads")}
            norm = net.p[prefix + "in_norm"].clone()
            heads = net.p[prefix + "idx.w_heads"].clone()
            old_readers = [net.p[prefix + k] for k in ("mla.qkv_a", "idx.wk", "idx.gate")]
            old_factor = scales(amax, old_readers)
            old_norm = norm.clone()
            fold(old_norm, old_factor)
            x = torch.randn(16, norm.numel(), generator=torch.Generator().manual_seed(17)).bfloat16()
            expected = rmsnorm(x, norm, 1e-6).float() @ heads.T
            omitted = rmsnorm(x, old_norm, 1e-6).float() @ heads.T
            smoothed = net.smooth_inputs(lambda key: amax if key == name else None)
            fixed = rmsnorm(x, net.p[prefix + "in_norm"], 1e-6).float() @ net.p[prefix + "idx.w_heads"].T
            torch.testing.assert_close(fixed, expected, rtol=0, atol=0)
            factor = smoothed[prefix + "mla.qkv_a"][1]
            results.append(dict(layer=layer, rank=rank, calibrated_tokens=blob.get("ntok"),
                                calibration_weights_id=blob.get("weights_id"),
                                amax_sha256=hashlib.sha256(amax.contiguous().numpy().tobytes()).hexdigest(),
                                old_scale_min=float(old_factor.min()), old_scale_max=float(old_factor.max()),
                                old_changed_channels=int((old_factor != 1).sum()),
                                changed_factors_with_head_reader=int((factor != old_factor).sum()),
                                omitted_gate_relative_l2=relative_l2(omitted, expected),
                                fixed_gate_relative_l2=relative_l2(fixed, expected),
                                fixed_gate_exact=bool(torch.equal(fixed, expected)),
                                head_dtype=str(net.p[prefix + "idx.w_heads"].dtype)))
    return dict(scope="CPU; real rank weights and calibration; synthetic inputs; no generation-quality verdict",
                cases=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_file", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.rank_file, args.cache_root, args.rank)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]),
                      "all_fixed_exact": all(row["fixed_gate_exact"] for row in result["cases"]),
                      "out": str(args.out)}))
