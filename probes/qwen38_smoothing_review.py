"""CPU feasibility check for Qwen's unit-offset norm; no serving change or quality claim."""
import argparse
import json
from pathlib import Path

import torch


def run():
    torch.manual_seed(919)
    x = torch.randn(64, 2560).bfloat16()
    w = (torch.randn(2560) * .125).bfloat16()
    weight = (torch.randn(320, 2560) * .02).bfloat16()
    factor = torch.exp2(torch.randint(-6, 7, (2560,)).float())
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    baseline = (normalized * (1 + w.float())).bfloat16()
    folded_w = ((1 + w.float()) / factor - 1).bfloat16()
    folded = (normalized * (1 + folded_w.float())).bfloat16()
    # Scale an already-rounded norm output. Power-of-two scaling is exact
    # for these finite, normal fixtures and keeps the model's BF16 boundary.
    explicit = (baseline.float() / factor).bfloat16()
    scaled_weight = (weight.float() * factor).bfloat16()
    reference = baseline.double() @ weight.double().T
    naive = folded.double() @ scaled_weight.double().T
    candidate = explicit.double() @ scaled_weight.double().T
    restored = (explicit.float() * factor).bfloat16()
    assert torch.equal(restored, baseline)
    assert torch.equal(candidate, reference)
    witness_weight, witness_scale = torch.tensor(.125).bfloat16(), 64.
    witness_new = ((1 + witness_weight.float()) / witness_scale - 1).bfloat16()
    return dict(seed=919, synthetic=True, real_checkpoint=False, dtype='bfloat16', shape=[64,2560,320],
        direct_fold_changed_norm_channels=int(((1 + folded_w.float()) * factor != 1 + w.float()).sum()),
        direct_fold_projection_relative_rmse=float(((naive-reference).square().mean()/reference.square().mean()).sqrt()),
        direct_fold_projection_max_abs=float((naive-reference).abs().max()),
        witness=dict(weight=float(witness_weight), scale=witness_scale, stored_new_weight=float(witness_new),
                     desired_gain=float(1 + witness_weight.float()), actual_gain=float((1 + witness_new.float()) * witness_scale)),
        explicit_scale_norm_roundtrip_exact=True, explicit_scale_fp64_projection_exact=True,
        limits='Finite normal synthetic values. No FP8/W4 repacking, model calibration, native kernel cost or output-quality measurement.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args=parser.parse_args(); result=run(); payload=json.dumps(result,indent=2)+'\n'
    print(payload,end='')
    if args.output: args.output.write_text(payload)
