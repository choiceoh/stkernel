"""The MTP head's experts in the export's FP8 (kernels/moe_fp8_rows) on one GB10: the kernel held to its torch form on
the rank's real side-file weights, and against the NVFP4 re-encoding the rank file serves (probe, single-GPU lane).

    qualify     moe_fp8_rows.qualify: random FP8 experts, the boot's D3 check
    real        the rank's own FP8 experts (mtp_fp8.py side file): the kernel against `reference` (the same FFN in torch,
                every expert dequantised) at 1, 4 and 16 rows of real-scale activations, routes to a few experts each
    nvfp4       the same rows through the served NVFP4 experts (b12x, the rank file's re-encoding): how far the draft
                layer's MoE moved -- the precision the side files restore, as a number

    python3 probes/engine_kernel_check.py --lanes qwen38_mtp_experts --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-mtp-experts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIDE = Path("/home/choiceoh/models/st-qwen38-mtp-fp8")
ROWS = (1, 4, 16)


def run(output=None, ranks=None, rank: "int | None" = None) -> dict:
    import torch
    from engine.base import kernel_shape
    from engine.kernels import moe_fp8_rows
    from engine.profiles.qwen38 import facts, mtp_fp8, specs
    from engine.profiles.qwen38 import lanes as lane_tables
    from engine.profiles.qwen38.fleet import rank_loader
    from engine.modules.modelopt_scales import ModelOptScales
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        rank = max(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
    kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    F = facts.load(ranks)
    report = {"device": torch.cuda.get_device_name(), "rank": rank, "qualify": moe_fp8_rows.qualify(torch.device("cuda"))}
    print(json.dumps({"qualify": report["qualify"]}), flush=True)
    side = rank_loader(mtp_fp8.path(SIDE, rank), expected_layout=mtp_fp8.LAYOUT).load(list(specs.MTP_FP8))
    w13, s13, w2, s2 = (side[n] for n in specs.MTP_FP8)
    nv = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout).load(list(specs.MTP_NVFP4))
    n = "mtp.L0.moe."
    lanes = lane_tables.served()
    scales = ModelOptScales.bind(*(nv[n + s] for s in ("w13_alpha", "a13_scale", "w2_alpha", "a2_scale")),
                                 experts=nv[n + "w13"].shape[0], device=nv[n + "w13"].device)
    gen = torch.Generator().manual_seed(0)
    E, k = w13.shape[0], F.topk_experts
    report["real"] = {}
    for m in ROWS:
        x = (torch.randn(m, F.hidden, generator=gen) * 0.5).bfloat16().cuda()
        ids = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(m)]).to(torch.int32).cuda()
        weights = torch.rand(m, k, generator=gen)
        weights = (weights / weights.sum(1, keepdim=True)).cuda()
        got = moe_fp8_rows.moe(x, ids, weights, w13, s13, w2, s2).float()
        want = moe_fp8_rows.reference(x, ids, weights, w13, s13, w2, s2).float()
        served = lanes.moe(x, ids + F.expert_range(rank)[0], weights, nv[n + "w13"], nv[n + "w13_sf"], nv[n + "w2"],
                           nv[n + "w2_sf"], scales=scales, first_expert=F.expert_range(rank)[0], compact=False).float()
        scale = want.abs().max().clamp_min(1e-30)
        row = {"vs_reference": round(float((got - want).abs().max() / scale), 6),
               "nvfp4_vs_reference": round(float((served - want).abs().max() / scale), 6),
               "nvfp4_rms_vs_reference": round(float((served - want).square().mean().sqrt() / want.square().mean().sqrt()), 6),
               "fp8_rms_vs_reference": round(float((got - want).square().mean().sqrt() / want.square().mean().sqrt()), 6)}
        report["real"][m] = row
        print(json.dumps({"rows": m, **row}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
