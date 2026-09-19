"""The MTP head's experts on one GB10: the side-file kernel (kernels/moe_rows) held to its torch form on the rank's real
weights, and every precision the head can serve its experts at held to the checkpoint's original BF16 (probe, lane).

    qualify     moe_rows.qualify at bf16 and fp8: random experts, the boot's D3 check
    bf16        the rank's own BF16 experts (mtp_side.py side file, the fleet's default): the kernel against `reference`
    fp8         the export's FP8 experts through the same kernel, against the BF16 reference
    nvfp4       the rank file's NVFP4 re-encoding on b12x (what the fleet served before 2026-09-19), against the BF16
                reference -- how far the draft layer's MoE had moved, as a number

Rows of real-scale activations routed to a few of the rank's experts each, at 1, 4 and 16 rows.

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

ROWS = (1, 4, 16)


def run(output=None, ranks=None, rank: "int | None" = None) -> dict:
    import torch
    from engine.base import kernel_shape
    from engine.kernels import moe_rows
    from engine.modules.modelopt_scales import ModelOptScales
    from engine.profiles.qwen38 import facts, mtp_side, specs
    from engine.profiles.qwen38 import lanes as lane_tables
    from engine.profiles.qwen38.fleet import rank_loader
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        rank = max(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
    kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    F = facts.load(ranks)
    report = {"device": torch.cuda.get_device_name(), "rank": rank,
              "qualify": {p: moe_rows.qualify(torch.device("cuda"), precision=p) for p in ("bf16", "fp8")}}
    print(json.dumps({"qualify": report["qualify"]}), flush=True)

    def side(precision, names):
        return rank_loader(mtp_side.path(mtp_side.DIRS[precision], rank, precision),
                           expected_layout=mtp_side.LAYOUTS[precision]).load(list(names))

    b = side("bf16", specs.MTP_BF16)
    f = side("fp8", specs.MTP_FP8)
    bf16 = (b[specs.MTP_BF16[0]], None, b[specs.MTP_BF16[1]], None)
    fp8 = tuple(f[n] for n in specs.MTP_FP8)
    nv = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout).load(list(specs.MTP_NVFP4))
    n = "mtp.L0.moe."
    lanes = lane_tables.served()
    scales = ModelOptScales.bind(*(nv[n + s] for s in ("w13_alpha", "a13_scale", "w2_alpha", "a2_scale")),
                                 experts=nv[n + "w13"].shape[0], device=nv[n + "w13"].device)
    first = F.expert_range(rank)[0]
    gen = torch.Generator().manual_seed(0)
    E, k = bf16[0].shape[0], F.topk_experts
    report["rows"] = {}
    for m in ROWS:
        x = (torch.randn(m, F.hidden, generator=gen) * 0.5).bfloat16().cuda()
        ids = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(m)]).to(torch.int32).cuda()
        weights = torch.rand(m, k, generator=gen)
        weights = (weights / weights.sum(1, keepdim=True)).cuda()
        want = moe_rows.reference(x, ids, weights, *bf16).float()
        got = {"bf16": moe_rows.moe(x, ids, weights, *bf16).float(), "fp8": moe_rows.moe(x, ids, weights, *fp8).float(),
               "nvfp4": lanes.moe(x, ids + first, weights, nv[n + "w13"], nv[n + "w13_sf"], nv[n + "w2"], nv[n + "w2_sf"],
                                  scales=scales, first_expert=first, compact=False).float()}
        peak, rms = want.abs().max().clamp_min(1e-30), want.square().mean().sqrt().clamp_min(1e-30)
        row = {name: {"max": round(float((o - want).abs().max() / peak), 6),
                      "rms": round(float((o - want).square().mean().sqrt() / rms), 6)} for name, o in got.items()}
        report["rows"][m] = row
        print(json.dumps({"rows": m, **row}), flush=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
