"""Synthetic router projection comparison; does not load or change a served model."""
import json
import statistics
from pathlib import Path

import torch

from engine.kernels.glm_pointwise import router_logits, route_weights
from engine.kernels.prefill_router import router_logits as prefill_logits


def timed(fn, iterations):
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    result = dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  torch=torch.__version__, synthetic=True, exclusive_gpu=False,
                  bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                  method="CUDA events, median of three alternating rounds; activation cast included except cached-x arm",
                  rows=[])
    try:
        torch.manual_seed(91523)
        gate = (torch.randn(288, 4096, device="cuda") * .02).bfloat16()
        resident = gate.float()
        bias = torch.randn(288, device="cuda") * .1
        assert torch.equal(resident.bfloat16(), gate)
        for rows in (1, 7, 8, 28, 64, 2304, 9216):
            x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
            xf = x.float()
            reference = x[:64].double() @ resident.double().T
            native = (lambda: prefill_logits(x, gate)) if rows > 8192 else (lambda: router_logits(x, gate))
            arms = {
                "bf16_fp32_accum": (False, native),
                "fp32_resident_ieee": (False, lambda: x.float() @ resident.T),
                "fp32_resident_ieee_cached_x": (False, lambda: xf @ resident.T),
                "fp32_resident_tf32": (True, lambda: x.float() @ resident.T),
            }
            outputs, costs = {}, {name: [] for name in arms}
            for name, (tf32, fn) in arms.items():
                torch.backends.cuda.matmul.allow_tf32 = tf32
                outputs[name] = fn()
            for round_ in range(3):
                names = list(arms)
                if round_ % 2:
                    names.reverse()
                for name in names:
                    tf32, fn = arms[name]
                    torch.backends.cuda.matmul.allow_tf32 = tf32
                    costs[name].append(timed(fn, 20 if rows < 2304 else 8))
            base_ids, base_weights = route_weights(outputs["bf16_fp32_accum"], bias, 8, 2.5)
            record = dict(rows=rows, oracle_rows=min(rows, 64), arms={})
            for name, value in outputs.items():
                ids, weights = route_weights(value, bias, 8, 2.5)
                error = value[:64].double() - reference
                record["arms"][name] = dict(
                    microseconds=statistics.median(costs[name]), rounds_us=costs[name],
                    fp64_relative_l2=(error.norm()/reference.norm()).item(),
                    fp64_max_abs=error.abs().max().item(),
                    expert_set_different_rows=(ids.sort(-1).values != base_ids.sort(-1).values).any(-1).sum().item(),
                    expert_order_different_rows=(ids != base_ids).any(-1).sum().item(),
                    max_route_weight_gap=(weights-base_weights).abs().max().item())
            result["rows"].append(record)
            print(json.dumps(record), flush=True)
            del x, xf, outputs, reference
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
    Path(__file__).with_name("router_precision.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
