"""One row of the sampler costs what forty-eight cost: the grid is (rows,), so a decode step uses six SMs of 48.

    docker run --rm --gpus all -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/sampler_occupancy.py
"""
import sys

import torch

sys.path.insert(0, "/w")
from engine.base.sampler import rows as sampler_rows   # noqa: E402

V = 154880
DEV = torch.device("cuda")


def timed(fn, iters=120, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000)
    out.sort()
    return out[len(out) // 2]


def main():
    print(f"  {torch.cuda.get_device_properties(0).multi_processor_count} SMs, V={V:,}")
    print(f"  {'rows':>5} {'median':>11} {'per row':>10} {'vs 6 rows':>11}")
    base = None
    for M in (1, 6, 12, 24, 48, 96):
        g = torch.Generator(device=DEV).manual_seed(3)
        logits = torch.randn(M, V, generator=g, device=DEV, dtype=torch.bfloat16)
        probs = torch.empty(M, V, dtype=torch.float32, device=DEV)
        temps = torch.full((M,), 0.7, device=DEV)
        topk = torch.zeros(M, dtype=torch.int32, device=DEV)
        topp = torch.full((M,), 0.95, device=DEV)
        t = timed(lambda: sampler_rows(logits, temps, topk, topp, None, None, probs))
        base = t if M == 6 else base
        print(f"  {M:>5} {t:>9.1f}us {t / M:>8.1f}us {t / base:>10.2f}x")
        del logits, probs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
