"""What folding block verification into one launch is worth, at the shape production runs.

Written as torch operations it issues about 180 device ops a call and spends 2,451 us of wall on 397 us of work
(probes/decode_middle_cost.py, ledger 45차 §68): at one sequence and K=5 most of those tensors hold five numbers,
so the cost is dispatch. engine/kernels/block_verify folds everything except the correction draw.

    docker run --rm --gpus all -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/block_verify_bench.py
"""
import sys, torch
sys.path.insert(0, "/w")
from engine.base.sampler import _block_verify_by_torch, block_verify_batch
from torch.profiler import profile, ProfilerActivity

V, C = 154880, 16
DEV = torch.device("cuda")

def timed(fn, iters=200):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); out.append(a.elapsed_time(b)*1000)
    out.sort(); return out[len(out)//2], out[0]

print(f"  {'shape':>12} {'torch ops':>14} {'one kernel':>14} {'speedup':>10}")
for n, K in ((1, 5), (4, 5)):
    T = K + 1
    g = torch.Generator(device=DEV).manual_seed(3)
    target = torch.softmax(torch.randn(n, T, V, generator=g, device=DEV, dtype=torch.float32), -1)
    cand = torch.randint(0, V, (n, K, C), generator=g, device=DEV)
    qp = torch.softmax(torch.randn(n, K, C, generator=g, device=DEV), -1)
    drafts = cand[:, :, 0].contiguous()
    gen = torch.Generator(device=DEV).manual_seed(5)
    a_med, a_min = timed(lambda: _block_verify_by_torch(target, drafts, cand, qp, gen))
    b_med, b_min = timed(lambda: block_verify_batch(target, drafts, cand, qp, gen))
    print(f"  n={n} K={K} {a_med:>10.1f}us {b_med:>11.1f}us {a_med/b_med:>9.1f}x   (min {a_min:.0f} -> {b_min:.0f})")
    if n == 1:
        call = lambda: block_verify_batch(target, drafts, cand, qp, gen)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            for _ in range(10): call()
            torch.cuda.synchronize()
        r = pr.key_averages()
        ops = sum(e.count for e in r if e.key.startswith("void ") or "kernel" in e.key.lower() or "Memcpy" in e.key)
        print(f"           device ops per call: ~{ops/10:.0f}   GPU {sum(e.self_device_time_total for e in r)/10:.0f}us"
              f"   host {sum(e.self_cpu_time_total for e in r)/10:.0f}us")
