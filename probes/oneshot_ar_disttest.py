import os
import time

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

RANK = int(os.environ["OSAR_RANK"])
WORLD = 4
IPS = ["10.10.10.2", "10.10.10.3", "10.10.10.1", "10.10.10.4"]

torch.cuda.set_device(0)
dist.init_process_group(
    "gloo", init_method="tcp://10.10.10.2:29556", rank=RANK, world_size=WORLD)

ext = load(name="dsv4_oneshot_ar", sources=["/w/dsv4_oneshot_ar.cu"],
           extra_cuda_cflags=["-O2", "-arch=sm_121a"],
           extra_ldflags=["-libverbs"], build_directory="/w/build",
           verbose=False)

ext.init(RANK, WORLD, IPS[RANK])
gathered = [None] * WORLD
dist.all_gather_object(gathered, ext.local_infos())
ext.connect(gathered)
dist.barrier()
print(f"rank{RANK} connected", flush=True)

# correctness: each rank sends (rank+1); AllReduce SUM must be 1+2+3+4 = 10
for M in (6, 16, 32):
    x = torch.full((M, 4096), float(RANK + 1), dtype=torch.bfloat16,
                   device="cuda")
    out = ext.oneshot_ar(x)
    torch.cuda.synchronize()
    err = (out.float() - 10.0).abs().max().item()
    print(f"rank{RANK} M={M} maxerr={err:.4f} (expect 0)", flush=True)
    dist.barrier()

# latency under CUDA graph (8 calls/graph, matches the prototype)
x = torch.full((6, 4096), float(RANK + 1), dtype=torch.bfloat16, device="cuda")
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(20):
        ext.oneshot_ar(x)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
dist.barrier()

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(8):
        out = ext.oneshot_ar(x)
torch.cuda.synchronize()
for _ in range(50):
    g.replay()
torch.cuda.synchronize()
dist.barrier()
t0 = time.time()
REP = 2000
for _ in range(REP):
    g.replay()
torch.cuda.synchronize()
us = (time.time() - t0) / (REP * 8) * 1e6
print(f"rank{RANK} GRAPH one-shot AR: {us:.2f} us/call | healthy={ext.healthy()}",
      flush=True)


def _wait_us(c0, c1):
    return (c1[2] - c0[2]) / max(1, c1[4] - c0[4]) / 1592.0  # SM_CLK_MHZ


# --- the peer wait as L2 prefetch time: a 12 MB hint (one KDA in_proj W4
# pack's worth) against none, same graph shape, per-call us and the kernel's
# own t_wait for both. A hint that costs the wait more than it warms is a
# loss; the consumer-side gain is measured by moe_decode_stream_probe.py
# (gemm cold vs L2-warm) and the step by the fleet bracket.
x8 = torch.full((8, 4096), float(RANK + 1), dtype=torch.bfloat16, device="cuda")
hint = torch.empty(12 << 20, dtype=torch.uint8, device="cuda")
flush = torch.empty(48 << 20, dtype=torch.uint8, device="cuda")
for label, ptrs, lens in (("no hint", [], []),
                          ("12 MB hint", [hint.data_ptr()], [hint.numel()])):
    gh = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gh):
        for _ in range(8):
            out = ext.oneshot_ar_hint(x8, ptrs, lens)
    torch.cuda.synchronize()
    for _ in range(20):
        gh.replay()
    torch.cuda.synchronize()
    err = (out.float() - 10.0).abs().max().item()
    dist.barrier()
    c0 = ext.phase_counters()
    t0 = time.time()
    for _ in range(500):
        flush.zero_()   # the hint must be cold every time, as in a step
        gh.replay()
    torch.cuda.synchronize()
    dt = time.time() - t0
    c1 = ext.phase_counters()
    # the flush (48 MB at ~200 GB/s, ~250 us) is inside the loop; report the
    # kernel's own wait, which excludes it, beside the wall time that does not
    print(f"rank{RANK} {label:>11}: wall {dt / (500 * 8) * 1e6:.1f} us/call "
          f"(incl. flush) | t_wait {_wait_us(c0, c1):.1f} us/call | "
          f"maxerr={err:.4f} (expect 0)", flush=True)
    dist.barrier()

dist.barrier()
ext.shutdown()
dist.destroy_process_group()
