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

dist.barrier()
ext.shutdown()
dist.destroy_process_group()
