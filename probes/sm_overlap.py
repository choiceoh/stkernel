"""Does concurrency fill the SMs, or does the GPU just queue? One sampler call is six programs of 48 SMs.

The decode chain runs one program per row and the production shape is one sequence, so 42 SMs are idle through
it (ledger 45차 §71). Before designing a pipeline to fill them it is worth knowing whether the hardware would
even overlap independent work at this size, or whether something serialises it.

    docker run --rm --gpus all -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/sm_overlap.py

It overlaps, and it saturates exactly where the arithmetic says: four concurrent calls are 4x the work for 1.21x
the time, and eight (8 x 6 = 48 programs) is where the curve bends.
"""
import sys, torch
sys.path.insert(0, "/w")
from engine.base.sampler import rows as sampler_rows
V, M = 154880, 6
DEV = torch.device("cuda")

def make():
    g = torch.Generator(device=DEV).manual_seed(3)
    return (torch.randn(M, V, generator=g, device=DEV, dtype=torch.bfloat16),
            torch.empty(M, V, dtype=torch.float32, device=DEV),
            torch.full((M,), 0.7, device=DEV), torch.zeros(M, dtype=torch.int32, device=DEV),
            torch.full((M,), 0.95, device=DEV))

def timed(fn, iters=60):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); o.append(a.elapsed_time(b)*1000)
    o.sort(); return o[len(o)//2]

print(f"  {torch.cuda.get_device_properties(0).multi_processor_count} SMs; one call is {M} programs\n")
print(f"  {'copies':>7} {'same stream':>13} {'own streams':>13} {'overlap gain':>13}")
for k in (1, 2, 4, 8):
    args = [make() for _ in range(k)]
    streams = [torch.cuda.Stream() for _ in range(k)]
    def serial():
        for (lg, pr, t, kk, pp) in args:
            sampler_rows(lg, t, kk, pp, None, None, pr)
    def parallel():
        cur = torch.cuda.current_stream()
        for s, (lg, pr, t, kk, pp) in zip(streams, args):
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                sampler_rows(lg, t, kk, pp, None, None, pr)
        for s in streams:
            cur.wait_stream(s)
    a, b = timed(serial), timed(parallel)
    print(f"  {k:>7} {a:>11.1f}us {b:>11.1f}us {a/b:>11.2f}x")
    del args; torch.cuda.empty_cache()
