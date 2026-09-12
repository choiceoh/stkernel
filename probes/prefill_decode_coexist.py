"""Can the decode chain ride alongside a prefill on another stream, or does it starve?

D9 says a step is pure prefill or pure decode, and its evidence (39차 DF1) measured a MIXED BATCH -- both in one
forward, which loses SP, the graphs and the megakernel. Two streams is a different shape and was never measured.
This is that shape in miniature on one GPU: a big matmul stands in for the prefill forward (it saturates the SMs),
the real sampler stands in for the decode chain (six programs of 48).
"""
import sys, threading, time, torch
sys.path.insert(0, "/w")
from engine.base.sampler import rows as sampler_rows

V, M = 154880, 6
DEV = torch.device("cuda")
print(f"  {torch.cuda.get_device_properties(0).multi_processor_count} SMs, "
      f"stream priority range {torch.cuda.Stream().priority_range() if hasattr(torch.cuda.Stream(), 'priority_range') else torch.cuda.get_stream_priority_range() if hasattr(torch.cuda,'get_stream_priority_range') else '?'}")

g = torch.Generator(device=DEV).manual_seed(3)
logits = torch.randn(M, V, generator=g, device=DEV, dtype=torch.bfloat16)
probs = torch.empty(M, V, dtype=torch.float32, device=DEV)
temps = torch.full((M,), 0.7, device=DEV); topk = torch.zeros(M, dtype=torch.int32, device=DEV)
topp = torch.full((M,), 0.95, device=DEV)
sample = lambda: sampler_rows(logits, temps, topk, topp, None, None, probs)

# a stand-in prefill: one big matmul, sized so it runs a few milliseconds
A = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
B = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
big = lambda: torch.mm(A, B)

def timed_on(fn, stream, iters=60):
    for _ in range(10):
        with torch.cuda.stream(stream): fn()
    torch.cuda.synchronize()
    o = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            a.record(stream); fn(); b.record(stream)
        torch.cuda.synchronize()
        o.append(a.elapsed_time(b)*1000)
    o.sort(); return o[len(o)//2]

cur = torch.cuda.current_stream()
print(f"\n  {'':<34}{'sampler':>11}{'big mm':>11}")
print(f"  {'alone':<34}{timed_on(sample, cur):>9.1f}us{timed_on(big, cur):>9.1f}us")

for label, prio in (("beside the matmul (same prio)", 0), ("beside the matmul (decode HIGH prio)", -1)):
    low = torch.cuda.Stream(priority=0)
    hi = torch.cuda.Stream(priority=prio)
    stop = threading.Event()
    def hammer():
        while not stop.is_set():
            with torch.cuda.stream(low):
                big()
            low.synchronize()
    t = threading.Thread(target=hammer, daemon=True); t.start()
    time.sleep(0.4)
    s = timed_on(sample, hi, iters=40)
    stop.set(); t.join(timeout=5)
    torch.cuda.synchronize()
    print(f"  {label:<34}{s:>9.1f}us")

# the other half: what the prefill pays for carrying a high-priority decode beside it
for prio in (0, -1, -3):
    hi = torch.cuda.Stream(priority=prio)
    low = torch.cuda.Stream(priority=0)
    stop = threading.Event()
    def tick():
        while not stop.is_set():
            with torch.cuda.stream(hi):
                sample()
            hi.synchronize()
    t = threading.Thread(target=tick, daemon=True); t.start()
    time.sleep(0.4)
    m = timed_on(big, low, iters=12)
    stop.set(); t.join(timeout=5); torch.cuda.synchronize()
    print(f"  matmul with a decode-priority {prio} sampler beside it: {m:.0f}us  ({m/15391.2:.2f}x solo)")
