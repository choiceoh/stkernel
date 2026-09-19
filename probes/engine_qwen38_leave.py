"""A decode site's leave behind its TP sum, three ways, on one GB10 (probe, single-GPU lane; carry H4).

On the fleet every leave follows a sublayer's sum, and the one-shot consumer that makes the sum publishes this rank's
packet, releases its programmatic dependents and then idles until the other three ranks' packets land -- about 20 us a
sum with the memory idle. engine/kernels/gated_residual.leave_norm serves the leave three ways (lanes.LEAVES):

    off        launched after the sum, as any launch is
    pdl        the sum's programmatic dependent: resident through the wait, started when the sum lands
    prefetch   pdl, and the site's down projection pulled into L2 during the wait (gated_residual.PREFETCH_BYTES of it)

One GPU has no other ranks, so a stand-in plays the sum: a launch that waits for its own predecessor, releases its
dependents at once (the consumer does so once it has published) and idles WAIT us before it writes the sum -- over an
output filled with NaN first, so a leave that read the sum before its wait could not pass the byte check. A site is
stand-in -> leave_norm -> the mixer as a decode step serves it (gated_residual.mix, two launches), 16 sites a graph over
weights rotated past the L2, the arms interleaved and reversed every round. Checks first: every arm's streams, normed
streams and mixed output byte for byte the `off` arm's, eager and replayed, behind a 3 ms wait.

    python3 probes/engine_kernel_check.py --lanes qwen38_leave
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HC, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6        # Qwen3.8's site at TP=4
ROWS = (1, 2, 4, 8, 16)                            # a draft step, K=1 and K=3 at C=1, K=3 at C=2 and C=4
WAITS_US = (0, 10, 20, 30)                         # the other ranks' packets in flight (the fleet: about 20 us)
CHECK_WAIT_US = 3000                               # long enough that an early read of the sum would read NaN
BUDGETS = (2 << 20, 4 << 20, None)                 # prefetch arms: bytes of the down projection (None: all 6.6 MB)
COPIES_BYTES = 64 << 20                            # weights rotated over at least this many bytes: more than GB10's L2
SITES = 16                                         # sites a graph
ROUNDS = 15


def gpu_busy():
    """The device's utilization as nvidia-smi reports it, or None: the lane runs beside production and anything else
    on the box, and a record says what shared the GPU while it timed (q38leave-0919a ran beside a 96% training job)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def stats(v):
    """min, p25 and median of a list of us: a replay the GPU shares with another context takes a whole time slice
    more (about 2.3 ms a 16-site graph beside that training job), so the low end is the uncontended launch."""
    v = sorted(v)
    return {"min": round(v[0], 2), "p25": round(v[len(v) // 4], 2), "median": round(statistics.median(v), 2)}


def _stand_in():
    import triton
    import triton.language as tl

    @triton.jit
    def sum_stand_in(SRC, OUT, N, WAIT_NS, BLOCK: tl.constexpr):
        # the one-shot consumer as its dependent sees it: its predecessor first, then its dependents released, then the
        # wait for the other ranks, then the sum written
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
        began = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
        now = began
        while now - began < WAIT_NS:
            now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False,
                                            pack=1)
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(OUT + i, tl.load(SRC + i, mask=i < N), mask=i < N)

    def run(src, out, wait_us):
        block = 1024
        sum_stand_in[(triton.cdiv(src.numel(), block),)](src, out, src.numel(), int(wait_us * 1000), BLOCK=block,
                                                         num_warps=4, launch_pdl=True)
        return out
    return run


def arms():
    """(name, leave mode, prefetch budget override)."""
    out = [("off", "off", None), ("pdl", "pdl", None)]
    for budget in BUDGETS:
        out.append((f"prefetch {'all' if budget is None else f'{budget >> 20} MB'}", "prefetch", budget))
    return out


def run(output=None) -> dict:
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.kernels.common import skinny_gemv
    torch.manual_seed(0)
    skinny_gemv.prepare("cuda")
    stand_in = _stand_in()
    width = HC * HIDDEN
    pair = (RANK + HC) * width * 2 + width * RANK * 2
    copies = max(SITES, -(-COPIES_BYTES // pair))
    downs = [torch.randn(RANK + HC, width, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    ups = [torch.randn(width, RANK, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    norms = [torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(copies)]
    report = {"device": torch.cuda.get_device_name(), "rounds": ROUNDS, "sites_a_graph": SITES,
              "down_MB": round((RANK + HC) * width * 2 / 1e6, 2), "pair_MB": round(pair / 1e6, 2),
              "qualify": {k: [round(x, 6) for x in v] for k, v in hcr.qualify(
                  torch.device("cuda"), hc=HC, hidden=HIDDEN, rank=RANK, eps=EPS).items()},
              "checks": {}, "rows": {}, "gpu_busy_percent": {"start": gpu_busy()}}
    print(json.dumps({"qualify": report["qualify"], "gpu_busy_percent": report["gpu_busy_percent"]}), flush=True)

    def site(i, h, src, out, inject, mode, budget, wait_us, poison=False):
        """One site: the stand-in's sum (over NaN when `poison`), the leave the mode serves, the mixer.
        -> (h, normed, mixed)."""
        if poison:
            out.fill_(float("nan"))
        stand_in(src, out, wait_us)
        hcr._PREFETCH_BYTES_OVERRIDE = budget
        try:
            h, normed = hcr.leave_norm(h, out, inject, norms[i % copies], EPS, HC, pdl=mode != "off",
                                       prefetch=downs[i % copies] if mode == "prefetch" else None)
        finally:
            hcr._PREFETCH_BYTES_OVERRIDE = None
        mixed, _ = hcr.mix(normed, downs[i % copies], ups[i % copies], HC)
        return h, normed, mixed

    for m in ROWS:
        src = [torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(SITES)]
        inject = torch.rand(m, HC, device="cuda", dtype=torch.bfloat16) * 2
        start = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)

        # checks: one site behind a 3 ms wait, eager and replayed, every arm against `off`
        outs = {}
        for name, mode, budget in arms():
            h, out = start.clone(), torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            eager = [t.clone() for t in site(0, h, src[0], out, inject, mode, budget, CHECK_WAIT_US, poison=True)]
            torch.cuda.synchronize()
            h.copy_(start)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                replayed = site(0, h, src[0], out, inject, mode, budget, CHECK_WAIT_US, poison=True)
            h.copy_(start)
            g.replay()
            torch.cuda.synchronize()
            outs[name] = (eager, [t.clone() for t in replayed])
            del g
        base_eager, base_replay = outs["off"]
        checks = {}
        for name, (eager, replayed) in outs.items():
            checks[name] = {
                "eager_equal_off": all(torch.equal(a, b) for a, b in zip(eager, base_eager)),
                "replay_equal_off": all(torch.equal(a, b) for a, b in zip(replayed, base_replay)),
                "finite": all(bool(torch.isfinite(t).all()) for t in eager + replayed)}
        report["checks"][m] = checks
        print(json.dumps({f"checks rows {m}": checks}), flush=True)

        # timings: 16 sites a graph, rotated weights, arms interleaved
        keep = []
        report["rows"][m] = {}
        for wait_us in WAITS_US:
            graphs = {}
            for name, mode, budget in arms():
                h, outs_ = start.clone(), [torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
                                          for _ in range(SITES)]
                for i in range(SITES):                                  # compile and warm outside the capture
                    h = site(i, h, src[i], outs_[i], inject, mode, budget, wait_us)[0]
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for i in range(SITES):
                        h, normed, mixed = site(i, h, src[i], outs_[i], inject, mode, budget, wait_us)
                        keep.append((normed, mixed))
                keep.append((h, outs_))
                graphs[name] = g
            times = {name: [] for name in graphs}
            order = list(graphs)
            for r in range(ROUNDS):
                for name in (order if r % 2 == 0 else order[::-1]):
                    g = graphs[name]
                    g.replay()
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter() - began) / SITES * 1e6)
            row = {name: stats(v) for name, v in times.items()}
            report["rows"][m][wait_us] = {
                "us_a_site": row, "gpu_busy_percent": gpu_busy(),
                "saved_us_a_site": {k: {q: round(row["off"][q] - v[q], 2) for q in v} for k, v in row.items()}}
            print(json.dumps({f"rows {m} wait {wait_us} us": report["rows"][m][wait_us]}), flush=True)
            del graphs
        del keep
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
