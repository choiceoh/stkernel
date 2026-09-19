"""The ranks' packets folded by the leave after a sum, on one GB10 through the one-shot oracle (probe, single-GPU lane;
carry H5).

tests/test_engine_direct_producer_cuda's oracle builds the transport's own kernels over mapped memory and a CPU thread
stands in for the NIC and the three peers. Two parts:

    checks   every rank 0..3, rows 1, 4 and 16 at Qwen3.8's width: distinct peer packets whose sum cancels (a fold out
             of rank order moves it); the consumer's sum and the leave after it against the packets and the leave that
             folds them -- ordinary, PDL and PDL with the prefetch -- byte for byte, eager and replayed across the ring's
             wrap, the peers landing 3 ms late behind the PDL arms
    timing   the GPU side alone, peers landed ahead (engine_oneshot_consumer_timing's method): a chain of producer ->
             sum -> leave, the producer a PDL copy that releases the sum early; four arms, B/A/A/B, warm and after a
             128 MiB eviction. No RDMA and no peer skew are in these numbers

    python3 probes/engine_kernel_check.py --lanes qwen38_rank_packets
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HC, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6
ROWS = (1, 4, 16)
LATE_S = 0.003
CHAIN, REPLAYS = 8, 32
PRODUCER_US, SM_MHZ = 40, 1592
CANCEL = ((2.0 ** 24, 256.0, 1.0, -1.0), (-2.0 ** 24, -256.0, 2.0, 1.0), (1.0, 2.0 ** -16, 3.0, 2.0 ** -24),
          (1.0, 2.0 ** -16, 4.0, -2.0 ** -24))


def ranks_of(rows, generator):
    import torch
    values = torch.randn(4, rows, HIDDEN, device="cuda", generator=generator)
    values[:, :, :4] = torch.tensor(CANCEL, device="cuda")[:, None, :]
    return values.bfloat16()


def rank_ordered(values):
    acc = values[0].float() + values[1].float()
    acc = acc + values[2].float()
    return (acc + values[3].float()).bfloat16()


class Peers:
    """The NIC and the three peers: every sequence this rank publishes gets the next queued (packets, delay)."""
    def __init__(self, ext, host):
        self.ext, self.host, self.queue, self.errors = ext, host, queue.Queue(), []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def put(self, values, rank, delay=0.0):
        """The next sequence's peers: `values` [4, rows, H] BF16, every rank but `rank` as bytes in rank order."""
        import torch
        others = [r for r in range(4) if r != rank]
        self.queue.put((values[others].contiguous().view(torch.uint8).reshape(3, -1).cpu(), delay))

    def run(self):
        last = 0
        while not self.stop.is_set():
            published = self.ext.published(self.host)
            if published <= last:
                time.sleep(0.00005)
                continue
            for sequence in range(last + 1, published + 1):
                try:
                    packets, delay = self.queue.get(timeout=5)
                    if delay:
                        time.sleep(delay)
                    self.ext.land_peers(self.host, sequence, packets)
                except BaseException as error:
                    self.errors.append(repr(error))
                    self.ext.land(self.host, sequence)
            last = published

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)


def checks(report) -> dict:
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.kernels.mapped_staging import allocate
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    generator = torch.Generator(device="cuda").manual_seed(1265)
    width = HC * HIDDEN
    weight = (torch.randn(RANK + HC, width, device="cuda", generator=generator) * 0.02).bfloat16()
    arms = {"ordinary": dict(), "pdl": dict(pdl=True), "pdl+prefetch": dict(pdl=True, prefetch=weight)}
    out = {}
    for rank in range(4):
        host, device = allocate(ext.bytes())
        ext.prepare(host, device, rank, 0, 0)
        peers = Peers(ext, host)
        graphs = []
        try:
            for rows in ROWS:
                values = ranks_of(rows, generator)
                x = values[rank].clone()
                h = torch.randn(rows, width, device="cuda", generator=generator).bfloat16()
                inject = (torch.rand(rows, HC, device="cuda", generator=generator) * 2).bfloat16()
                w = (torch.randn(width, device="cuda", generator=generator) * 0.1).bfloat16()
                peers.put(values, rank)
                reduced = ext.oneshot_ar_consumer(x)
                want = hcr.leave_norm(h.clone(), reduced, inject, w, EPS, HC)
                torch.cuda.synchronize()
                row = {"consumer_rank_ordered": bool(torch.equal(reduced, rank_ordered(values)))}
                for name, kw in arms.items():
                    peers.put(values, rank, LATE_S if kw.get("pdl") else 0.0)
                    got = hcr.leave_norm(h.clone(), x, inject, w, EPS, HC, packets=ext.oneshot_packets(x), **kw)
                    torch.cuda.synchronize()
                    row[f"eager {name}"] = all(torch.equal(a, b) for a, b in zip(got, want))
                # replayed: one capture, new own and peer packets each replay, past the ring's four slots
                hbuf, xbuf = h.clone(), x.clone()
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with torch.cuda.graph(graph):
                    captured = hcr.leave_norm(hbuf, xbuf, inject, w, EPS, HC, packets=ext.oneshot_packets(xbuf),
                                              pdl=True, prefetch=weight)
                replays = []
                for _ in range(6):
                    values = ranks_of(rows, generator)
                    xbuf.copy_(values[rank])
                    hbuf.copy_(h)
                    peers.put(values, rank, LATE_S)
                    graph.replay()
                    torch.cuda.synchronize()
                    got = [t.clone() for t in captured]
                    peers.put(values, rank)
                    want = hcr.leave_norm(h.clone(), ext.oneshot_ar_consumer(values[rank].clone()), inject, w, EPS, HC)
                    torch.cuda.synchronize()
                    replays.append(all(torch.equal(a, b) for a, b in zip(got, want)))
                row["replayed pdl+prefetch"] = all(replays)
                row["tickets"] = ext.tickets(host) == ext.published(host) * 48
                out[f"rank {rank} rows {rows}"] = row
                report({f"checks rank {rank} rows {rows}": row})
        finally:
            torch.cuda.synchronize()
            for graph in graphs:
                graph.reset()
            peers.close()
        if peers.errors:
            raise RuntimeError(f"the oracle's peers failed: {peers.errors[:3]}")
    failed = [k for k, row in out.items() if not all(row.values())]
    if failed:
        raise RuntimeError(f"rank packets differ from the consumer's sum: {failed}")
    return out


def timing(report) -> list:
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.kernels.mapped_staging import allocate
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    host, device = allocate(ext.bytes())
    rank = 1
    ext.prepare(host, device, rank, 0, 0)
    cold = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(1266)
    width, cycles = HC * HIDDEN, PRODUCER_US * SM_MHZ
    rows_out = []
    for rows in ROWS:
        values = ranks_of(rows, generator)
        staged = values[rank].clone()
        peers = values[[r for r in range(4) if r != rank]].contiguous().view(torch.uint8).view(3, -1).cpu()
        x = torch.empty_like(staged)
        h = torch.randn(rows, width, device="cuda", generator=generator).bfloat16() * 0.01
        inject = (torch.rand(rows, HC, device="cuda", generator=generator) * 2).bfloat16()
        w = (torch.randn(width, device="cuda", generator=generator) * 0.1).bfloat16()
        arms = {
            "consumer, leave": lambda: hcr.leave_norm(h, ext.oneshot_ar_consumer(x), inject, w, EPS, HC),
            "consumer, pdl leave": lambda: hcr.leave_norm(h, ext.oneshot_ar_consumer(x), inject, w, EPS, HC, pdl=True),
            "packets, pdl leave": lambda: hcr.leave_norm(h, x, inject, w, EPS, HC, pdl=True,
                                                          packets=ext.oneshot_packets(x)),
            "packets, leave": lambda: hcr.leave_norm(h, x, inject, w, EPS, HC, packets=ext.oneshot_packets(x)),
        }
        for cache in ("warm", "evicted"):
            graphs = {}
            try:
                for name, step in arms.items():
                    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))

                    def chain(step=step, start=start, end=end):
                        if cache == "evicted":
                            cold.fill_(19)
                        start.record()
                        for _ in range(CHAIN):
                            ext.staged_copy(staged, x, cycles)
                            step()
                        end.record()

                    torch.cuda.synchronize()
                    ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                    chain()                                                    # compile and warm, eagerly
                    torch.cuda.synchronize()
                    ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        chain()
                    graphs[name] = graph, start, end
                samples = {name: [] for name in arms}
                order = list(arms)
                for block in range(4):
                    for name in (order if block % 2 == 0 else order[::-1]):
                        graph, start, end = graphs[name]
                        for _ in range(REPLAYS // 4):
                            torch.cuda.synchronize()
                            ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                            graph.replay()
                            end.synchronize()
                            samples[name].append(start.elapsed_time(end) * 1000.0 / CHAIN)
                row = {"rows": rows, "cache": cache, "chain": CHAIN, "producer_us": PRODUCER_US,
                       "us_a_sum": {name: {"median": round(median(v), 2), "min": round(min(v), 2)}
                                    for name, v in samples.items()}}
                base = row["us_a_sum"]["consumer, leave"]["median"]
                row["saved_us_median"] = {name: round(base - v["median"], 2) for name, v in row["us_a_sum"].items()}
                report({"timing": row})
                rows_out.append(row)
            finally:
                torch.cuda.synchronize()
                for graph, *_ in graphs.values():
                    graph.reset()
    if ext.tickets(host) != ext.published(host) * 48:
        raise RuntimeError("publication tickets drifted during timing")
    return rows_out


def moe_checks(report) -> dict:
    """Carry X2: the packet grid computing Qwen3.8's gated MoE output into TX (`moe_gated_packets`) and the leave
    folding it, against moe_output.gated_sum, the consumer's sum and the leave after it -- every rank, rows 1, 4, 16,
    eager and replayed, peers late behind the PDL leave; the TX payload itself byte for byte the finalizer's output."""
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.kernels import moe_output
    from engine.kernels.mapped_staging import allocate
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    generator = torch.Generator(device="cuda").manual_seed(904)
    width = HC * HIDDEN
    out = {}
    for rank in range(4):
        host, device = allocate(ext.bytes())
        ext.prepare(host, device, rank, 0, 0)
        peers = Peers(ext, host)
        graphs = []
        try:
            for rows in ROWS:
                def operands():
                    routed = torch.randn(rows, HIDDEN, device="cuda", generator=generator).bfloat16()
                    shared = torch.randn(rows, HIDDEN, device="cuda", generator=generator).bfloat16()
                    gate = torch.rand(rows, device="cuda", generator=generator)
                    values = ranks_of(rows, generator)
                    values[rank] = moe_output.gated_sum(routed, shared, gate[:, None])
                    return routed, shared, gate, values
                h = torch.randn(rows, width, device="cuda", generator=generator).bfloat16()
                inject = (torch.rand(rows, HC, device="cuda", generator=generator) * 2).bfloat16()
                w = (torch.randn(width, device="cuda", generator=generator) * 0.1).bfloat16()
                routed, shared, gate, values = operands()
                peers.put(values, rank)
                want = hcr.leave_norm(h.clone(), ext.oneshot_ar_consumer(values[rank].clone()), inject, w, EPS, HC)
                torch.cuda.synchronize()
                row = {}
                for name, kw in (("ordinary", {}), ("pdl", dict(pdl=True))):
                    peers.put(values, rank, LATE_S if kw else 0.0)
                    descriptor = ext.moe_gated_packets(routed, shared, gate)
                    got = hcr.leave_norm(h.clone(), shared, inject, w, EPS, HC, packets=descriptor, **kw)
                    torch.cuda.synchronize()
                    row[f"eager {name}"] = all(torch.equal(a, b) for a, b in zip(got, want))
                    tx = ext.payload(host, ext.published(host))
                    row[f"tx {name}"] = bool(torch.equal(tx, values[rank].contiguous().view(torch.uint8).flatten().cpu()))
                hbuf, rbuf, sbuf, gbuf = h.clone(), routed.clone(), shared.clone(), gate.clone()
                graph = torch.cuda.CUDAGraph()
                graphs.append(graph)
                with torch.cuda.graph(graph):
                    captured = hcr.leave_norm(hbuf, sbuf, inject, w, EPS, HC, pdl=True,
                                              packets=ext.moe_gated_packets(rbuf, sbuf, gbuf))
                replays = []
                for _ in range(6):
                    routed, shared, gate, values = operands()
                    rbuf.copy_(routed)
                    sbuf.copy_(shared)
                    gbuf.copy_(gate)
                    hbuf.copy_(h)
                    peers.put(values, rank, LATE_S)
                    graph.replay()
                    torch.cuda.synchronize()
                    got = [t.clone() for t in captured]
                    peers.put(values, rank)
                    want = hcr.leave_norm(h.clone(), ext.oneshot_ar_consumer(values[rank].clone()), inject, w, EPS,
                                          HC)
                    torch.cuda.synchronize()
                    replays.append(all(torch.equal(a, b) for a, b in zip(got, want)))
                row["replayed pdl"] = all(replays)
                row["tickets"] = ext.tickets(host) == ext.published(host) * 48
                out[f"rank {rank} rows {rows}"] = row
                report({f"moe checks rank {rank} rows {rows}": row})
        finally:
            torch.cuda.synchronize()
            for graph in graphs:
                graph.reset()
            peers.close()
        if peers.errors:
            raise RuntimeError(f"the oracle's peers failed: {peers.errors[:3]}")
    failed = [k for k, row in out.items() if not all(row.values())]
    if failed:
        raise RuntimeError(f"gated MoE packets differ from the finalizer and the consumer's sum: {failed}")
    return out


def moe_timing(report) -> list:
    """The GPU side of a MoE layer's output after its experts, peers landed ahead: gated_sum then the consumer then the
    leave (today), gated_sum then the packets then the folding leave (H5), the gated packets then the folding leave
    (X2); every leave a PDL dependent (H4). The producer is a PDL copy standing in for the routed experts."""
    import torch
    from engine.kernels import gated_residual as hcr
    from engine.kernels import moe_output
    from engine.kernels.mapped_staging import allocate
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    host, device = allocate(ext.bytes())
    rank = 1
    ext.prepare(host, device, rank, 0, 0)
    cold = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(906)
    width, cycles = HC * HIDDEN, PRODUCER_US * SM_MHZ
    rows_out = []
    for rows in ROWS:
        staged = torch.randn(rows, HIDDEN, device="cuda", generator=generator).bfloat16()
        routed = torch.empty_like(staged)
        shared = torch.randn(rows, HIDDEN, device="cuda", generator=generator).bfloat16()
        gate = torch.rand(rows, device="cuda", generator=generator)
        gate2 = gate[:, None].contiguous()
        peers = ranks_of(rows, generator)[[0, 2, 3]].contiguous().view(torch.uint8).reshape(3, -1).cpu()
        h = torch.randn(rows, width, device="cuda", generator=generator).bfloat16() * 0.01
        inject = (torch.rand(rows, HC, device="cuda", generator=generator) * 2).bfloat16()
        w = (torch.randn(width, device="cuda", generator=generator) * 0.1).bfloat16()
        arms = {
            "gated_sum, consumer, leave": lambda: hcr.leave_norm(
                h, ext.oneshot_ar_consumer(moe_output.gated_sum(routed, shared, gate2)), inject, w, EPS, HC, pdl=True),
            "gated_sum, packets, leave": lambda: hcr.leave_norm(
                h, shared, inject, w, EPS, HC, pdl=True,
                packets=ext.oneshot_packets(moe_output.gated_sum(routed, shared, gate2))),
            "gated packets, leave": lambda: hcr.leave_norm(
                h, shared, inject, w, EPS, HC, pdl=True, packets=ext.moe_gated_packets(routed, shared, gate)),
        }
        for cache in ("warm", "evicted"):
            graphs = {}
            try:
                for name, step in arms.items():
                    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))

                    def chain(step=step, start=start, end=end):
                        if cache == "evicted":
                            cold.fill_(19)
                        start.record()
                        for _ in range(CHAIN):
                            ext.staged_copy(staged, routed, cycles)
                            step()
                        end.record()

                    torch.cuda.synchronize()
                    ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                    chain()
                    torch.cuda.synchronize()
                    ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        chain()
                    graphs[name] = graph, start, end
                samples = {name: [] for name in arms}
                order = list(arms)
                for block in range(4):
                    for name in (order if block % 2 == 0 else order[::-1]):
                        graph, start, end = graphs[name]
                        for _ in range(REPLAYS // 4):
                            torch.cuda.synchronize()
                            ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                            graph.replay()
                            end.synchronize()
                            samples[name].append(start.elapsed_time(end) * 1000.0 / CHAIN)
                row = {"rows": rows, "cache": cache, "chain": CHAIN, "producer_us": PRODUCER_US,
                       "us_a_layer": {name: {"median": round(median(v), 2), "min": round(min(v), 2)}
                                      for name, v in samples.items()}}
                base = row["us_a_layer"]["gated_sum, consumer, leave"]["median"]
                row["saved_us_median"] = {name: round(base - v["median"], 2) for name, v in row["us_a_layer"].items()}
                report({"moe timing": row})
                rows_out.append(row)
            finally:
                torch.cuda.synchronize()
                for graph, *_ in graphs.values():
                    graph.reset()
    if ext.tickets(host) != ext.published(host) * 48:
        raise RuntimeError("publication tickets drifted during timing")
    return rows_out


def run_moe(output=None) -> dict:
    report = lambda row: print(json.dumps(row), flush=True)
    result = {"moe_checks": moe_checks(report), "moe_timing": moe_timing(report)}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=1) + "\n")
    return result


def run(output=None) -> dict:
    report = lambda row: print(json.dumps(row), flush=True)
    result = {"checks": checks(report), "timing": timing(report)}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
