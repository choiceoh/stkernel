"""GPU-side cost of one one-shot sum on one GB10: the PDL consumer against the ordinary kernel, same oracle build.

The CPU lands every peer packet, flag and acknowledgement before a chain publishes, so neither RDMA transfer nor peer
skew is in these numbers; they are not transport or engine speed. What is left is what the dispatch choice changes on
the GPU: the launch behind its producer (a PDL producer that releases the sum early, or an ordinary torch copy), the
ring guard, copy, fences and tickets on the owning CTAs (the consumer's 16 or 32 at 8 or 16 rows, the ordinary
kernel's 48), publication, the landed-flag check and the reduce. Each arm is a captured chain of CHAIN producer->sum
pairs with events inside the graph; B/A/A/B (B = ordinary control), warm and after a 128 MiB eviction.
"""
from statistics import mean, median

import torch

HIDDEN = 4096
ROWS = (8, 16)
CHAIN, REPLAYS = 8, 32
PRODUCER_US = 40          # about one 16-row mk_gemm2 span, as SM cycles at 1592 MHz below
SM_MHZ = 1592


def check(report):
    from engine.kernels.mapped_staging import allocate
    from tests.test_engine_direct_producer_cuda import build_oracle
    ext = build_oracle()
    host, device = allocate(ext.bytes())
    rank = 1
    ext.prepare(host, device, rank, 0, 0)
    cold = torch.empty(128 << 20, dtype=torch.uint8, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(967)
    cycles = PRODUCER_US * SM_MHZ
    rows_out = []
    for rows in ROWS:
        values = (torch.randn(4, rows * HIDDEN, device='cuda', generator=generator)).bfloat16()
        staged = values[rank].clone()
        peers = values[[r for r in range(4) if r != rank]].contiguous().view(torch.uint8).cpu()
        x = torch.empty_like(staged)
        expected = values[0].float()
        for r in range(1, 4):   # the transport's order: global ranks 0, 1, 2, 3 in FP32, one BF16 rounding
            expected = expected + values[r].float()
        expected = expected.bfloat16()
        for producer in ('pdl', 'ordinary'):
            for cache in ('warm', 'evicted'):
                graphs = {}
                try:
                    for arm, kernel in (('ordinary', ext.oneshot_ar), ('consumer', ext.oneshot_ar_consumer)):
                        start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))

                        def chain(kernel=kernel, start=start, end=end):
                            if cache == 'evicted':
                                cold.fill_(19)
                            start.record()
                            for _ in range(CHAIN):
                                if producer == 'pdl':
                                    ext.staged_copy(staged, x, cycles)
                                else:
                                    x.copy_(staged)
                                out = kernel(x)
                            end.record()
                            return out

                        torch.cuda.synchronize()
                        ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            out = chain()
                        torch.cuda.synchronize()
                        graphs[arm] = graph, start, end, out
                    samples = []
                    for arm in ('ordinary', 'consumer', 'consumer', 'ordinary'):
                        graph, start, end, out = graphs[arm]
                        us = []
                        for _ in range(REPLAYS):
                            torch.cuda.synchronize()
                            ext.land_ahead(host, ext.published(host) + 1, CHAIN, peers)
                            graph.replay()
                            end.synchronize()
                            us.append(start.elapsed_time(end) * 1000. / CHAIN)
                        if not torch.equal(out.view(torch.int16), expected.view(torch.int16)):
                            raise RuntimeError(f'{arm} sum differs from the rank-ordered fold at {rows} rows')
                        samples.append(dict(arm=arm, median_us=round(median(us), 2), min_us=round(min(us), 2),
                                            mean_us=round(mean(us[1:]), 2)))
                    arms = {arm: [s for s in samples if s['arm'] == arm] for arm in ('ordinary', 'consumer')}
                    row = dict(rows=rows, producer=producer, cache=cache, chain=CHAIN, replays=REPLAYS,
                               producer_us=PRODUCER_US if producer == 'pdl' else None, samples=samples,
                               **{f'{arm}_{stat}': round(mean(s[stat] for s in picked), 2)
                                  for arm, picked in arms.items() for stat in ('median_us', 'min_us', 'mean_us')})
                    row['consumer_vs_ordinary_median'] = round(row['consumer_median_us'] / row['ordinary_median_us'] - 1, 4)
                    row['consumer_vs_ordinary_min'] = round(row['consumer_min_us'] / row['ordinary_min_us'] - 1, 4)
                    report('oneshot_consumer_timing', **row)
                    rows_out.append(row)
                finally:
                    for graph, *_ in graphs.values():
                        graph.reset()
    if ext.tickets(host) != ext.published(host) * 48:
        raise RuntimeError('publication tickets drifted during timing')
    return rows_out
