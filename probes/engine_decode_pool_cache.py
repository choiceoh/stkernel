"""Mapped tail reads and fused cache writes: bytes, changed replay and B/A/A/B."""
import torch


def check(report, *, timing=True):
    from engine.kernels import indexer as native
    from probes.engine_decode_dsa_inputs import ROWS, timings
    from probes.engine_decode_fusions import _capture

    per, stride, capacity = 192, 11 * 192, 32768
    for rows in ROWS:
        n = rows // 8
        contexts = torch.full((n,), 31997, device='cuda', dtype=torch.int64)
        slots = torch.tensor([8, 2, 5, 0], device='cuda')[:n].clone()
        table = torch.empty((n, 352), device='cuda', dtype=torch.int32)[:, ::2]
        page_ids = torch.arange(176, device='cuda')[None, :]
        sequence_pages = 4 * torch.arange(n, device='cuda')[:, None]
        table.copy_(sequence_pages + page_ids % 4)
        source = torch.randn(n, 8, 264, device='cuda', dtype=torch.bfloat16)
        k, gate = source[:, :, :128], source[:, :, 128:256]
        pk = torch.randint(0, 256, (n * 2, 128), device='cuda', dtype=torch.uint8)
        ps = torch.randn(n * 2, device='cuda')
        raw = [torch.empty(4 * n * stride, 136, device='cuda', dtype=torch.uint8) for _ in range(2)]
        pool_keys = [r[:, :128] for r in raw]
        scales = [r[:, 128:132].view(torch.float32).flatten() for r in raw]
        # flatten() of this one-column record view must retain its record stride.
        assert all(s.data_ptr() == r.data_ptr() + 128 and s.stride(0) == 34 for s, r in zip(scales, raw))
        tail_raw = [[torch.empty(9, 10, 2, 136, device='cuda', dtype=torch.bfloat16)
                     for _ in range(11)] for _ in range(2)]
        fields = [[t[:, :, :, :128] for t in arm] for arm in tail_raw]
        tail_seed = torch.randn_like(tail_raw[0][0])
        graphs, outputs = [], []
        try:
            for arm in range(2):
                def run():
                    windows = []
                    for layer in range(11):
                        field = fields[arm][layer]
                        if arm == 0:
                            windows.append(native.pool_window(field.index_select(0, slots), k, gate, contexts, 4, 2))
                            counts, addresses = native.pool_addresses(contexts, table, per, stride, layer * per, 4, 8, 2, capacity)
                            native.scatter_pools(pk, ps, pool_keys[arm], scales[arm], addresses, counts)
                            native.write_tails(field, slots, contexts, k, gate)
                        else:
                            windows.append(native.pool_window(field, k, gate, contexts, 4, 2, slots=slots))
                            native.update_pool_cache(pk, ps, pool_keys[arm], scales[arm], field, slots, contexts, k, gate,
                                                     table, per, stride, layer * per, 4, capacity)
                    return windows
                raw[arm].fill_(0x55)
                for tail in tail_raw[arm]:
                    tail.zero_()
                graph, out = _capture(run)
                graphs.append(graph)
                outputs.append(out)
            for phase, ctx in enumerate((31997, 131061, 32252, 31990, 0, 131064)):
                contexts.copy_(torch.tensor([max(ctx - i, 0) for i in range(n)], device='cuda'))
                slots.copy_((torch.tensor([8, 2, 5, 0], device='cuda')[:n] + phase) % 9)
                table.copy_(sequence_pages + (page_ids + phase) % 4)
                source.normal_()
                pk.random_(0, 256)
                ps.normal_()
                tail_seed.normal_()
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        raw[arm].fill_(0x55)
                        for layer, tail in enumerate(tail_raw[arm]):
                            tail.copy_(tail_seed + layer)
                        for window in outputs[arm]:
                            for value in window:
                                value.fill_(float('nan'))
                        graphs[arm].replay()
                    torch.testing.assert_close(raw[1], raw[0], rtol=0, atol=0)
                    for a, b in zip(tail_raw[1], tail_raw[0]):
                        torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)
                    for a, b in zip(outputs[1], outputs[0]):
                        for got, want in zip(a, b):
                            torch.testing.assert_close(got.view(torch.uint8), want.view(torch.uint8), rtol=0, atol=0)
            report('pool_cache_numerics', rows=rows, layers=11, exact=True, rebound_slots=True,
                   rebound_tables=True, rollback=True, poison_padding=True, replay_orders='BA/AB',
                   contexts=[31997, 131061, 32252, 31990, 0, 131064])
            if timing:
                timings(report, 'pool_cache_glue', rows, graphs, layers=11,
                        baseline_launches=55, candidate_launches=22, compression_included=False)
        finally:
            for graph in graphs:
                graph.reset()
