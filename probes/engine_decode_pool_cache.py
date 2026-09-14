"""Pool completion: original, mapped-window and direct-reader captured comparisons."""
import hashlib

import torch


def check(report, ranks=None, *, timing=True):
    from engine.kernels import indexer as native
    from engine.kernels.kpool import compress_pool_keys, compress_decode_pools
    from probes.engine_decode_dsa_inputs import ROWS, timings
    from probes.engine_decode_fusions import _capture

    if ranks:
        from probes.engine_decode_scatter_check import rank_path
        from engine.profiles.glm53.weights import rank_loader
        path = rank_path(ranks)
        loader = rank_loader(path)
        names = sorted(k for k in loader.keys() if k.endswith('.idx.ape'))
        if len(names) != 11:
            raise RuntimeError('pool gate requires all 11 real DSA biases')
        loaded = loader.load(names, device='cuda')
        apes = [loaded[name] for name in names]
        origin = str(path)
    else:
        apes = [torch.randn(4, 128, device='cuda') for _ in range(11)]
        origin = 'synthetic FP32 pool biases'
    report('pool_weights', source=origin, layers=11,
           weight_sha256=[hashlib.sha256(w.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for w in apes])
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
        raw = [torch.empty(4 * n * stride, 136, device='cuda', dtype=torch.uint8) for _ in range(3)]
        pool_keys = [r[:, :128] for r in raw]
        scales = [r[:, 128:132].view(torch.float32).flatten() for r in raw]
        # flatten() of this one-column record view must retain its record stride.
        assert all(s.data_ptr() == r.data_ptr() + 128 and s.stride(0) == 34 for s, r in zip(scales, raw))
        tail_raw = [[torch.empty(9, 10, 2, 136, device='cuda', dtype=torch.bfloat16)
                     for _ in range(11)] for _ in range(3)]
        fields = [[t[:, :, :, :128] for t in arm] for arm in tail_raw]
        tail_seed = torch.randn_like(tail_raw[0][0])
        graphs, outputs = [], []
        try:
            for arm in range(3):
                def run():
                    compressed = []
                    for layer in range(11):
                        field = fields[arm][layer]
                        if arm == 2:
                            pk, ps = compress_decode_pools(field, k, gate, apes[layer], contexts, slots)
                        else:
                            kw, gw = (native.pool_window(field.index_select(0, slots), k, gate, contexts, 4, 2) if arm == 0 else
                                      native.pool_window(field, k, gate, contexts, 4, 2, slots=slots))
                            pk, ps = compress_pool_keys(kw, gw, apes[layer])
                        compressed.append((pk, ps))
                        if arm == 0:
                            counts, addresses = native.pool_addresses(contexts, table, per, stride, layer * per, 4, 8, 2, capacity)
                            native.scatter_pools(pk, ps.view(-1), pool_keys[arm], scales[arm], addresses, counts)
                            native.write_tails(field, slots, contexts, k, gate)
                        else:
                            native.update_pool_cache(pk, ps.view(-1), pool_keys[arm], scales[arm], field, slots, contexts, k, gate,
                                                     table, per, stride, layer * per, 4, capacity)
                    return compressed
                raw[arm].fill_(0x55)
                for tail in tail_raw[arm]:
                    tail.zero_()
                graph, out = _capture(run)
                graphs.append(graph)
                outputs.append(out)
            for phase, ctx in enumerate((31997, 131059, 32252, 31990, 0, 131064)):
                contexts.copy_(torch.tensor([max(ctx - i, 0) for i in range(n)], device='cuda'))
                slots.copy_((torch.tensor([8, 2, 5, 0], device='cuda')[:n] + phase) % 9)
                table.copy_(sequence_pages + (page_ids + phase) % 4)
                source.normal_()
                tail_seed.normal_()
                magnitude = (1e-6, 100., 1., .1, 0., 10.)[phase]
                k.mul_(magnitude)
                tail_seed[:, :, 0].mul_(magnitude)
                for order in ((0, 1, 2), (2, 1, 0)):
                    for arm in order:
                        raw[arm].fill_(0x55)
                        for layer, tail in enumerate(tail_raw[arm]):
                            tail.copy_(tail_seed + layer)
                        for value, scale in outputs[arm]:
                            value.view(torch.uint8).fill_(0x7f)
                            scale.fill_(float('nan'))
                        graphs[arm].replay()
                    for arm in (1, 2):
                        torch.testing.assert_close(raw[arm], raw[0], rtol=0, atol=0)
                        for a, b in zip(tail_raw[arm], tail_raw[0]):
                            torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)
                        for a, b in zip(outputs[arm], outputs[0]):
                            for got, want in zip(a, b):
                                if not got.float().isfinite().all().item():
                                    raise RuntimeError('pooling left non-finite output')
                                torch.testing.assert_close(got.view(torch.uint8), want.view(torch.uint8), rtol=0, atol=0)
            report('pool_cache_numerics', rows=rows, layers=11, exact=True, rebound_slots=True,
                   rebound_tables=True, rollback=True, poison_padding=True, replay_orders='012/210',
                   contexts=[31997, 131059, 32252, 31990, 0, 131064])
            if timing:
                timings(report, 'pool_direct_reader', rows, graphs[1:], layers=11,
                        baseline_launches=33, candidate_launches=22, compression_included=True)
                timings(report, 'pool_completion_combined', rows, [graphs[0], graphs[2]], layers=11,
                        baseline_launches=66, candidate_launches=22, compression_included=True)
        finally:
            for graph in graphs:
                graph.reset()


def ids_check(report, *, timing=True):
    """Same selected ids: cast+finalize versus direct int64 input, all layer offsets."""
    from engine.kernels.indexer import pool_slots
    from probes.engine_decode_dsa_inputs import ROWS, timings
    from probes.engine_decode_fusions import _capture
    for rows in ROWS:
        n = rows // 8
        ids = torch.randint(0, 32768, (rows, 512), device='cuda', dtype=torch.int64)
        lengths = torch.full((rows,), 32000, device='cuda', dtype=torch.int32)
        table = torch.empty(n, 352, device='cuda', dtype=torch.int32)[:, ::2]
        pages = torch.arange(176, device='cuda')[None, :]
        table.copy_((pages + 3 * torch.arange(n, device='cuda')[:, None]) % 16)
        graphs, outputs = [], []
        try:
            for arm in range(2):
                def run():
                    out = []
                    for layer in range(11):
                        dst = torch.empty(rows, 2051, device='cuda', dtype=torch.int32)
                        count = torch.empty(rows, device='cuda', dtype=torch.int32)
                        pool_slots(ids.to(torch.int32) if arm == 0 else ids, lengths, 4, table,
                                   768, 8448, layer * 768, dst, count, tokens=8)
                        out.append((dst, count))
                    return out
                graph, out = _capture(run)
                graphs.append(graph)
                outputs.append(out)
            for phase, ctx in enumerate((0, 31997, 31990, 131059)):
                lengths.copy_((ctx + torch.arange(rows, device='cuda') % 8 + 1).int())
                ids.random_(-3, 32768)
                ids[:, ::7] = 0
                ids[:, 1::7] = lengths[:, None] // 4
                table.copy_((pages + phase + 3 * torch.arange(n, device='cuda')[:, None]) % 16)
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        for values in outputs[arm]:
                            for value in values:
                                value.fill_(-777)
                        graphs[arm].replay()
                    for a, b in zip(outputs[1], outputs[0]):
                        for got, want in zip(a, b):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
            report('topk_ids_numerics', rows=rows, layers=11, exact=True, rebound_tables=True,
                   rollback=True, invalid_and_duplicate_ids=True, sort_dtype='int32', replay_orders='BA/AB')
            if timing:
                timings(report, 'topk_ids_read', rows, graphs, layers=11, baseline_launches=22, candidate_launches=11)
        finally:
            for graph in graphs:
                graph.reset()
