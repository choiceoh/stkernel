"""Bounded single-GPU A/B for drafter weight ownership and C=1/C=4 blocks.

Synthetic weights exercise real native packing, attention, block execution and
graph replay. TP rank arithmetic is isolated: this is not full-model acceptance.
"""
import gc
import hashlib
import json
import weakref
from types import SimpleNamespace

import torch

from engine.base.arena import Arena
from engine.profiles.glm53.drafter import Drafter, DrafterFacts, specs
from engine.profiles.glm53.drafter_storage import nbytes


def check():
    assert torch.cuda.get_device_capability() == (12, 1), 'requires GB10'
    F = DrafterFacts(layers=1, hidden=512, heads=4, kv_heads=4, head_dim=128,
                    inter=512, rms_eps=1e-6, rope_theta=10000., window=16, block=8,
                    mask_id=1, conv_taps=2, conv_group=16, sel_rank=16,
                    sel_top_k=4, target_layers=(1,), k=6)
    declared = specs(F)
    source_bytes = sum((s.nbytes() + 255) // 256 * 256 for s in declared)
    expected, hashes, results = {}, {}, {}

    def run(compact):
        torch.manual_seed(1183)
        before = torch.cuda.memory_allocated()
        source = Arena(source_bytes)
        source_ref = weakref.ref(source.buf)
        views = {}
        for spec in declared:
            value = source.carve(spec.nbytes(), spec.name).view(spec.dtype).view(spec.shape)
            if 'norm.weight' in spec.name:
                value.fill_(1)
            else:
                value.normal_(std=.02)
            views[spec.name] = value
        del value
        target = SimpleNamespace(comm=SimpleNamespace(world_size=4, rank=0, all_reduce=lambda x: x))
        d = Drafter(F, target, 512)
        d.bind(views)
        views.clear()
        resident = Arena(nbytes(F, 4, 4)) if compact else source
        d.prepare_fast(max_seqs=4 if compact else None,
                       compact_into=resident if compact else None, consume_weights=not compact)
        if compact:
            del source  # boot drops normal loader owners; do not force storage invalidation
        gc.collect()
        if compact:
            assert source_ref() is None, 'a retired raw weight still owns the source allocation'
        torch.cuda.synchronize()
        live_allocated = torch.cuda.memory_allocated() - before
        assert all((layer.fp8 is not None) == (not compact or name == 'fc.weight')
                   for name, layer in d.dense.items())
        for name, layer in d.dense.items():
            raw = b''.join(t.view(torch.uint8).cpu().numpy().tobytes()
                           for p in layer.packs for t in (p.data, p.scale, p.rowscale))
            digest = hashlib.sha256(raw).hexdigest()
            if compact:
                assert digest == hashes[name], ('pack bytes', name)
            else:
                hashes[name] = digest
            for rows in ((7, 28, 128) if name == 'fc.weight' else (7, 28)):
                torch.manual_seed(730 + rows)
                x = torch.randn(rows, layer.cols, device='cuda', dtype=torch.bfloat16)
                out = layer(x).cpu()
                key = (name, rows)
                if compact:
                    assert torch.equal(out, expected[key]), key
                else:
                    expected[key] = out
        torch.manual_seed(138)
        embed = torch.randn(512, F.hidden, device='cuda', dtype=torch.bfloat16) * .02
        target.embed = lambda ids: torch.nn.functional.embedding(ids, embed)
        for concurrency in (1, 4):
            n, t = concurrency, F.k + 1
            torch.manual_seed(815 + n)
            field = torch.randn(5, F.layers, 2, F.window, 1, F.head_dim,
                                device='cuda', dtype=torch.bfloat16) * .02
            original = field.clone()
            slots = torch.arange(1, n + 1, device='cuda')
            ctx = torch.arange(n, device='cuda') * 9 + 5
            ids = torch.randint(512, (n * t,), device='cuda')
            pos = (ctx[:, None] + torch.arange(t, device='cuda')).flatten()
            call = lambda: d.block_rows(ids, pos, slots, ctx, field, n, t)
            out = call().clone()
            key = ('block', n)
            if compact:
                assert torch.equal(out.cpu(), expected[key]), key
            else:
                expected[key] = out.cpu()
            # Capture the compact addresses; changed inputs and ring contents
            # must remain identical to eager execution after source retirement.
            graph = torch.cuda.CUDAGraph()
            call()
            with torch.cuda.graph(graph):
                captured = call()
            try:
                for _ in range(3):
                    ids.random_(512)
                    field.copy_(original)
                    eager = call().clone()
                    field.copy_(original)
                    graph.replay()
                    assert torch.equal(captured, eager), ('replay', compact, n)
            finally:
                graph.reset()
        results['compact' if compact else 'baseline'] = dict(
            arena_bytes=resident.nbytes, allocated_bytes=live_allocated,
            block_fp8_packs=sum(v.fp8 is not None for k, v in d.dense.items() if k != 'fc.weight'))
        resident.release()

    run(False)
    gc.collect()
    torch.cuda.empty_cache()
    run(True)
    assert results['compact']['allocated_bytes'] < results['baseline']['allocated_bytes'], results
    results.update(passed=True, dense_output_cases=len(expected) - 2, block_concurrency=[1, 4],
                   replay_cases=12, equal_w4_packs=len(hashes),
                   scope='synthetic native weights, isolated TP-rank arithmetic')
    return results


if __name__ == '__main__':
    print(json.dumps(check()), flush=True)
