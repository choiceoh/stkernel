"""Output-copy and private-page-table graph contracts; no MLA arithmetic or timing."""
from types import SimpleNamespace as NS

import torch

from engine.profiles.glm53.decode_graphs import GraphCaches
from engine.profiles.glm53.lanes import _mla_output
from probes.engine_decode_fusions import _capture


def check(report):
    # clone stands in for the unchanged MLA kernel's fresh, contiguous result.
    # This qualifies only its output ownership and the removed cat, without
    # another native build, weight load, or numerical attention experiment.
    for m in (1, 4, 8, 16, 24, 32):
        source = torch.zeros(m, 16, 512, dtype=torch.bfloat16, device='cuda')
        arms = (lambda: torch.cat([source.clone()], dim=1),
                lambda: _mla_output([source.clone()]))
        graphs, outputs = [], []
        try:
            for fn in arms:
                graph, out = _capture(fn)
                graphs.append(graph); outputs.append(out)
            for phase in range(3):
                bits = (torch.arange(source.numel(), dtype=torch.int32, device='cuda')
                        + phase * 8191).to(torch.int16).reshape_as(source)
                source.view(torch.int16).copy_(bits)
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        outputs[arm].view(torch.int16).fill_(23)
                        graphs[arm].replay()
                    for out in outputs:
                        assert out.is_contiguous() and out.data_ptr() != source.data_ptr()
                        assert torch.equal(out.view(torch.int16), bits)
            report('mla_output_copy', rows=m, exact_bytes=True, fresh_storage=True,
                   removed_copy_bytes=source.numel()*source.element_size(),
                   scope='fresh-result delivery only; unchanged MLA kernel is not executed')
        finally:
            for graph in graphs:
                graph.reset()

    for n in (1, 4):
        table = torch.zeros(6, 352, dtype=torch.int32, device='cuda')[:, ::2]
        ids = torch.arange(n, device='cuda', dtype=torch.int64)
        real = NS(F=NS(kpool=4), layout=None, block_table=table)
        caches = GraphCaches(real, ids, ids, 131072)
        def candidate():
            caches.gather()
            return caches.block_table
        graphs, outputs = [], []
        try:
            for fn in (lambda: table.index_select(0, ids).clamp_min(0), candidate):
                graph, out = _capture(fn)
                graphs.append(graph); outputs.append(out)
            for phase in range(4):
                before = (torch.arange(table.numel(), device='cuda', dtype=torch.int32)
                          .reshape_as(table) + phase) % 19 - 1
                if phase == 3:
                    before.fill_(-1)
                table.copy_(before)
                ids.copy_((torch.arange(n, device='cuda') + phase).flip(0) % 6)
                expected = before.index_select(0, ids).clamp_min(0)
                for order in ((0, 1), (1, 0)):
                    for arm in order:
                        outputs[arm].fill_(-77)
                        graphs[arm].replay()
                    assert all(torch.equal(out, expected) for out in outputs)
                    assert torch.equal(table, before), 'private clamp changed the arena table'
            report('private_page_table', rows=n, exact=True, arena_unchanged=True,
                   removed_allocation_bytes=outputs[1].numel()*outputs[1].element_size())
        finally:
            for graph in graphs:
                graph.reset()
