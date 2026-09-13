"""CPU adapter checks: expanded ABI ownership and prewarm lifetime, no device."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from probes.engine_moe_scatter import RouteScatter, route_scatter_owner


class ScatterOwnerTests(unittest.TestCase):
    def test_owned_buffer_replaces_only_scatter_argument_and_reduces_each_call(self):
        empty = torch.empty
        calls = []
        def compiled(*args):
            calls.append(args)
            args[21].fill_(len(calls))
        class Reduce:
            def __getitem__(self, grid):
                def run(partial, output, width, parts, block, **kwargs):
                    torch.sum(partial.view(grid[0], parts, width), dim=1, out=output)
                return run
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                patch.object(torch, 'empty', side_effect=lambda shape, **kw: empty(shape, dtype=kw['dtype'])), \
                patch('probes.engine_moe_scatter._reduce_routes', Reduce()):
            owner = RouteScatter(compiled, 7)
            output = torch.full((7, 4096), float('nan'))
            args = [object() for _ in range(27)]
            args[21] = output
            for repeat in (1, 2):
                owner(*args)
                self.assertTrue(output.eq(32 * repeat).all())
                self.assertIs(calls[-1][21], owner.scratch)
                for i, arg in enumerate(args):
                    if i != 21:
                        self.assertIs(calls[-1][i], arg)
            args[21] = output.bfloat16()
            with self.assertRaises(ValueError):
                owner(*args)
            self.assertEqual(len(calls), 2)

    def test_capture_misses_fail_before_allocating_and_existing_owner_is_reused(self):
        compiled = object()
        md = SimpleNamespace(_get_static_kernel_v2=lambda *a, **kw: (compiled, 48))
        owners = []
        original = md._get_static_kernel_v2
        with patch('probes.engine_moe_scatter.RouteScatter', side_effect=lambda fn, rows: SimpleNamespace(rows=rows)), \
                route_scatter_owner(md, owners):
            base = md._get_static_kernel_v2(288, 288, 7, config={})
            first = md._get_static_kernel_v2(288, 288, 7, config={'probe_route_scatter': True})
            second = md._get_static_kernel_v2(288, 288, 7, config={'probe_route_scatter': True})
            self.assertIs(base[0], compiled)
            self.assertIs(first, second)
            self.assertEqual(len(owners), 1)
        self.assertIs(md._get_static_kernel_v2, original)
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True), \
                patch.object(torch, 'empty') as empty:
            with self.assertRaisesRegex(RuntimeError, 'prewarm'):
                RouteScatter(compiled, 7)
            empty.assert_not_called()


if __name__ == '__main__':
    unittest.main()
