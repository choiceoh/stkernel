"""Row ceilings of the shared FP32 scatter workspace: legacy tp vs long-prefill."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'engine/kernels/b12x/moe_dispatch.py'


def extract(path, name, ns):
    fn = copy.deepcopy(next(n for n in ast.walk(ast.parse(path.read_text()))
                            if isinstance(n, ast.FunctionDef) and n.name == name))
    fn.decorator_list = []
    exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0], fn],
                            type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


class Tensor:
    def __init__(self, shape=(4, 4096), dtype='bf16', device='cuda:0', events=None, parent=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.events = events if events is not None else []
        self.parent = parent or self
        self.contiguous = True
        self.ndim = len(self.shape)
    def data_ptr(self): return id(self.parent)
    def is_contiguous(self): return self.contiguous
    def numel(self): return math.prod(self.shape)
    def record_stream(self, stream): self.events.append(('record', stream, self.data_ptr()))
    def __getitem__(self, s): return Tensor((s.stop, self.shape[1]), self.dtype, self.device, events=self.events, parent=self.parent)


class ScatterCeilingTests(unittest.TestCase):
    def test_compact_ep_rows_use_bound_width_and_owned_grow_only_storage(self):
        ns, events, allocations = self.namespace()
        ns['_admitted_moe'] = lambda: SimpleNamespace(hidden=2560)
        fn = ns['_ep_local_scatter_buffer']
        ws = SimpleNamespace(device='cuda:0', ep_scatter_fp32=None)
        first = fn(ws, Tensor((40960, 2560)), 40960, 2560, bound_ep=True)
        again = fn(ws, Tensor((17, 2560)), 17, 2560, bound_ep=True)
        self.assertEqual(first.data_ptr(), again.data_ptr())
        self.assertEqual(len(allocations), 1)
        self.assertEqual(len(events), 2)
        with self.assertRaises(ValueError):
            fn(ws, Tensor((17, 4096)), 17, 4096, bound_ep=True)
        with self.assertRaises(ValueError):
            fn(ws, Tensor((2**31 // (2560*4) + 1, 2560)), 2**31 // (2560*4) + 1, 2560, bound_ep=True)

    def namespace(self):
        events, allocations = [], []
        def empty(shape, **kw):
            result = Tensor(shape, events=events, **kw); allocations.append(result); return result
        torch = SimpleNamespace(bfloat16='bf16', float32='f32', empty=empty,
                                cuda=SimpleNamespace(current_stream=lambda dev: 'side-stream'))
        ns = dict(torch=torch)
        extract(MD, '_ep_local_scatter_buffer', ns)
        return ns, events, allocations

    def test_legacy_tp_path_keeps_the_16384_ceiling(self):
        ns, _, allocations = self.namespace(); fn = ns['_ep_local_scatter_buffer']
        ws = SimpleNamespace(device='cuda:0', ep_scatter_fp32=None)
        fn(ws, Tensor((16384, 4096)), 16384, 4096, tp=True)
        self.assertEqual(allocations[0].shape, (16384, 4096))
        with self.assertRaises(ValueError):
            fn(SimpleNamespace(device='cuda:0', ep_scatter_fp32=None), Tensor((16385, 4096)), 16385, 4096, tp=True)

    def test_long_prefill_path_grows_to_its_32768_predicate(self):
        ns, _, allocations = self.namespace(); fn = ns['_ep_local_scatter_buffer']
        ws = SimpleNamespace(device='cuda:0', ep_scatter_fp32=None)
        plane = fn(ws, Tensor((32256, 4096)), 32256, 4096, tp=True, long_prefill=True)
        self.assertEqual(allocations[0].shape, (32256, 4096))
        self.assertEqual(32256 * 4096 * 4, 504 * 1024 * 1024)
        again = fn(ws, Tensor((32768, 4096)), 32768, 4096, tp=True, long_prefill=True)
        self.assertNotEqual(plane.data_ptr(), again.data_ptr())
        self.assertEqual(allocations[-1].shape, (32768, 4096))
        with self.assertRaises(ValueError):
            fn(ws, Tensor((32769, 4096)), 32769, 4096, tp=True, long_prefill=True)

    def test_default_call_is_unchanged_ep_local_semantics(self):
        ns, _, _ = self.namespace(); fn = ns['_ep_local_scatter_buffer']
        ws = SimpleNamespace(device='cuda:0', ep_scatter_fp32=None)
        with self.assertRaises(ValueError):
            fn(ws, Tensor((4095, 4096)), 4095, 4096)
        fn(ws, Tensor((16384, 4096)), 16384, 4096)
        with self.assertRaises(ValueError):
            fn(ws, Tensor((16385, 4096)), 16385, 4096)

    def test_tiled_ep_still_refuses_lazy_allocation(self):
        ns, _, _ = self.namespace(); fn = ns['_ep_local_scatter_buffer']
        ws = SimpleNamespace(device='cuda:0', ep_scatter_fp32=None, ep_tiled=True)
        with self.assertRaises(RuntimeError):
            fn(ws, Tensor((8192, 4096)), 8192, 4096)


if __name__ == '__main__':
    unittest.main()
