"""Lossless scale reconstruction; CPU byte maps do not replace CUDA proof."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Import the actual pure-Python files without b12x.__init__'s CUDA wrapper.
# GPU cases below separately import the ordinary production package.
_package = '_sf6_prefill_scale_cpu_test'
_namespace = ModuleType(_package)
_namespace.__path__ = [str(ROOT / 'engine/kernels/b12x')]
sys.modules[_package] = _namespace


def pure_module(name):
    spec = importlib.util.spec_from_file_location(_package + '.' + name,
                                                  ROOT / 'engine/kernels/b12x' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_layout = pure_module('moe_reform_sf_pack')
pack_stage_bytes, stage_shape, stage_source_offset = (
    _layout.pack_stage_bytes, _layout.stage_shape, _layout.stage_source_offset)
plane_geometry = pure_module('moe_sf6_prefill_scales').plane_geometry


def fixture(experts, rows, k, kind, seed=0):
    nr, nk = stage_shape(rows, k, kind)
    packed = bytearray()
    expected = bytearray(experts * rows * k // 16)
    for e in range(experts):
        for rt in range(nr):
            for kt in range(nk):
                stage = (e * nr + rt) * nk + kt
                base = (stage * 71 + seed * 31) % 193
                raw = bytes(base + (i * 13 + stage * 7 + seed * 11) % 64 for i in range(2048))
                packed.extend(pack_stage_bytes(raw))
                for i, code in enumerate(raw):
                    expected[stage_source_offset(rows, k, kind, e, rt, kt, i)] = code
    return packed, expected


class Address:
    def __init__(self, storage, offset=0):
        self.storage, self.offset = storage, offset

    def __add__(self, value):
        return Address(self.storage, self.offset + value)


def scalar_kernel():
    """Execute the actual integer byte-map body one lane at a time.

    Remove DSL decorators, annotations and integer casts only. All tested
    addresses fit signed int64 and all byte values are already 0..255.
    This checks arithmetic and indexing, not Triton lowering or GPU ordering.
    """
    tree = ast.parse((ROOT / 'engine/kernels/b12x/moe_sf6_prefill_scales_kernel.py').read_text())
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)))
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    class Casts(ast.NodeTransformer):
        def visit_Call(self, n):
            n = self.generic_visit(n)
            return n.func.value if isinstance(n.func, ast.Attribute) and n.func.attr == 'to' else n
    node = Casts().visit(node)
    lane = SimpleNamespace(stage=0, byte=0)
    tl = SimpleNamespace(program_id=lambda _: lane.stage, arange=lambda *_: lane.byte,
                         load=lambda p: p.storage[p.offset],
                         store=lambda p, v: p.storage.__setitem__(p.offset, v))
    ns = {'tl': tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<actual-scale-kernel>', 'exec'), ns)
    return ns['expand'], lane


class ScaleExpansionCpuTests(unittest.TestCase):
    def test_actual_kernel_indexing_against_independent_stage_map(self):
        kernel, lane = scalar_kernel()
        for kind, rows, k in (('fc1', 256, 512), ('fc2', 512, 512)):
            packed, expected = fixture(3, rows, k, kind)
            actual = bytearray(len(expected))
            for stage in range(len(packed) // 1552):
                lane.stage = stage
                for byte in range(2048):
                    lane.byte = byte
                    kernel(Address(packed), Address(actual), k // (256 if kind == 'fc1' else 128), kind == 'fc2')
            self.assertEqual(actual, expected, kind)

    def test_one_layer_storage_bound(self):
        a, first = plane_geometry(288, 1024, 4096, 'fc1')
        b, second = plane_geometry(288, 4096, 512, 'fc2')
        self.assertEqual(a, (288, 128, 1552))
        self.assertEqual(b, (288, 64, 1552))
        self.assertEqual(first + second, 113246208)  # 108 MiB, not 42 resident layers.

    def test_geometry_rejects_invalid_planes(self):
        for args in ((True, 1024, 4096, 'fc1'), (0, 1024, 4096, 'fc1'),
                     (288, 1023, 4096, 'fc1'), (288, 4096, 511, 'fc2'),
                     (288, 1024, 4096, 'other')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                plane_geometry(*args)

    def test_selector_excludes_decode_and_other_geometries(self):
        tree = ast.parse((ROOT / 'engine/kernels/b12x/moe_dispatch.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_prefill_scale_expansion_eligible')
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<production-selector>', 'exec'), ns)
        select = ns[fn.name]
        args = dict(m=2672, E=288, k=4096, n=512, num_topk=8, tile_m=128,
                    quant_mode='nvfp4', tiled=True, activation='swigluoai_uninterleave',
                    swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10., share_input_across_experts=False)
        for rows in (65, 2672, 8192, 8193, 32256, 32768):
            self.assertTrue(select(**dict(args, m=rows)))
        for changed in ({'m': 7}, {'m': 64}, {'m': 32769}, {'m': True}, {'E': 72},
                        {'k': 5120}, {'tile_m': 64}, {'tiled': False},
                        {'share_input_across_experts': True}, {'swiglu_limit': 0.}):
            self.assertFalse(select(**dict(args, **changed)), changed)


torch = None
if importlib.util.find_spec('torch'):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires admitted CUDA device')
class ScaleExpansionCudaTests(unittest.TestCase):
    def test_original_byte_layout_and_changed_inputs(self):
        from engine.kernels.b12x.moe_sf6_prefill_scales import expand_plane
        for kind, rows, k in (('fc1', 256, 512), ('fc2', 512, 512)):
            shape, _ = plane_geometry(3, rows, k, kind)
            source = torch.empty(shape, dtype=torch.uint8, device='cuda')
            prior_output = None
            for seed in (0, 1):
                packed, expected = fixture(3, rows, k, kind, seed)
                source.copy_(torch.tensor(list(packed), dtype=torch.uint8, device='cuda').reshape(shape))
                before = source.clone()
                actual = expand_plane(source, experts=3, rows=rows, k=k, kind=kind)
                self.assertEqual(bytes(actual.cpu().tolist()), expected)
                self.assertTrue(torch.equal(source, before))
                if prior_output is not None:
                    self.assertFalse(torch.equal(actual, prior_output))
                prior_output = actual

    def test_capture_refused(self):
        from engine.kernels.b12x.moe_sf6_prefill_scales import expand_plane
        packed, _ = fixture(1, 128, 256, 'fc1')
        source = torch.tensor(list(packed), dtype=torch.uint8, device='cuda').reshape(1, 1, 1552)
        from unittest.mock import patch
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'eager prefill'):
                expand_plane(source, experts=1, rows=128, k=256, kind='fc1')


if __name__ == '__main__':
    unittest.main()
