"""Actual-source CPU oracles for rounded BF16 contributions in an FP32 sum.

No accelerator imports. The immutable-image helper is the conversion oracle;
the frozen micro-kernel AST binds the unchanged arithmetic and synchronization.
Device ordering, FTZ behavior, and performance still require the serving gate.
"""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
MICRO = ROOT/'overlay/modules/glm53_moe/moe_micro_kernel.py'
ORACLE = ROOT/'measurements/glm53_ep_local_20260908/micro-stock-oracle'
STOCK_SOURCE_SHA256 = 'a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1'
# MoEMicroKernel.kernel at frozen CPU11/onepass11 c7dec80a0f73; ast.dump
# without source locations, so formatting/comments do not affect this check.
STOCK_KERNEL_AST_SHA256 = '8525a206b8e020137fe5dcb86cb887214b0e0d51fe23468de927b2225a315eab'


def function(name):
    return next(n for n in ast.walk(ast.parse(MICRO.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def extract(fn, namespace):
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom('__future__', [ast.alias('annotations')], 0), fn],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(MICRO), 'exec'), namespace)
    return namespace[fn.name]


def inline_asm(fn):
    return next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'llvm.inline_asm')


class DType:
    def __init__(self, name): self.name = name
    def __call__(self, value): return (self.name, value)


class ScatterTests(unittest.TestCase):
    def test_exact_stock_saturating_conversion_and_lane_order_are_retained(self):
        raw = gzip.decompress((ORACLE/'fp4_common.py.gz').read_bytes())
        identity = json.loads((ORACLE/'identity.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), STOCK_SOURCE_SHA256)
        self.assertEqual(identity['source_sha256'], STOCK_SOURCE_SHA256)
        self.assertEqual(identity['image'],
                         'sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211')
        stock = next(n for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.FunctionDef)
                     and n.name == 'scatter_add_bf16x2')
        old, new = inline_asm(stock), inline_asm(function('scatter_add_bf16x2_to_f32'))
        self.assertEqual(ast.dump(old.args[1]), ast.dump(new.args[1]))
        self.assertEqual(old.args[3].value, new.args[3].value)
        old_text, new_text = old.args[2].value, new.args[2].value
        convert = 'cvt.rn.satfinite.bf16x2.f32 packed, $2, $1;'
        self.assertEqual(old_text.count(convert), 1)
        self.assertEqual(new_text.count(convert), 1)
        self.assertNotIn('red.relaxed.gpu.global.add.noftz.bf16x2', new_text)
        self.assertIn('mov.b32 {h0,h1}, packed;', new_text)
        self.assertIn('cvt.f32.bf16 v0, h0; cvt.f32.bf16 v1, h1;', new_text)
        self.assertIn('red.relaxed.gpu.global.add.v2.f32 [$0], {v0,v1};', new_text)
        self.assertEqual(new_text.count('red.'), 1)
        self.assertLess(new_text.index(convert), new_text.index('cvt.f32.bf16'))
        self.assertLess(new_text.index('cvt.f32.bf16'), new_text.index('red.'))
        self.assertTrue(next(k.value.value for k in new.keywords if k.arg == 'has_side_effects'))

    def test_packed_finite_bf16_words_widen_without_rounding_or_lane_swap(self):
        # Every finite BF16 word, including signed zero and subnormals, has an
        # exact FP32 representation. This checks the conversion boundary, not
        # FP32 RED (whose subnormal FTZ distinction is explicitly documented).
        for low in range(65536):
            if low & 0x7f80 == 0x7f80:
                continue
            high = low ^ 0x8000
            packed = low | high << 16
            recovered = []
            for shift in (0, 16):
                word = packed >> shift & 0xffff
                value = struct.unpack('<f', struct.pack('<I', word << 16))[0]
                f32bits = struct.unpack('<I', struct.pack('<f', value))[0]
                self.assertEqual(f32bits & 0xffff, 0)
                recovered.append(f32bits >> 16)
            self.assertEqual(recovered, [low, high])

    def test_constructor_default_is_stock_and_variants_retain_same_geometry(self):
        dtype = SimpleNamespace(Float32='f32')
        namespace = dict(cutlass=dtype, DenseGemmKernel=object,
                         is_gated_activation=lambda a: a == 'swigluoai_uninterleave',
                         utils=SimpleNamespace(get_smem_capacity_in_bytes=lambda arch: 101376),
                         pipeline=SimpleNamespace(NamedBarrier=lambda **kw: kw))
        init = extract(function('__init__'), namespace)
        base, candidate = SimpleNamespace(), SimpleNamespace()
        common = dict(sf_vec_size=16, mma_tiler_mn=(32,128), output_tile_count_n=16,
                      activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                      swiglu_limit=10., skip_zero_weight_expert_id=72)
        init(base, **common)
        init(candidate, **common, scatter_fp32=True)
        self.assertIs(base.scatter_fp32, False)
        self.assertIs(candidate.scatter_fp32, True)
        self.assertEqual({k:v for k,v in vars(base).items() if k != 'scatter_fp32'},
                         {k:v for k,v in vars(candidate).items() if k != 'scatter_fp32'})

    def test_wrong_output_dtype_fails_before_layout_or_tma_work(self):
        fn = function('__call__')
        end = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == 'self.a_dtype')
        code = compile(ast.Module(body=fn.body[:end], type_ignores=[]), str(MICRO), 'exec')
        bf16, f32 = DType('bf16'), DType('f32')
        cutlass = SimpleNamespace(BFloat16=bf16, Float32=f32, const_expr=lambda x:x)
        for enabled in (False, True):
            for actual in (bf16, f32, object()):
                ns = dict(self=SimpleNamespace(scatter_fp32=enabled), cutlass=cutlass,
                          scatter_output=SimpleNamespace(element_type=actual))
                if actual is (f32 if enabled else bf16):
                    exec(code, ns)
                else:
                    with self.assertRaisesRegex(ValueError, 'scatter output dtype'):
                        exec(code, ns)

    def test_zeroing_covers_exact_element_addresses_for_both_widths(self):
        fn = function('kernel')
        loop = next(n for n in ast.walk(fn) if isinstance(n,ast.While)
                    and ast.unparse(n.test) == 'j < scatter_total')
        code = compile(ast.Module(body=[loop], type_ignores=[]), str(MICRO), 'exec')
        for enabled, width in ((False,2),(True,4)):
            writes = {}
            class Output:
                def __setitem__(self, index, value):
                    row,col=index
                    address=(row*4096+col)*width
                    if address in writes: raise AssertionError('duplicate writer')
                    writes[address]=value
            cutlass = SimpleNamespace(BFloat16=DType('bf16'), Float32=DType('f32'),
                                      const_expr=lambda x:x)
            for tid in range(160*48):
                exec(code, dict(j=tid, scatter_total=8*4096, cols=4096, flat_stride=160*48,
                                scatter_output=Output(), cutlass=cutlass,
                                self=SimpleNamespace(scatter_fp32=enabled)))
            self.assertEqual(set(writes), set(range(0,8*4096*width,width)))
            self.assertEqual(set(writes.values()), {('f32' if enabled else 'bf16',0.0)})

    def test_entire_kernel_math_routing_and_barriers_are_unchanged(self):
        class SelectScatter(ast.NodeTransformer):
            def __init__(self, enabled): self.enabled=enabled
            def visit_If(self, node):
                if ast.unparse(node.test) == 'cutlass.const_expr(self.scatter_fp32)':
                    return [self.visit(n) for n in (node.body if self.enabled else node.orelse)]
                return self.generic_visit(node)
            def visit_Name(self, node):
                if node.id == 'scatter_add_bf16x2_to_f32':
                    node.id='scatter_add_bf16x2'
                return node
            def visit_Assign(self, node):
                if (len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript)
                        and ast.unparse(node.targets[0].value)=='scatter_output'
                        and ast.unparse(node.value)=='cutlass.Float32(0.0)'):
                    node.value.func.attr='BFloat16'
                return self.generic_visit(node)
        for enabled in (False,True):
            normalized=SelectScatter(enabled).visit(copy.deepcopy(function('kernel')))
            digest=hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest,STOCK_KERNEL_AST_SHA256)


if __name__ == '__main__':
    unittest.main()
