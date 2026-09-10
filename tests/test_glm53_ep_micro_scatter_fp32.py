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
import runpy
import struct
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
MICRO = ROOT/'overlay/modules/glm53_moe/moe_micro_kernel.py'
ORACLE = ROOT/'measurements/glm53_ep_local_20260908/micro-stock-oracle'
OWNERSHIP = ROOT/'measurements/glm53_ep_local_20260908/micro-scatter-ownership'
STOCK_SOURCE_SHA256 = 'a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1'
# Frozen CPU11/onepass11 source, parsed under the same Python as the candidate.
# ast.dump's empty-field representation differs between Python 3.12 and 3.14.
STOCK_KERNEL_SOURCE_SHA256 = '70b9f9f75c67d854da25ba36e1943af1182ea0155e1d393994eb1e4aa4178655'


def function(name):
    node = next(n for n in ast.walk(ast.parse(MICRO.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    if name == 'kernel':
        # This module preserves the independent barrier-based FP32 control and
        # the original BF16 path. Direct-register behavior has its own tests.
        class SelectBarrierPath(ast.NodeTransformer):
            def selected(self, test):
                value = ast.unparse(test)
                if value in ('self.shared_fc1_a',
                             'cutlass.const_expr(self.shared_fc1_a)',
                             'self.ep_m16', 'cutlass.const_expr(self.ep_m16)',
                             'cutlass.const_expr(self.ep_direct_scatter)'):
                    return False
                if value in ('not self.shared_fc1_a',
                             'cutlass.const_expr(not self.shared_fc1_a)',
                             'not self.ep_m16', 'cutlass.const_expr(not self.ep_m16)'):
                    return True
                return None

            def body(self, statements):
                result = []
                for statement in statements:
                    selected = self.visit(statement)
                    if isinstance(selected, list):
                        result.extend(selected)
                    elif selected is not None:
                        result.append(selected)
                return result

            def visit_If(self, item):
                enabled = self.selected(item.test)
                if enabled is not None:
                    return self.body(item.body if enabled else item.orelse)
                return self.generic_visit(item)

            def visit_IfExp(self, item):
                enabled = self.selected(item.test)
                if enabled is not None:
                    return self.visit(item.body if enabled else item.orelse)
                return self.generic_visit(item)
        node = SelectBarrierPath().visit(node)
    return node


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


def scatter_block(fn):
    return next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                and any(isinstance(child, ast.While)
                        and ast.unparse(child.test) == 'pair_idx < warp_epi_rows * Int32(32)'
                        for child in n.body))


def publication_if(block):
    return next(n for n in block.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'cutlass.const_expr(self.scatter_fp32)'
                and len(n.body) == 1 and isinstance(n.body[0], ast.Expr)
                and ast.unparse(n.body[0]) == 'self.epilog_sync_barrier.arrive_and_wait()')


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

    def test_original_ptx_proves_cross_warp_reads_need_publication(self):
        proof = runpy.run_path(str(OWNERSHIP/'verify.py'))['verify'](OWNERSHIP)
        self.assertEqual(proof['verdict'], 'PASS')
        self.assertEqual(set(proof['variants']), {'m32-topk8-fp32', 'm64-topk1-fp32'})
        for result in proof['variants'].values():
            self.assertEqual(result['pair_reads'], 512)
            self.assertEqual(result['cross_warp_pair_reads'], 384)
            self.assertEqual(result['witnesses'][0],
                             dict(reader_tid=8, producer_tid=64, row=0, column=16,
                                  shared_offsets=[57376, 57378]))

    def test_ep_publication_precedes_every_scatter_and_retains_post_barrier(self):
        block = scatter_block(function('kernel'))
        guard = publication_if(block)
        idx = block.body.index(guard)
        self.assertEqual(ast.unparse(block.body[idx-1]),
                         "cute.arch.fence_proxy('async.shared', space='cta')")
        self.assertTrue(ast.unparse(block.body[idx-2]).startswith('cute.copy('))
        self.assertEqual(ast.unparse(block.body[idx+1]),
                         'rows_offset = Int32(epi_m) * Int32(self.epi_tile[0])')
        self.assertFalse(guard.orelse)
        self.assertEqual(ast.unparse(block.body[-1]),
                         'self.epilog_sync_barrier.arrive_and_wait()')
        # Simulate a delayed producer warp. The generic/async fence alone does
        # not release its pending stores; the actual selected barrier must.
        prefix = compile(ast.Module(body=block.body[idx-2:idx+1], type_ignores=[]),
                         str(MICRO), 'exec')
        post = compile(ast.Module(body=[block.body[-1]], type_ignores=[]), str(MICRO), 'exec')
        class Slice:
            def __getitem__(self, key): return key
        for enabled in (False, True):
            events, pending, shared = [], {}, {}
            def store(*args):
                events.append('store'); pending[57376] = 'producer-warp2'
            def fence(*args, **kwargs): events.append('fence')
            def barrier():
                events.append('barrier'); shared.update(pending); pending.clear()
            ns = dict(cute=SimpleNamespace(copy=store, arch=SimpleNamespace(fence_proxy=fence)),
                      cutlass=SimpleNamespace(const_expr=lambda x:x),
                      self=SimpleNamespace(scatter_fp32=enabled,
                                           epilog_sync_barrier=SimpleNamespace(arrive_and_wait=barrier)),
                      tiled_copy_r2s=None, tRS_rD_out=None, tRS_sD=Slice(), epi_buffer=0)
            exec(prefix, ns)
            self.assertEqual(57376 in shared, enabled)
            self.assertEqual(events, ['store', 'fence'] + (['barrier'] if enabled else []))
            events.append('scatter')
            exec(post, ns)
            self.assertEqual(events[-2:], ['scatter', 'barrier'])

    def test_ep_scatter_bounds_cover_every_route_exactly_once(self):
        fn = function('kernel')
        parent = next(n for n in ast.walk(fn) if isinstance(n, ast.While)
                      and any(isinstance(x, ast.Assign)
                              and ast.unparse(x.targets[0]) == 'valid_tile_rows'
                              for x in n.body))
        idx = next(i for i,n in enumerate(parent.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == 'valid_tile_rows')
        valid_code = compile(ast.Module(body=parent.body[idx:idx+3], type_ignores=[]),
                             str(MICRO), 'exec')
        block = scatter_block(fn)
        row_guard = next(n for n in block.body if isinstance(n, ast.If)
                         and ast.unparse(n.test) == 'cutlass.const_expr(self.scatter_fp32)'
                         and isinstance(n.body[0], ast.Assign))
        row_idx = block.body.index(row_guard)
        row_code = compile(ast.Module(body=block.body[row_idx:row_idx+3], type_ignores=[]),
                           str(MICRO), 'exec')
        for tile_m in (32, 64):
            for count in (0, 1, 8, 31, 32, 33, 48, 63, 64):
                observed = []
                for tile in range((count+tile_m-1)//tile_m):
                    tile_base = tile*tile_m
                    for warp in range(4):
                        ns = dict(valid_rows=count, tile_m_base=tile_base, rows_offset=0,
                                  warp_m_base=(warp//2)*64, Int32=int,
                                  self=SimpleNamespace(scatter_fp32=True,
                                                       tile_shape_mnk=(tile_m,128,128)),
                                  cutlass=SimpleNamespace(const_expr=lambda x:x))
                        exec(valid_code, ns); exec(row_code, ns)
                        for row in range(ns['warp_epi_rows']):
                            logical_row = ns['warp_m_base']+row
                            self.assertLess(logical_row, tile_m)
                            self.assertLess(logical_row, ns['valid_tile_rows'])
                            for col in range((warp%2)*64, (warp%2+1)*64):
                                observed.append((tile_base+logical_row, col))
                self.assertEqual(len(observed), count*128)
                self.assertEqual(set(observed), {(r,c) for r in range(count) for c in range(128)})

    def test_entire_kernel_preserves_stock_except_exact_ep_scatter_changes(self):
        raw = gzip.decompress((ORACLE/'moe_micro_kernel_cpu11.py.gz').read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), STOCK_KERNEL_SOURCE_SHA256)
        identity = json.loads((ORACLE/'micro-kernel-cpu11-identity.json').read_text())
        self.assertEqual(identity['source_sha256'], STOCK_KERNEL_SOURCE_SHA256)
        self.assertEqual(identity['revision'], 'c7dec80a0f73d4a2b683ce2c4813978938694095')
        stock = next(n for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.FunctionDef)
                     and n.name == 'kernel')
        expected = hashlib.sha256(ast.dump(stock, include_attributes=False).encode()).hexdigest()
        current = function('kernel')
        block = scatter_block(current)
        pre = publication_if(block)
        bounded = next(n for n in block.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == 'cutlass.const_expr(self.scatter_fp32)'
                       and isinstance(n.body[0], ast.Assign))
        self.assertEqual(ast.unparse(bounded.body[0]),
                         'warp_epi_rows = valid_tile_rows - rows_offset - warp_m_base')
        self.assertEqual(ast.unparse(bounded.orelse[0]),
                         'warp_epi_rows = valid_rows - tile_m_base - rows_offset - warp_m_base')
        class SelectScatter(ast.NodeTransformer):
            def __init__(self, enabled): self.enabled=enabled
            def visit_If(self, node):
                if node.lineno == pre.lineno:
                    return []  # Only the exact, separately verified publication.
                if node.lineno == bounded.lineno:
                    return [self.visit(n) for n in node.orelse]
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
            normalized=SelectScatter(enabled).visit(copy.deepcopy(current))
            digest=hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest, expected)


if __name__ == '__main__':
    unittest.main()
