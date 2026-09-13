"""Execute the native unpack AST against independent scalar byte equations."""
import ast
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'engine/kernels/b12x/moe_dynamic_gated_sf6_words.py'


def extract(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom('__future__', [ast.alias('annotations')], 0), *nodes],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace


def u32(value):
    return int(value) & 0xffffffff


class PrefillSF6WordTests(unittest.TestCase):
    def test_pinned_parent_and_producer_publication_are_unchanged(self):
        parent = SOURCE.with_name('moe_dynamic_gated_sf6.py')
        q0 = ast.parse(SOURCE.with_name('moe_dynamic_gated_sf6_q0.py').read_text())
        expected = next(ast.literal_eval(n.value) for n in q0.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id == 'SF6_SOURCE_SHA256' for t in n.targets))
        self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(), expected)
        original = ast.parse(parent.read_text())
        candidate = ast.parse(SOURCE.read_text())
        class RemoveWordArgument(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func,ast.Name) and node.func.id == '_sf6_expand_dynamic_tile':
                    self_word = node.args.pop()
                    if ast.unparse(self_word) != 'self.prefill_word_unpack':
                        raise AssertionError('unexpected producer specialization')
                return self.generic_visit(node)
        for name in ('load_fc1_tma_slice','load_fc2_tma_tile'):
            a = next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name == name)
            b = next(n for n in ast.walk(candidate) if isinstance(n,ast.FunctionDef) and n.name == name)
            b = RemoveWordArgument().visit(b)
            self.assertEqual(ast.dump(a,include_attributes=False), ast.dump(b,include_attributes=False))

    def word(self):
        return extract(SOURCE, {'_sf6_unpack_word'},
                       dict(cutlass=SimpleNamespace(Uint32=u32)))['_sf6_unpack_word']

    def test_bit_placement_and_all_base_delta_lanes(self):
        fn = self.word()
        def check(low, high, base):
            expected = sum(((base + ((low >> (4*i)) & 15)
                             + 16*((high >> (2*i)) & 3)) & 255) << (8*i)
                           for i in range(4))
            self.assertEqual(u32(fn(low, high, (base & 127)*0x01010101,
                                    (base & 128)*0x01010101)), expected)
        for low in range(65536):
            check(low, 0, 0)
        for high in range(256):
            check(0, high, 0)
        for base in range(256):
            for delta in range(64):
                for lane in range(4):
                    ds = [0, 63, 0, 63]
                    ds[lane] = delta
                    check(sum((d & 15) << (4*i) for i,d in enumerate(ds)),
                          sum((d >> 4) << (2*i) for i,d in enumerate(ds)), base)

    def test_actual_stage_matches_scalar_with_identical_addresses(self):
        rng = random.Random(913)
        start = 2**35 + 1552*719
        for fc2 in (False, True):
            for half in (0, 1):
                for base in (0, 73, 127, 128, 192, 240, 255):
                    ds = [rng.randrange(min(63, 255-base)+1) for _ in range(2048)]
                    raw = bytes(base+d for d in ds)
                    packed = bytearray(1552)
                    for i,d in enumerate(ds):
                        packed[i//2] |= (d & 15) << (4*(i%2))
                        packed[1024+i//4] |= (d >> 4) << (2*(i%4))
                    packed[1536] = base
                    packed[1537:] = rng.randbytes(15)
                    expected = bytes(raw[(i//512)*1024+half*512+i%512 if fc2 else half*1024+i]
                                     for i in range(1024))
                    previous_events = None
                    for word in (False, True):
                        output, events = bytearray(1024), []
                        def load(address):
                            offset = address-start
                            self.assertEqual(offset % 4, 0)
                            self.assertTrue(0 <= offset <= 1548)
                            events.append(('read', offset))
                            return int.from_bytes(packed[offset:offset+4], 'little', signed=True)
                        def store(address, value):
                            self.assertEqual(address % 4, 0)
                            self.assertTrue(0 <= address <= 1020)
                            events.append(('write', address))
                            output[address:address+4] = u32(value).to_bytes(4, 'little')
                        ns = extract(SOURCE, {'_sf6_unpack_word', '_sf6_expand_dynamic_tile'}, dict(
                            Int32=int, Int64=int, _sf6_ld_global_u32=load, _st_shared_i32=store,
                            cutlass=SimpleNamespace(Uint32=u32, const_expr=bool, range_constexpr=range),
                            cute=SimpleNamespace(make_rmem_tensor=lambda shape,dtype: [0]*shape[0])))
                        for lane in range(32):
                            ns['_sf6_expand_dynamic_tile'](start, 0, half, lane, fc2, word)
                        self.assertEqual(output, expected)
                        if previous_events is not None:
                            self.assertEqual(events, previous_events)
                        previous_events = events
                        self.assertEqual(sorted(v for kind,v in events if kind == 'write'), list(range(0,1024,4)))

    def test_dispatch_excludes_decode_short_q0_and_other_geometries(self):
        fn = extract(ROOT/'engine/kernels/b12x/moe_dispatch.py',
                     {'_long_prefill_sf6_word_unpack'}, {})['_long_prefill_sf6_word_unpack']
        args = dict(m=32256, E=288, k=4096, n=512, num_topk=8, tile_m=128,
                    quant_mode='nvfp4', tiled=True, reform_sf_pack=True,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                    swiglu_limit=10., ep_local=False, tp_sf6_q0=False,
                    share_input_across_experts=False)
        for m in (1,6,24,64,65,289,4095,8192,8193,16128,32256,32768,32769):
            self.assertEqual(fn(**dict(args,m=m)), 8192 < m <= 32768)
        for key,value in dict(m=32256., E=72, k=2048, n=2048, num_topk=1, tile_m=32,
                              quant_mode='w4a16', tiled=False, reform_sf_pack=False,
                              activation='silu', swiglu_alpha=1.702, swiglu_beta=1.,
                              swiglu_limit=7., ep_local=True, tp_sf6_q0=True,
                              share_input_across_experts=True).items():
            self.assertFalse(fn(**dict(args,**{key:value})), key)


if __name__ == '__main__':
    unittest.main()
