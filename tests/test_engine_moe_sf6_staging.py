"""Execute the engine's exact SF6 helper under adversarial shared-memory schedules."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'engine/kernels/b12x/moe_static_kernel_v4.py'
TREE = ast.parse(SOURCE.read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef))


class SuspendMemory(ast.NodeTransformer):
    """One scheduler step per actual load/store, including cross-warp interleavings."""
    def visit_Call(self, node):
        name = ast.unparse(node.func)
        event = {'_ld_shared_i32_volatile': 'load', '_st_shared_i32': 'store',
                 'self.sf_expand_barrier.arrive_and_wait': 'barrier'}.get(name)
        if event:
            return ast.copy_location(ast.Yield(ast.Tuple(
                elts=[ast.Constant(event), *node.args], ctx=ast.Load())), node)
        return self.generic_visit(node)


def method(name, env, *, suspend=False):
    node = copy.deepcopy(next(n for n in CLASS.body
                             if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    if suspend:
        node = SuspendMemory().visit(node)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), env)
    return env[name]


def geometry(reform=True, packed=True, separate=True, stages=2, word_expand=True,
             fc2_word_expand=True, reuse=True, compact=True, fc2_stages=None, registers=False, sync_cleanup=True,
             **flags):
    env = dict(cutlass=SimpleNamespace(Float32=object()), DenseGemmKernel=object(),
        utils=SimpleNamespace(get_smem_capacity_in_bytes=lambda _: 101376),
        pipeline=SimpleNamespace(NamedBarrier=lambda **kw: SimpleNamespace(**kw)),
        is_gated_activation=lambda _: True)
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            try:
                env[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, AttributeError):
                pass
    owner = SimpleNamespace()
    method('__init__', env)(owner, 16, 4, decode_reform=reform,
        reform_sf_pack=packed, sf6_separate=separate, fc1_stages=stages,
        fc2_stages=stages if fc2_stages is None else fc2_stages,
        sf6_word_expand=word_expand, sf6_fc2_word_expand=fc2_word_expand,
        fc1_reuse_a=reuse, compact_staging=compact, sf6_registers=registers, sync_cleanup=sync_cleanup,
        **flags)
    slot = method('_fc1_input_slot', dict(Int32=int))
    owner._fc1_input_slot = lambda stage: slot(owner, stage)
    return owner


def packed_codes(base, size, seed):
    """Independent bit-plane encoder, covering all 64 codes and byte wraparound."""
    codes = [(i * 31 + seed) % 64 for i in range(size)]
    packed = bytearray(size*3//4 + 16)
    for i, code in enumerate(codes):
        packed[i//2] |= (code & 15) << (4*(i % 2))
        packed[size//2+i//4] |= (code >> 4) << (2*(i % 4))
    packed[size*3//4] = base
    return bytes(packed), bytes((base+code) % 256 for code in codes)


def packed_add(a, b):
    # Interpret only the PTX instruction's independent byte-lane contract.
    return sum((((a >> shift) & 255) + ((b >> shift) & 255)) % 256 << shift
               for shift in (0, 8, 16, 24))


def expand(mem, dest, size, source=None, seed=0, word_expand=True, word_override=None):
    helper = method('_sf_expand_stage', dict(Int32=int), suspend=True)
    word = method('_sf6_expand_word', dict(Int32=int, add_u8x4=packed_add))
    owner = SimpleNamespace(sf6_word_expand=word_expand,
        _sf6_expand_word=lambda *args: word(None, *args))
    workers = [helper(owner, dest, t, size, packed_addr=source, word_expand=word_override)
               for t in range(128)]
    ready, blocked = list(range(128)), []
    replies = [None]*128
    rng = random.Random(seed)
    loads, stores, barriers = [], [], []
    while ready or blocked:
        if not ready:
            assert len(blocked) == 128, 'a peer escaped the publication barrier'
            barriers.append(len(stores))
            ready, blocked = blocked, []
        at = rng.randrange(len(ready))
        t = ready[at]
        try:
            event = workers[t].send(replies[t])
        except StopIteration:
            ready.pop(at)
            continue
        replies[t] = None
        if event[0] == 'barrier':
            blocked.append(ready.pop(at))
            continue
        addr = event[1]
        assert addr % 4 == 0 and 0 <= addr <= len(mem)-4
        if event[0] == 'load':
            loads.append(addr)
            replies[t] = int.from_bytes(mem[addr:addr+4], 'little', signed=True)
        else:
            stores.append(addr)
            mem[addr:addr+4] = (event[2] & 0xFFFFFFFF).to_bytes(4, 'little')
    assert barriers == ([0, size//4] if source is None else [size//4]), barriers
    assert sorted(stores) == list(range(dest, dest+size, 4))
    input_base = dest if source is None else source
    assert all(input_base <= addr <= input_base+size*3//4 for addr in loads)
    return len(barriers)


class Sf6StagingTests(unittest.TestCase):
    def test_fc2_inplace_word_all_bases_codes_and_cross_warp_orders(self):
        size, dest = 2048, 256
        for base in range(256):
            packed, expected = packed_codes(base, size, base*17)
            reference = bytearray([0xA5])*8192
            reference[dest:dest+len(packed)] = packed
            for word in (False, True):
                mem = bytearray(reference)
                expand(mem, dest, size, seed=base, word_override=word)
                self.assertEqual(mem[dest:dest+size], expected)
                self.assertEqual(mem[:dest], reference[:dest])
                self.assertEqual(mem[dest+size:], reference[dest+size:])

    def test_fc2_inplace_word_ring_reuse_and_neighbor_slots(self):
        # Simulate the producer refilling a consumed slot while other slots
        # retain their prior packed or expanded bytes. Every expansion must
        # finish its packed reads before the first in-place overwrite.
        for stages in (1, 2, 3):
            mem = bytearray([0xA5])*16384
            for generation in range(stages*8):
                dest = 256+(generation % stages)*2048
                packed, expected = packed_codes(generation*37 % 256, 2048, generation*13)
                mem[dest:dest+len(packed)] = packed
                before = bytes(mem)
                expand(mem, dest, 2048, seed=generation, word_override=True)
                self.assertEqual(mem[dest:dest+2048], expected)
                self.assertEqual(mem[:dest], before[:dest])
                self.assertEqual(mem[dest+2048:], before[dest+2048:])

    def test_separate_all_bases_codes_and_cross_warp_orders(self):
        size, dest, source = 2048, 256, 4096
        for word_expand in (False, True):
            for base in range(256):
                packed, expected = packed_codes(base, size, base*17)
                mem = bytearray([0xA5])*8192
                mem[source:source+len(packed)] = packed
                before = bytes(mem)
                expand(mem, dest, size, source, seed=base, word_expand=word_expand)
                self.assertEqual(mem[dest:dest+size], expected)
                self.assertEqual(mem[:dest], before[:dest])
                self.assertEqual(mem[dest+size:], before[dest+size:])

    def test_word_arithmetic_all_byte_values_without_cross_lane_carries(self):
        word = method('_sf6_expand_word', dict(Int32=int, add_u8x4=packed_add))
        for base in range(256):
            for lane in range(4):
                for code in range(64):
                    codes = [63, 32, 31, 0]
                    codes[lane] = code
                    low = sum((v & 15) << (4*i) for i, v in enumerate(codes))
                    high = sum((v >> 4) << (2*i) for i, v in enumerate(codes))
                    actual = word(None, low, high, base*0x01010101)
                    expected = sum(((base+v) % 256) << (8*i) for i, v in enumerate(codes))
                    self.assertEqual(actual & 0xFFFFFFFF, expected, (base, lane, code))

    def test_sass_count_keeps_native_packed_byte_instructions(self):
        from probes.engine_moe_sf6_compile import instruction_opcodes
        sass = '''/*0170*/ VIADD.U8x4 R9, R9, R0;
                  /*0180*/ @!P0 VIADD.U8x4 R8, R7, R6;
                  /*0190*/ EXIT;'''
        self.assertEqual(instruction_opcodes(sass), ['VIADD.U8x4', 'VIADD.U8x4', 'EXIT'])

    def test_fc1_ring_reuse_and_neighbor_slots(self):
        for stages in (1, 2, 3):
            g = geometry(stages=stages)
            self.assertTrue(g.sf6_separate)
            self.assertEqual(g.sf6_packed_bytes, stages*1552)
            mem = bytearray([0xA5])*32768
            for generation in range(stages*4):
                slot = generation % stages
                for kind, stage_bytes, dest_base, source_base in (
                    ('fc1', g.sf1_stage_bytes, 256, 16384),):
                    dest, source = dest_base+slot*2048, source_base+slot*stage_bytes
                    packed, expected = packed_codes(generation*17, 2048, generation+dest_base)
                    self.assertEqual(len(packed), stage_bytes, kind)
                    self.assertEqual(source % 16, 0)
                    mem[source:source+len(packed)] = packed
                    before = bytes(mem)
                    expand(mem, dest, 2048, source, seed=generation)
                    self.assertEqual(mem[dest:dest+2048], expected, kind)
                    self.assertEqual(mem[:dest], before[:dest])
                    self.assertEqual(mem[dest+2048:], before[dest+2048:])

    def test_legacy_inplace_keeps_both_barriers_and_exact_bytes(self):
        for size in (1024, 2048, 4096):
            for base in (0, 64, 128, 192, 255):
                packed, expected = packed_codes(base, size, base)
                mem = bytearray([0xA5])*(size+1024)
                mem[256:256+len(packed)] = packed
                before = bytes(mem)
                expand(mem, 256, size, seed=base)
                self.assertEqual(mem[256:256+size], expected)
                self.assertEqual(mem[:256], before[:256])
                self.assertEqual(mem[256+size:], before[256+size:])

    def test_only_c1_sf6_geometry_uses_separate_storage(self):
        for reform in (False, True):
            for packed in (False, True):
                for separate in (False, True):
                    g = geometry(reform, packed, separate)
                    self.assertEqual(g.sf6_separate, reform and packed and separate)
                    self.assertEqual(g.scatter_cache_rows, 16 if g.sf6_separate else 128)
                    self.assertEqual(g.sf6_word_expand, g.sf6_separate)
                    self.assertEqual(g.sf6_fc2_word_expand, reform and packed)
        self.assertFalse(geometry(word_expand=False).sf6_word_expand)
        self.assertFalse(geometry(fc2_word_expand=False).sf6_fc2_word_expand)

    def test_actual_scatter_initialization_bounds_and_changed_routes(self):
        kernel = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
        block = next(n for n in ast.walk(kernel) if isinstance(n, ast.While)
                     and ast.unparse(n.test) == 'is_valid_tile')
        start = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'cache_row')
        code = compile(ast.Module(body=copy.deepcopy(block.body[start:start+2]),
                                  type_ignores=[]), str(SOURCE), 'exec')

        class Scalar:
            def __init__(self, value):
                self.value = value
            def to(self, dtype):
                return dtype(self.value)

        class Matrix:
            def __init__(self, values):
                self.values = values
            def __getitem__(self, index):
                expert, row = index
                assert expert == 2
                return Scalar(self.values[row])

        for valid in range(17):
            for generation in (0, 1):
                tokens, weights = [-1]*32, [-1]*32
                def store(buffer, addr, value):
                    self.assertLess(addr, 16*4, 'write escaped the M16 cache')
                    buffer[addr//4] = value
                offset = 16
                ids = list(range(offset+valid)) if generation == 0 else list(reversed(range(offset+valid)))
                scales = [i/17 for i in ids]
                env = dict(self=geometry(), Int32=int, cutlass=SimpleNamespace(Float32=float),
                    valid_tile_rows=valid, local_expert_idx=2, tile_m_base=offset,
                    token_map=Matrix(ids), token_weights=Matrix(scales),
                    scatter_tok_base_addr=0, scatter_weight_base_addr=0,
                    _st_shared_i32=lambda addr, val: store(tokens, addr, val),
                    _st_shared_f32=lambda addr, val: store(weights, addr, val))
                for tidx in range(128):
                    exec(code, dict(env, tidx=tidx))
                self.assertEqual(tokens[:16], ids[offset:]+[0]*(16-valid))
                self.assertEqual(weights[:16], scales[offset:]+[0.]*(16-valid))
                self.assertEqual(tokens[16:], [-1]*16)
                self.assertEqual(weights[16:], [-1]*16)

    def test_dispatch_default_rollback_cache_and_repeated_normalization(self):
        path = ROOT / 'engine/kernels/b12x/moe_dispatch.py'
        tree = ast.parse(path.read_text())
        names = {'_static_v2_decode_config', '_static_v2_cache_key'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = dict(Tuple=tuple, _static_kernel_cache_key=lambda **kw: tuple(sorted(kw.items())))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        base = dict(tile_m=32, fc1=2, fc2=2, a_rows=16, stamps=False,
                    decode_reform=True, reform_sf_pack=True)
        for rows in (1, 6, 7, 8, 9, 14, 16, 21, 24, 28, 32, 128):
            selected = normalize(base, rows)
            rollback = normalize(dict(base, sf6_separate=False), rows)
            scalar = normalize(dict(base, sf6_word_expand=False), rows)
            fc2_scalar = normalize(dict(base, sf6_fc2_word_expand=False), rows)
            self.assertEqual(selected['sf6_separate'], rows <= 8)
            self.assertEqual(selected['sf6_word_expand'], rows <= 8)
            self.assertEqual(selected['sf6_fc2_word_expand'], rows <= 8)
            self.assertEqual(rollback['sf6_fc2_word_expand'], rows <= 8)
            self.assertEqual(scalar['sf6_fc2_word_expand'], rows <= 8)
            self.assertFalse(fc2_scalar['sf6_fc2_word_expand'])
            self.assertFalse(scalar['sf6_word_expand'])
            self.assertFalse(rollback['sf6_word_expand'])
            self.assertEqual(normalize(selected, rows), selected)
            self.assertEqual(normalize(rollback, rows), rollback)
            self.assertEqual(key(selected, m=rows) == key(rollback, m=rows), rows > 8)
            self.assertEqual(key(selected, m=rows) == key(scalar, m=rows), rows > 8)
            self.assertEqual(normalize(scalar, rows), scalar)
            self.assertEqual(normalize(fc2_scalar, rows), fc2_scalar)
            self.assertEqual(key(selected, m=rows) == key(fc2_scalar, m=rows), rows > 8)
        self.assertFalse(normalize(dict(base, reform_sf_pack=False), 8)['sf6_separate'])
        self.assertFalse(normalize(dict(base, reform_sf_pack=False), 8)['sf6_fc2_word_expand'])


if __name__ == '__main__':
    unittest.main()
