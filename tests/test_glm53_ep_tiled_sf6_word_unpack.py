"""CPU checks of actual SF6 word arithmetic and in-place shared ownership.

No accelerator imports or launches. CuTe lowering and the unchanged startup
numerical/graph gates remain required before any throughput interpretation.
"""
import ast
import copy
import random
import types
import unittest

from test_glm53_ep_tiled_static import SOURCE, STOCK, extract, function, route_helpers


def u32(value):
    return int(value) & 0xFFFFFFFF


def i32(value):
    value = u32(value)
    return value if value < (1 << 31) else value - (1 << 32)


def word_function():
    return extract('_sf6_unpack_word', {'cutlass': types.SimpleNamespace(Uint32=u32)})


def reference_word(low, high, base):
    # Scalar encoding equation; does not share the candidate's spreading or
    # packed addition. Deliberately covers invalid/overflowing byte codes too.
    return sum(((base + ((low >> (4 * lane)) & 15)
                 + 16 * ((high >> (2 * lane)) & 3)) & 255) << (8 * lane)
               for lane in range(4))


def run_word(fn, low, high, base):
    return fn(low, high, (base & 127) * 0x01010101,
              (base & 128) * 0x01010101)


def expansion_class():
    """Execute the real constructor/override against an inert parent."""
    class Parent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def _sf_expand_stage(self, *args):
            return ('stock', args)

    ns = dict(Parent=Parent, ep_tiled_source_contract=lambda: None)
    extract('ep_tiled_geometry', ns)
    extract('ep_tiled_scale_mode', ns)
    route_helpers(ns)
    cls = ast.ClassDef(name='Candidate', bases=[ast.Name('Parent', ast.Load())],
                      keywords=[], body=[function('__init__'),
                                         function('_sf_expand_stage')], decorator_list=[])
    mod = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(SOURCE), 'exec'), ns)
    return ns['Candidate']


def run_stage(payload, *, stock=False, seed=0, remove_read_barrier=False):
    """Run all 128 real thread bodies across each collective barrier.

    The first barrier must see every thread's compressed reads. The second
    publishes all stores. Shared storage is overwritten in place, so a
    premature store is rejected rather than concealed by a separate input.
    """
    body = function('_sf_expand_stage', STOCK if stock else SOURCE)

    class YieldBarriers(ast.NodeTransformer):
        def __init__(self): self.count = 0
        def visit_Expr(self, node):
            if ast.unparse(node.value) == 'self.sf_expand_barrier.arrive_and_wait()':
                self.count += 1
                if remove_read_barrier and self.count == 1:
                    return ast.Pass()
                return ast.Expr(ast.Yield(ast.Constant(self.count)))
            return self.generic_visit(node)

    body = YieldBarriers().visit(body)
    shared = bytearray(payload)
    events = []
    active = [-1]
    phase = [0]

    def load(addr):
        if phase[0] != 0:
            raise AssertionError('compressed read after read barrier')
        if not 0 <= addr <= 1548 or addr % 4:
            raise AssertionError('out-of-range or unaligned compressed read')
        events.append(('load', active[0], addr))
        return int.from_bytes(shared[addr:addr + 4], 'little', signed=True)

    def store(addr, value):
        if phase[0] != 1:
            raise AssertionError('in-place write before all compressed reads')
        if not 0 <= addr <= 2044 or addr % 4:
            raise AssertionError('out-of-range or unaligned expanded write')
        events.append(('store', active[0], addr))
        shared[addr:addr + 4] = u32(value).to_bytes(4, 'little')

    ns = dict(Int32=i32, cutlass=types.SimpleNamespace(Uint32=u32),
              _sf6_unpack_word=word_function(), _ld_shared_i32_volatile=load,
              _st_shared_i32=store)
    mod = ast.Module(body=[body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(SOURCE), 'exec'), ns)
    owner = types.SimpleNamespace(word_unpack=True)
    threads = [ns['_sf_expand_stage'](owner, 0, lane, 2048) for lane in range(128)]
    order = list(range(128)); random.Random(seed).shuffle(order)
    for lane in order:
        active[0] = lane
        if next(threads[lane]) != 1:
            raise AssertionError('missing read-before-write barrier')
    phase[0] = 1
    random.Random(seed + 1).shuffle(order)
    for lane in order:
        active[0] = lane
        if next(threads[lane]) != 2:
            raise AssertionError('missing publication barrier')
    phase[0] = 2
    for lane, thread in enumerate(threads):
        active[0] = lane
        try:
            next(thread)
        except StopIteration:
            pass
        else:
            raise AssertionError('unexpected extra phase')
    return bytes(shared), events


class EPTiledSF6WordUnpackTests(unittest.TestCase):
    def test_all_low_and_high_bit_placements_match_scalar_encoding(self):
        word = word_function()
        for low in range(1 << 16):
            self.assertEqual(run_word(word, low, 0, 0), reference_word(low, 0, 0))
        for high in range(1 << 8):
            self.assertEqual(run_word(word, 0, high, 0), reference_word(0, high, 0))

    def test_every_base_delta_and_lane_preserves_modulo_without_neighbour_carry(self):
        word = word_function()
        for base in range(256):
            for delta in range(64):
                # Move the exhaustive value across every byte, alongside
                # alternating extreme neighbours that expose leaked carries.
                for lane in range(4):
                    ds = [0, 63, 0, 63]; ds[lane] = delta
                    low = sum((d & 15) << (4 * i) for i, d in enumerate(ds))
                    high = sum((d >> 4) << (2 * i) for i, d in enumerate(ds))
                    self.assertEqual(run_word(word, low, high, base),
                                     reference_word(low, high, base))
        # Actual shared loads are signed i32; unused upper bits must not leak.
        rng = random.Random(731)
        for _ in range(2048):
            low, high, base = rng.getrandbits(32), rng.getrandbits(32), rng.randrange(256)
            self.assertEqual(run_word(word, i32(low), i32(high), base),
                             reference_word(low, high, base))

    def test_actual_stage_bytes_and_thread_ownership_match_pinned_stock(self):
        rng = random.Random(510)
        for seed, base in enumerate((0, 63, 127, 128, 192, 255)):
            payload = bytearray(rng.randbytes(2048))
            payload[1536] = base  # Other tail bytes are deliberately poisoned.
            expected = bytes((base + ((payload[i // 2] >> (4 * (i % 2))) & 15)
                              + 16 * ((payload[1024 + i // 4] >> (2 * (i % 4))) & 3)) & 255
                             for i in range(2048))
            actual, events = run_stage(payload, seed=seed)
            stock, stock_events = run_stage(payload, stock=True, seed=seed)
            self.assertEqual(actual, expected)
            self.assertEqual(actual, stock)
            self.assertEqual(events, stock_events)
            self.assertEqual(len([e for e in events if e[0] == 'load']), 512)
            writes = [e[2] for e in events if e[0] == 'store']
            self.assertEqual(sorted(writes), list(range(0, 2048, 4)))

    def test_interleaving_oracle_rejects_removed_read_barrier(self):
        with self.assertRaisesRegex(AssertionError, 'before all compressed reads'):
            run_stage(bytes(2048), remove_read_barrier=True)

    def test_actual_constructor_selects_only_native_sf6_and_delegates_other_modes(self):
        cls = expansion_class()
        for m in range(1, 33):
            for sf6 in (False, True):
                kernel = cls(num_tokens=m, max_rows=256, max_active_clusters=48,
                             reform_sf_pack=sf6)
                selected = sf6 and m <= 8
                self.assertIs(kernel.word_unpack, selected)
                self.assertEqual((kernel.ep_route_mode,kernel.ep_route_map_len,kernel.ep_local_expert_offset),
                                 ('local',None,0))
                self.assertEqual((kernel.reform_sf_pack, kernel.decode_reform), (sf6, m <= 8))
                if not selected:
                    for size in (1024, 2048, 4096):
                        self.assertEqual(kernel._sf_expand_stage(32, 17, size),
                                         ('stock', (32, 17, size)))

    def test_selected_expansion_rejects_wrong_stage_size_before_any_read(self):
        kernel = expansion_class()(num_tokens=6, max_rows=256,
                                   max_active_clusters=48, reform_sf_pack=True)
        for size in (0, 1024, 1552, 4096):
            with self.assertRaisesRegex(ValueError, '2048-byte'):
                kernel._sf_expand_stage(0, 0, size)


if __name__ == '__main__':
    unittest.main()
