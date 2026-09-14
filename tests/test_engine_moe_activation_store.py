"""Exact C1 packed activation stores against FC2's independent byte mapping."""
from __future__ import annotations

import random
from types import SimpleNamespace
import unittest

from tests.test_engine_moe_sf6_staging import method, geometry
from tests.test_engine_moe_scatter_config import namespace


class ActivationStoreTests(unittest.TestCase):
    def test_changed_tiles_partial_rows_and_canaries(self):
        rng = random.Random(915)
        base, extent = 1024, 4096
        layout = SimpleNamespace(outer=object())
        reference = bytearray([0xA5])*extent
        memories = {flag: bytearray(reference) for flag in (False, True)}
        # Partial rows and repeated reuse must leave every inactive row intact.
        for generation, valid in enumerate(list(range(17)) + list(reversed(range(17)))):
            work = [(r, b) for r in range(valid) for b in range(8)]
            rng.shuffle(work)
            values = {(r, b): rng.getrandbits(64) for r, b in work}
            if work:
                values[work[0]] = 1 << (generation % 64)
                values[work[-1]] = 0xFFFFFFFFFFFFFFFF if generation % 2 else 0
            for (r, b), value in values.items():
                # Reference is the consumer row-major layout with its row XOR,
                # not the producer helper's outer-layout/swizzle calculation.
                for i, byte in enumerate(value.to_bytes(8, 'little')):
                    offset = r*64 + ((b*8+i) ^ (((r >> 1) & 3) << 4))
                    reference[base+offset] = byte
            for packed_store, memory in memories.items():
                writes = []
                def store(addr, value, size):
                    self.assertEqual(addr % size, 0)
                    self.assertGreaterEqual(addr, base)
                    self.assertLessEqual(addr+size, base+1024)
                    writes.extend(range(addr, addr+size))
                    memory[addr:addr+size] = value.to_bytes(size, 'little')
                owner = SimpleNamespace(packed_activation_store=packed_store, sf_vec_size=16)
                helper = method('_store_packed_activation', dict(Int32=int, Uint64=int, Uint8=int,
                    cutlass=SimpleNamespace(range_constexpr=range),
                    cute=SimpleNamespace(crd2idx=lambda p, _: p[0]*128+p[1]),
                    _st_shared_u64=lambda addr, val: store(addr, val, 8),
                    st_shared_u8=lambda addr, val: store(addr, val, 1)))
                for r, b in work:
                    helper(owner, base, layout, r, b*8, values[r, b])
                self.assertEqual(len(writes), valid*64)
                self.assertEqual(len(set(writes)), len(writes), 'two blocks overlap')
                self.assertEqual(memory, reference, (generation, valid, packed_store))

    def test_all_64_payload_bits_survive_each_swizzled_block(self):
        layout = SimpleNamespace(outer=None)
        writes = []
        helper = method('_store_packed_activation', dict(Int32=int,
            cute=SimpleNamespace(crd2idx=lambda p, _: p[0]*128+p[1]),
            _st_shared_u64=lambda addr, val: writes.append((addr, val))))
        for row in range(16):
            for block in range(8):
                for bit in range(64):
                    writes.clear()
                    helper(SimpleNamespace(packed_activation_store=True), 1024,
                           layout, row, block*8, 1 << bit)
                    self.assertEqual(writes, [(1024+row*64+((block*8)^(((row >> 1)&3)<<4)), 1 << bit)])

    def test_default_and_control_have_distinct_idempotent_cache_keys(self):
        ns = namespace()
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        for recipe in ('t,r', 't,r,sf6', 't'):
            config = ns['_parse_glm53_static_v2'](recipe)
            for rows in (0, 1, 6, 7, 8, 9, 14, 16, 21, 24, 28, 32, 128):
                selected = normalize(config, rows)
                control = normalize(dict(config, packed_activation_store=False), rows)
                enabled = config.get('decode_reform', False) and 1 <= rows <= 8
                self.assertEqual(selected['packed_activation_store'], enabled)
                self.assertFalse(control['packed_activation_store'])
                self.assertEqual(normalize(selected, rows), selected)
                self.assertEqual(normalize(control, rows), control)
                self.assertEqual(key(selected, m=rows) != key(control, m=rows), enabled)
        self.assertTrue(geometry().packed_activation_store)
        self.assertFalse(geometry(reform=False).packed_activation_store)


if __name__ == '__main__':
    unittest.main()
