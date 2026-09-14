"""Exact SF6 operand words and lifetime of the C1 direct-register path."""
import ast
import copy
from types import SimpleNamespace
import unittest

from tests.test_engine_moe_fc1_reuse import execute
from tests.test_engine_moe_scatter_config import namespace
from tests.test_engine_moe_sf6_staging import CLASS, SOURCE, geometry, method, packed_codes


def load_words(memory, packed_base, offsets, tidbase=0):
    """Execute the production helper with explicit physical copy-word offsets."""
    reads = []
    def load(address):
        assert address % 4 == 0
        assert packed_base <= address <= packed_base + 1536
        reads.append(address)
        return int.from_bytes(memory[address:address+4], 'little', signed=True)
    destinations = [[None]*len(offsets) for _ in range(4)]
    cute = SimpleNamespace(recast_tensor=lambda tensor, dtype: tensor,
                           size=lambda tensor: len(offsets))
    word = method('_sf6_expand_word', dict(Int32=int))
    owner = SimpleNamespace(_sf6_expand_word=lambda *args: word(None, *args),
                            sf6_register_offsets={"test": [[[offset - ((0, 256, 128, 384)[tidbase//32]
                                + (tidbase % 32)//4*16) for offset in offsets]]]})
    class Shuffle(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node.func) == 'cute.arch.shuffle_sync':
                return ast.copy_location(ast.Yield(ast.Tuple(elts=node.args, ctx=ast.Load())), node)
            return self.generic_visit(node)
    node = copy.deepcopy(next(n for n in CLASS.body
        if isinstance(n, ast.FunctionDef) and n.name == '_sf6_load_fragment'))
    node = Shuffle().visit(node)
    env = dict(cute=cute, Int32=int, _ld_shared_i32_volatile=load,
               shared_ptr_to_u32=lambda pointer: pointer)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), 'exec'), env)
    # Four actual helper instances, one per member of the first lane quad.
    prepare = method('_sf6_prepare_stage', dict(Int32=int, _ld_shared_i32_volatile=load))
    workers = [env['_sf6_load_fragment'](owner, dest, prepare(owner, packed_base, tidbase+lane), 'test', 0)
               for lane, dest in enumerate(destinations)]
    replies = [None]*4
    while True:
        requests, ended = [], 0
        for worker, reply in zip(workers, replies):
            try:
                requests.append(worker.send(reply))
            except StopIteration:
                ended += 1
        if ended:
            assert ended == 4, 'quad diverged before a shuffle'
            break
        assert all((tidbase & 28) <= lane <= (tidbase & 28)+3 for _, lane in requests)
        replies = [requests[lane-(tidbase & 28)][0] for _, lane in requests]
    words = [[value & 0xFFFFFFFF for value in dest] for dest in destinations]
    assert words.count(words[0]) == 4
    return words[0], reads



class RegisterScalesTests(unittest.TestCase):
    def test_every_base_code_and_aligned_word_matches_independent_encoder(self):
        # Includes the final word in both bit planes and bases crossing 127/255.
        offsets = list(range(0, 2048, 4))
        for base in range(256):
            packed, expected = packed_codes(base, 2048, base*17)
            memory = bytearray([0xA5])*4096
            memory[256:256+1552] = packed
            before = bytes(memory)
            actual, reads = load_words(memory, 256, offsets)
            self.assertEqual(actual, [int.from_bytes(expected[i:i+4], 'little') for i in offsets])
            self.assertEqual(bytes(memory), before, 'register load wrote shared storage')
            self.assertEqual(len(reads), 4+2*len(offsets))

    def test_every_warp_and_quad_maps_to_the_original_scale_rows(self):
        packed, expected = packed_codes(251, 2048, 193)
        memory = bytearray([0xA5])*4096
        memory[256:256+1552] = packed
        for tidbase in range(0, 128, 4):
            row = (0, 256, 128, 384)[tidbase//32] + (tidbase % 32)//4*16
            for block in range(4):
                offsets = [block*512+row+4*j for j in range(4)]
                actual, _ = load_words(memory, 256, offsets, tidbase)
                self.assertEqual(actual, [int.from_bytes(expected[i:i+4], 'little') for i in offsets])

    def test_changed_packed_rings_and_nonmonotonic_duplicated_lane_words(self):
        offsets = [*range(2032, 2048, 4), *range(0, 16, 4), *range(1024, 1040, 4), *range(0, 16, 4)]
        for stages in (1, 2, 3, 4):
            memory = bytearray([0xA5])*(256+stages*1552+256)
            for generation in range(stages*16):
                start = 256+(generation % stages)*1552
                packed, expected = packed_codes(generation*37 % 256, 2048, generation*13)
                memory[start:start+1552] = packed
                before = bytes(memory)
                actual, _ = load_words(memory, start, offsets)
                self.assertEqual(actual, [int.from_bytes(expected[i:i+4], 'little') for i in offsets])
                self.assertEqual(bytes(memory), before)

    def test_actual_fc1_operands_release_order_and_k_blocks(self):
        for stages in (2, 4, 6):
            for tid in range(128):
                self.assertEqual(execute(True, stages, tid, pairs=4, compact=True, direct_scales=True),
                                 execute(True, stages, tid, pairs=4, compact=True, direct_scales=False))

    def test_default_scope_explicit_control_and_cache_identity(self):
        ns = namespace()
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        for recipe in ('t', 't,r', 't,r,sf6'):
            for rows in (0, 1, 6, 7, 8, 9, 16, 32, 128):
                for stages in (1, 2, 3, 4):
                    base = dict(ns['_parse_glm53_static_v2'](recipe), fc1=stages)
                    chosen = normalize(base, rows)
                    control = normalize(dict(base, sf6_registers=False), rows)
                    enabled = recipe == 't,r,sf6' and 1 <= rows <= 8 and stages % 2 == 0
                    self.assertEqual(chosen['sf6_registers'], enabled)
                    self.assertFalse(control['sf6_registers'])
                    self.assertEqual(normalize(chosen, rows), chosen)
                    self.assertEqual(normalize(control, rows), control)
                    self.assertEqual(key(chosen, m=rows) != key(control, m=rows), enabled)
                    g = geometry(reform=chosen['decode_reform'],
                        packed=chosen.get('reform_sf_pack', False), stages=stages, registers=True)
                    self.assertEqual(g.sf6_registers, enabled)
        for field in ('compact_staging', 'sf6_separate', 'fc1_reuse_a'):
            chosen = normalize(dict(ns['_parse_glm53_static_v2']('t,r,sf6'), **{field: False}), 8)
            self.assertFalse(chosen['sf6_registers'])


if __name__ == '__main__':
    unittest.main()
