"""Lifetime and exact-byte gates for the compact C1 MoE shared buffers."""
import ast
import copy
import random
from types import SimpleNamespace
import unittest

from tests.test_engine_moe_fc1_reuse import execute
from tests.test_engine_moe_scatter_config import namespace
from tests.test_engine_moe_sf6_staging import CLASS, SOURCE, expand, geometry, packed_codes


def fc2_scale_code(producer):
    kernel = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
    call = 'fc2_pipeline.producer_acquire' if producer else 'fc2_pipeline.consumer_wait'
    loop = next(n for n in ast.walk(kernel) if isinstance(n, ast.For)
                and ast.unparse(n.target) == 'output_tile_idx'
                and any(isinstance(c, ast.Call) and ast.unparse(c.func) == call for c in ast.walk(n)))
    block = next(n for n in loop.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'cutlass.const_expr(self.reform_sf_pack)')
    return compile(ast.Module(body=[copy.deepcopy(block)], type_ignores=[]), str(SOURCE), 'exec')


class CompactStagingTests(unittest.TestCase):
    def test_actual_consumer_operands_and_releases_match_uncompacted_reuse(self):
        for stages in (1, 2, 3, 4, 6):
            for tid in (0, 31, 32, 63, 64, 95, 96, 127):
                # The existing production-loop executor poisons released
                # slots and makes omitted up input unreadable.
                self.assertEqual(execute(True, stages, tid, compact=True),
                                 execute(True, stages, tid, compact=False), (stages, tid))

    def test_four_warp_lifetime_with_producer_running_ahead(self):
        # An independent pipeline model deliberately permits more warp skew
        # than the named SF expansion barrier. Every warp releases each
        # original B/SFB slot only after its four K64 inputs are in registers.
        # The producer can refill as soon as all four have released, while
        # up still reads the retained registers. No additional A ring exists.
        for stages in (1, 2, 3, 4, 6):
            for compact in (False, True):
                g = geometry(stages=stages, compact=compact)
                for seed in range(20):
                    rng = random.Random(seed)
                    epochs, readers = [None]*stages, [set() for _ in range(stages)]
                    inputs, regs = {}, [{} for _ in range(4)]
                    cursor, block = [0]*4, [0]*4
                    produced, total = 0, 96  # three H4096 items, no ring reset
                    while min(cursor) < total:
                        ready = [w for w in range(4) if cursor[w] < total
                                 and epochs[cursor[w] % stages] == cursor[w]]
                        if produced < total and epochs[produced % stages] is None:
                            ready.append(4)
                        self.assertTrue(ready, 'pipeline deadlock')
                        actor = rng.choice(ready)
                        if actor == 4:
                            slot = produced % stages
                            epochs[slot], readers[slot] = produced, set(range(4))
                            if produced % 2 == 0:
                                a_slot = g._fc1_input_slot(slot)
                                self.assertLess(a_slot, g.fc1_input_stages)
                                # No other in-flight gate may own this input.
                                for i, epoch in enumerate(epochs):
                                    if i != slot and epoch is not None and epoch % 2 == 0:
                                        self.assertNotEqual(a_slot, g._fc1_input_slot(i))
                                inputs[a_slot] = [(produced//2, k) for k in range(4)]
                            produced += 1
                        else:
                            stage, k = cursor[actor], block[actor]
                            if stage % 2 == 0:
                                regs[actor][k] = inputs[g._fc1_input_slot(stage % stages)][k]
                            self.assertEqual(regs[actor][k], (stage//2, k))
                            block[actor] += 1
                            if block[actor] == 4:
                                slot = stage % stages
                                readers[slot].remove(actor)
                                if not readers[slot]:
                                    epochs[slot] = None
                                cursor[actor] += 1
                                block[actor] = 0
                    self.assertEqual(produced, total)

    def test_actual_fc2_transfer_and_expansion_on_prefilled_rings(self):
        producer, consumer = fc2_scale_code(True), fc2_scale_code(False)
        items = [(expert, part, out) for expert, part in ((2, 3), (0, 1), (1, 0))
                 for out in range(16)]
        # Independent packed source, including distinct expert/slice bases.
        source, expected = bytearray(), []
        for i in range(3*16*4):
            packed, raw = packed_codes((i*37) % 256, 2048, i*13)
            source.extend(packed)
            expected.append(raw)
        for stages in (1, 2, 3):
            for compact in (False, True):
                g = geometry(stages=2, fc2_stages=stages, compact=compact)
                mem = bytearray([0xA5])*16384
                env = dict(self=g, Int32=int, Int64=int, is_dma_lane0=True,
                    cutlass=SimpleNamespace(const_expr=bool), shared_ptr_to_u32=lambda x: x,
                    sf2_input_base_addr=64, sfb2_base_addr=8192, sfb2_packed_base=0,
                    sf2_blocks_per_expert=64, n_slices=4, bar2=0)
                barriers, transfers = [], []
                def transfer(dest, src, size, bar):
                    self.assertEqual(size, 1552)
                    self.assertEqual(dest % 16, 0)
                    self.assertEqual(src % 1552, 0)
                    transfers.append((dest, src, size))
                    mem[dest:dest+size] = source[src:src+size]
                def restore(dest, tid, size, *, packed_addr, word_expand):
                    self.assertEqual(size, 2048)
                    before = bytes(mem)
                    if compact:
                        self.assertLessEqual(packed_addr+1552, 64+g.sf2_packed_bytes)
                        self.assertLessEqual(packed_addr+1552, dest)
                    else:
                        self.assertIsNone(packed_addr)
                    barriers.append(expand(mem, dest, size, source=packed_addr,
                                           seed=len(barriers), word_override=word_expand))
                    # Includes neighboring queued packed inputs, expanded
                    # slots and header canaries: only this raw slot changes.
                    self.assertEqual(mem[:dest], before[:dest])
                    self.assertEqual(mem[dest+size:], before[dest+size:])
                env['_bulk_g2s'], g._sf_expand_stage = transfer, restore
                produced = 0
                for consumed, (expert, part, out) in enumerate(items):
                    while produced < min(len(items), consumed+stages):
                        e, p, o = items[produced]
                        exec(producer, dict(env, weight_expert_idx=e, intermediate_slice=p,
                            output_tile_idx=o, fc2_prod_state=SimpleNamespace(index=produced % stages)))
                        dest = (64+(produced % stages)*1552 if compact else
                                8192+(produced % stages)*2048)
                        self.assertEqual(transfers[-1], (dest, (e*64+o*4+p)*1552, 1552))
                        produced += 1
                    exec(consumer, dict(env, tidx=0,
                        fc2_cons_state=SimpleNamespace(index=consumed % stages)))
                    raw = 8192+(consumed % stages)*2048
                    self.assertEqual(mem[raw:raw+2048], expected[expert*64+out*4+part])
                self.assertEqual(barriers, [1 if compact else 2]*len(items))
                self.assertEqual(len(transfers), len(items))

    def test_scope_default_rollback_and_cache_identity(self):
        ns = namespace()
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        for recipe in ('t', 't,r', 't,r,sf6'):
            config = ns['_parse_glm53_static_v2'](recipe)
            for rows in (0, 1, 6, 7, 8, 9, 16, 32, 128):
                for stages in (1, 2, 3, 4):
                    for reuse, separate in ((True, True), (False, True), (True, False)):
                        raw = dict(config, fc1=stages, fc1_reuse_a=reuse, sf6_separate=separate)
                        chosen = normalize(raw, rows)
                        control = normalize(dict(raw, compact_staging=False), rows)
                        enabled = recipe == 't,r,sf6' and 1 <= rows <= 8 and stages % 2 == 0 and reuse and separate
                        self.assertEqual(chosen['compact_staging'], enabled)
                        self.assertFalse(control['compact_staging'])
                        self.assertEqual(normalize(chosen, rows), chosen)
                        self.assertEqual(normalize(control, rows), control)
                        self.assertEqual(key(chosen, m=rows) != key(control, m=rows), enabled)
                        g = geometry(reform=chosen['decode_reform'], packed=chosen.get('reform_sf_pack', False),
                                     separate=separate, stages=stages, reuse=reuse)
                        self.assertEqual(g.compact_staging, enabled)
                        self.assertEqual(g.fc1_input_stages, stages//2 if enabled else stages)
                        self.assertEqual(g.sf2_packed_bytes, stages*1552 if enabled else 0)


if __name__ == '__main__':
    unittest.main()
