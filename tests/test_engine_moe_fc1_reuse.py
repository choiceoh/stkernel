"""Execute the production FC1 gate/up loop with distinct stage/block payloads."""
import ast
from collections import Counter
import copy
import itertools
from types import SimpleNamespace
import unittest

from tests.test_engine_moe_sf6_staging import CLASS, SOURCE, geometry, method
from tests.test_engine_moe_scatter_config import namespace


def consumer_code():
    kernel = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
    loop = next(n for n in ast.walk(kernel) if isinstance(n, ast.For)
                and ast.unparse(n.target) == 'gu'
                and any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'cute.gemm'
                        for c in ast.walk(n)))
    return compile(ast.Module(body=[copy.deepcopy(loop)], type_ignores=[]), str(SOURCE), 'exec')


def execute(reuse, stages, tid, pairs=32):
    """32 K256 pairs span two H4096 items; registers persist across both.

    A/SFA match between gate/up, B/SFB deliberately differ. Released shared
    slots are poisoned immediately, exposing accidental reads after release.
    This tests dataflow and lifetime, not a CPU implementation of GPU MMA.
    """
    state = SimpleNamespace(cursor=0, index=0)
    shared, registers, loads, events, observed, scales = {}, {}, Counter(), [], [], {}

    def payload(kind, stage, block):
        # Input fragments belong to a K tile; weight fragments belong to a
        # gate/up stage. Every thread, item and K block gets different bytes.
        epoch = stage // 2 if kind in ('A', 'SFA') else stage
        return (kind, epoch, tid, block, (epoch*101+tid*17+block*29) & 0xFFFFFFFF)

    class View:
        def __init__(self, kind, stage=None, block=None):
            self.kind, self.stage, self.block = kind, stage, block
        def __getitem__(self, key):
            return View(self.kind, key[3] if len(key) == 4 else self.stage,
                        key[2] if len(key) == 3 else self.block)
        @property
        def iterator(self):
            return self
        def read(self):
            if self.kind.startswith('r'):
                return registers[self.kind, self.block]
            value = shared[self.kind, self.stage, self.block]
            assert value is not None, 'read from a released pipeline slot'
            loads[self.kind] += 1
            return value

    def wait(s, peek):
        events.append(('wait', s.cursor))
        for kind in ('A', 'B', 'SFA', 'SFB'):
            for block in range(4):
                shared[kind, s.index, block] = (None if reuse and s.cursor % 2
                    and kind in ('A', 'SFA') else payload(kind, s.cursor, block))
    def release(s):
        events.append(('release', s.cursor))
        for key in list(shared):
            if key[1] == s.index:
                shared[key] = None
    def advance():
        state.cursor += 1
        state.index = state.cursor % stages
    state.advance = advance

    def copy_fragment(atom, source, dest):
        assert dest.kind == 'r'+source.kind
        registers[dest.kind, dest.block] = source.read()
    def mma(atom, output, a, b, prior):
        values = (a.read(), b.read(), scales['SFA'].read(), scales['SFB'].read())
        expected = tuple(payload(kind, state.cursor, a.block) for kind in ('A', 'B', 'SFA', 'SFB'))
        assert values == expected, (state.cursor, values, expected)
        observed.append((state.cursor, output.kind, output.block, values))

    owner = geometry()
    owner.fc1_reuse_a, owner.num_m_tiles, owner.num_n_tiles1 = reuse, 1, 4
    owner._sf_expand_stage = lambda *a, **kw: None  # unchanged, separately byte-tested
    env = dict(self=owner, Int32=int, tidx=tid, num_k_blocks1=4,
        cutlass=SimpleNamespace(const_expr=bool, range_constexpr=range),
        cute=SimpleNamespace(copy=copy_fragment, filter_zeros=lambda x: x, gemm=mma),
        fc1_pipeline=SimpleNamespace(consumer_try_wait=lambda s: True,
            consumer_wait=wait, consumer_release=release), fc1_cons_state=state,
        sfb1_base_addr=4096, sf1_input_base_addr=240,
        mma_atom=SimpleNamespace(set=lambda field, value: scales.__setitem__(field, value)),
        WarpField=SimpleNamespace(SFA='SFA', SFB='SFB'),
        gate_acc=View('gate'), up_acc=View('up'))
    for name, kind in (('csA1','A'), ('csB1','B'), ('csSFA1_tile','SFA'), ('csSFB1','SFB'),
                       ('crA1','rA'), ('crB1','rB'), ('fz_crSFA1_tile','rSFA'), ('fz_crSFB1','rSFB'),
                       ('tCrA1','rA'), ('tCrB1','rB'), ('tCrSFA1_tile','rSFA'), ('tCrSFB1','rSFB')):
        env[name] = View(kind)
    for name in ('smem_copy_A1','smem_copy_B1','smem_copy_SFA1','smem_copy_SFB1'):
        env[name] = name
    code = consumer_code()
    for _ in range(pairs):
        exec(code, env)
    assert state.cursor == 2*pairs
    assert len(observed) == pairs*2*4*4
    return observed, loads, events


class Fc1ReuseTests(unittest.TestCase):
    def test_actual_producer_transaction_bytes_and_single_lane_expect(self):
        kernel = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
        start = next(i for i, n in enumerate(kernel.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'fc1_tma_bytes')
        end = next(i for i, n in enumerate(kernel.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == 'b2_smem_one')
        budget = compile(ast.Module(body=copy.deepcopy(kernel.body[start:end]), type_ignores=[]), str(SOURCE), 'exec')
        loop = next(n for n in ast.walk(kernel) if isinstance(n, ast.For)
                    and ast.unparse(n.target) == 'gu'
                    and any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'fc1_pipeline.producer_acquire'
                            for c in ast.walk(n)))
        producer = compile(ast.Module(body=[copy.deepcopy(loop)], type_ignores=[]), str(SOURCE), 'exec')
        class Tensor:
            def __getitem__(self, key):
                return self
        for reuse in (False, True):
            for skip_a in (False, True):
                totals, expects = Counter(), []
                for lane in range(32):
                    owner = geometry()
                    owner.fc1_reuse_a, owner.skip_a = reuse, skip_a
                    owner.a_dtype = owner.b_dtype = owner.sf_dtype = None
                    state = SimpleNamespace(index=0, cursor=0)
                    pending, issued = [0], [False]
                    def advance():
                        state.cursor += 1
                        state.index = state.cursor % 2
                    state.advance = advance
                    def acquire(s):
                        if lane == 0:
                            pending[0], issued[0] = env['fc1_tma_bytes'], False
                            self.assertGreater(pending[0], 0)
                    def expect(bar, size):
                        self.assertEqual(lane, 0, 'expect_tx ran on more than the DMA leader')
                        self.assertFalse(issued[0], 'expected bytes added after a transfer')
                        pending[0] += size
                        expects.append((state.cursor, size))
                    def transfer(kind, size):
                        if lane == 0:
                            issued[0] = True
                            totals[kind] += size
                            pending[0] -= size
                            self.assertGreaterEqual(pending[0], 0, 'transaction completed before all bytes were issued')
                    def commit(s):
                        if lane == 0:
                            self.assertEqual(pending[0], 0, 'consumer would wait for unissued bytes')
                    sizes = {'A': 2048, 'SFA': 2048, 'B': 16384}
                    env = dict(self=owner, Int32=int, Int64=int, is_dma_lane0=lane == 0,
                        cutlass=SimpleNamespace(const_expr=bool, range_constexpr=range),
                        cute=SimpleNamespace(size_in_bytes=lambda dtype, layout: layout,
                            copy=lambda atom, src, dst, **kw: transfer(atom, sizes[atom]),
                            arch=SimpleNamespace(mbarrier_expect_tx=expect)),
                        b1_smem_one=16384, a1_smem_one=2048, sfa1_smem_one=2048,
                        fc1_prod_state=state, fc1_pipeline=SimpleNamespace(producer_acquire=acquire,
                            producer_get_barrier=lambda s: s.index, producer_commit=commit),
                        tma_a='A', tma_sfa='SFA', tma_b_w13='B',
                        sf1_input_base_addr=240, sfb1_base_addr=4096, sfb1_packed_base=0,
                        sfb_gate_idx=4, sfb_up_idx=0, weight_expert_idx=3, sf_blocks_per_expert=128,
                        k_tile_cnt1=16, shared_ptr_to_u32=lambda x: x,
                        _bulk_g2s=lambda dest, src, size, bar: transfer('SFB', size))
                    for name in ('tAgA_mk', 'tAsA', 'tBgB_gate_nk', 'tBgB_up_nk', 'tBsB1', 'tAgSFA_mk', 'tAsSFA'):
                        env[name] = Tensor()
                    exec(budget, env)
                    for tile in range(16):
                        exec(producer, dict(env, k_tile=tile))
                expected_input = 0 if skip_a else 2048*16*(1 if reuse else 2)
                self.assertEqual(totals, Counter(A=expected_input, SFA=expected_input,
                                                B=16384*32, SFB=1552*32))
                self.assertEqual(expects, [(2*k, 4096) for k in range(16)] if reuse and not skip_a else [])

    def test_fragment_guard_rejects_partial_or_aliased_k_blocks(self):
        def size(value, mode=None):
            shape = value.shape
            return shape[mode[0]] if mode else shape[0]*shape[1]*shape[2]
        validate = method('_validate_fc1_reuse_fragment', dict(cute=SimpleNamespace(
            rank=lambda x: len(x.shape), size=size,
            make_identity_tensor=lambda shape: list(itertools.product(*(range(n) for n in shape))),
            crd2idx=lambda point, layout: layout(point))))
        # Broadcast within one K block is valid; overlap between K blocks is not.
        valid = SimpleNamespace(shape=(8, 2, 4), layout=lambda p: p[0]+p[2]*8)
        self.assertEqual(validate(geometry(), valid, 'A'), (8, 8, 8, 8))
        for stride in (0, 4):
            alias = SimpleNamespace(shape=(8, 2, 4), layout=lambda p: p[0]+p[2]*stride)
            with self.assertRaisesRegex(ValueError, 'aliased K blocks'):
                validate(geometry(), alias, 'SFA')
        with self.assertRaisesRegex(ValueError, 'full K tile'):
            validate(geometry(), SimpleNamespace(shape=(8, 2, 3)), 'A')

    def test_actual_gate_up_fragments_survive_release_and_changed_items(self):
        for stages in (1, 2, 3):
            for tid in (0, 1, 31, 32, 63, 64, 95, 96, 127):
                control, old_loads, old_events = execute(False, stages, tid)
                candidate, new_loads, new_events = execute(True, stages, tid)
                self.assertEqual(candidate, control, (stages, tid))
                self.assertEqual(new_events, old_events)
                self.assertEqual(new_loads['A'], old_loads['A']//2)
                self.assertEqual(new_loads['SFA'], old_loads['SFA']//2)
                self.assertEqual(new_loads['B'], old_loads['B'])
                self.assertEqual(new_loads['SFB'], old_loads['SFB'])

    def test_default_control_cache_and_geometry_scope(self):
        ns = namespace()
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        for recipe in ('t', 't,r', 't,r,sf6'):
            config = ns['_parse_glm53_static_v2'](recipe)
            for rows in (0, 1, 6, 7, 8, 9, 14, 16, 21, 24, 28, 32, 128):
                a = normalize(config, rows)
                b = normalize(dict(config, fc1_reuse_a=False), rows)
                enabled = recipe == 't,r,sf6' and 1 <= rows <= 8
                self.assertEqual(a['fc1_reuse_a'], enabled)
                self.assertFalse(b['fc1_reuse_a'])
                self.assertEqual(normalize(a, rows), a)
                self.assertEqual(normalize(b, rows), b)
                self.assertEqual(key(a, m=rows) != key(b, m=rows), enabled)
        self.assertTrue(geometry().fc1_reuse_a)
        self.assertFalse(geometry(reform=False).fc1_reuse_a)
        self.assertFalse(geometry(packed=False).fc1_reuse_a)


if __name__ == '__main__':
    unittest.main()
