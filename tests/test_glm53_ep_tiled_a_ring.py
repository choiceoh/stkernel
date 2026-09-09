"""Actual-source SF6 A-ring admission, transactions and finite pipeline model.

The model checks producer/consumer ownership under interleaving; it cannot
establish CuTe lowering, hardware synchronization, numerics or throughput.
"""
import ast
import copy
import random
import types
import unittest
from unittest.mock import patch

from test_glm53_ep_tiled_static import SOURCE, STOCK, constants, extract, fake_torch, function


def compile_function(body, ns, name='run'):
    fn = ast.FunctionDef(name=name,
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(SOURCE), 'exec'), ns)
    return ns[name]


def kernel_loop(target):
    return next(n for n in ast.walk(function('kernel'))
                if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and n.target.id == target)


class State:
    def __init__(self): self.count = 0
    @property
    def index(self): return self.count % 2
    def advance(self): self.count += 1


class Pipeline:
    def __init__(self, name, events): self.name, self.events = name, events
    def emit(self, action, state):
        self.events.append((action, self.name, state.count, state.index))
    def producer_acquire(self, state): self.emit('acquire', state)
    def producer_get_barrier(self, state): return self.name, state.count, state.index
    def producer_commit(self, state): self.emit('commit', state)
    def consumer_try_wait(self, state): return None
    def consumer_wait(self, state, _): self.emit('wait', state)
    def consumer_release(self, state): self.emit('release', state)


class Tensor:
    def __init__(self, name): self.name = name
    def __getitem__(self, index): return self.name, index


def traces(*, wrong_a_slot=False, tasks=2):
    """Run actual FC1 loops, replacing only MMA/copies-to-register with a read event."""
    producer, consumer = kernel_loop('k_tile'), kernel_loop('_k_tile')
    for loop in (producer, consumer): loop.iter.keywords = []  # Python range lacks unroll.
    gu_loop = next(n for n in consumer.body if isinstance(n, ast.For))
    slot_if = next(n for n in gu_loop.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'cutlass.const_expr(self.a_ring)')
    last_slot = gu_loop.body.index(slot_if)
    # Keep both waits, actual SF6 expansion, the actual a_slot selection, and
    # both original release/advance sites. Kernel arithmetic is separately
    # checked against the pinned stock body by test_glm53_ep_tiled_static.
    gu_loop.body = gu_loop.body[:last_slot + 1] + ast.parse(
        'consume(_k_tile, gu, a_slot, fc1_cons_state.index)').body + gu_loop.body[-2:]
    if wrong_a_slot:
        slot_if.body[0].value = ast.parse('fc1_cons_state.index', mode='eval').body
    all_events = []
    for body in (producer, consumer):
        events = []
        ap, bp, ac, bc = State(), State(), State(), State()
        def copy_tma(atom, src, dst, *, tma_bar_ptr):
            events.append(('write', *tma_bar_ptr, atom, src, dst))
        def bulk(dst, src, size, barrier):
            assert size == 1552
            events.append(('write', *barrier, 'SF6', src, dst))
        def expand(addr, tidx, size):
            assert size == 2048
            events.append(('expand', 'B', bc.count, addr // 2048))
        def consume(k, gu, a_slot, b_slot):
            events.append(('read', ac.count, bc.count, k, gu, a_slot, b_slot))
        ns = dict(Int32=int, Int64=int, range=range,
            cutlass=types.SimpleNamespace(const_expr=bool, range_constexpr=range),
            cute=types.SimpleNamespace(copy=copy_tma),
            self=types.SimpleNamespace(a_ring=True, skip_a=False, skip_sf=False,
                sf_pack=False, reform_sf_pack=True, sf1_packed_blocks=1,
                sf1_block_bytes=2048, _sf_expand_stage=expand),
            a_pipeline=Pipeline('A', events), fc1_pipeline=Pipeline('B', events),
            a_prod_state=ap, fc1_prod_state=bp, a_cons_state=ac, fc1_cons_state=bc,
            k_tile_cnt1=16, tidx=0, is_dma_lane0=True, weight_expert_idx=71,
            sf_blocks_per_expert=512, sfb_gate_idx=19, sfb_up_idx=3,
            sfb1_base_addr=0, sfb1_packed_base=0, _bulk_g2s=bulk,
            shared_ptr_to_u32=lambda value: value, consume=consume)
        for name in ('tAgA_mk', 'tAgSFA_mk', 'tAsA', 'tAsSFA',
                     'tBgB_gate_nk', 'tBgB_up_nk', 'tBsB1'):
            ns[name] = Tensor(name)
        ns.update(tma_a='A', tma_sfa='SFA', tma_b_w13='B')
        run = compile_function([body], ns)
        for _ in range(tasks): run()
        all_events.append(events)
    return all_events


def schedule(traces, seed):
    """Interleave independent agents; only stage acquire/wait may block."""
    rng = random.Random(seed)
    slots = {(name, i): {'status': 'empty'} for name in ('A', 'B') for i in range(2)}
    positions = [0, 0]
    reads = []
    while any(positions[i] < len(traces[i]) for i in (0, 1)):
        ready = []
        for actor in (0, 1):
            if positions[actor] == len(traces[actor]): continue
            op = traces[actor][positions[actor]]
            if op[0] == 'acquire' and slots[op[1], op[3]]['status'] != 'empty': continue
            if op[0] == 'wait':
                cell = slots[op[1], op[3]]
                if cell['status'] != 'full' or cell['seq'] != op[2]: continue
            ready.append(actor)
        assert ready, 'producer/consumer deadlock'
        actor = rng.choice(ready)
        op = traces[actor][positions[actor]]; positions[actor] += 1
        if op[0] == 'read':
            _, a_seq, b_seq, k, gu, a_slot, b_slot = op
            a, b = slots['A', a_slot], slots['B', b_slot]
            assert a['status'] == b['status'] == 'reading'
            assert a['seq'] == a_seq and b['seq'] == b_seq == 2 * a_seq + gu
            assert a['writes']['A'][0] == ('tAgA_mk', (None, k))
            assert a['writes']['SFA'][0] == ('tAgSFA_mk', (None, k))
            assert b['writes']['B'][0] == ('tBgB_gate_nk' if gu == 0 else 'tBgB_up_nk', (None, k))
            assert 'SF6' in b['writes'] and b['expanded']
            a['reads'].append(gu); b['reads'].append(gu); reads.append((a_seq, gu, k))
            continue
        action, name, seq, slot = op[:4]
        cell = slots[name, slot]
        if action == 'acquire':
            assert cell['status'] == 'empty'
            cell.update(status='writing', seq=seq, writes={}, reads=[], expanded=False)
        else:
            assert cell['seq'] == seq
            if action == 'write':
                assert cell['status'] == 'writing'
                cell['writes'][op[4]] = op[5:]
                if op[4] != 'SF6': assert op[6][1][-1] == slot
                else: assert op[6] == slot * 2048
            elif action == 'commit':
                assert cell['status'] == 'writing'
                assert set(cell['writes']) == ({'A', 'SFA'} if name == 'A' else {'B', 'SF6'})
                cell['status'] = 'full'
            elif action == 'wait':
                assert cell['status'] == 'full'; cell['status'] = 'reading'
            elif action == 'expand':
                assert cell['status'] == 'reading'; cell['expanded'] = True
            elif action == 'release':
                assert cell['status'] == 'reading'
                assert cell['reads'] == ([0, 1] if name == 'A' else [seq % 2])
                cell['status'] = 'empty'
            else: raise AssertionError(action)
    assert all(cell['status'] == 'empty' for cell in slots.values())
    return reads


class EPTiledARingTests(unittest.TestCase):
    def test_actual_constructor_preserves_sf6_attributes_and_other_modes(self):
        base, ep = function('__init__', STOCK), function('__init__')
        ns = constants()
        ns.update(DenseGemmKernel=object, cutlass=types.SimpleNamespace(Float32='f32'),
            utils=types.SimpleNamespace(get_smem_capacity_in_bytes=lambda _: 101376),
            pipeline=types.SimpleNamespace(NamedBarrier=lambda **kw: types.SimpleNamespace(**kw)),
            is_gated_activation=lambda _: True, ep_tiled_source_contract=lambda: None)
        extract('ep_tiled_geometry', ns); extract('ep_tiled_scale_mode', ns)
        classes = [ast.ClassDef(name='MoEStaticKernelV5', bases=[], keywords=[], body=[base], decorator_list=[]),
                   ast.ClassDef(name='Candidate', bases=[ast.Name('MoEStaticKernelV5', ast.Load())],
                                keywords=[], body=[ep], decorator_list=[])]
        module = ast.Module(body=[ast.ImportFrom('__future__', [ast.alias('annotations')], 0), *classes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
        for m in range(1, 33):
            for sf6 in (False, True):
                actual = ns['Candidate'](num_tokens=m, max_rows=256, max_active_clusters=48, reform_sf_pack=sf6)
                reference = ns['MoEStaticKernelV5'](16, 16, decode_reform=m<=8, reform_sf_pack=sf6,
                    fast_math=True, activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                self.assertEqual(actual.a_ring, sf6 and m<=8)
                self.assertEqual(actual.word_unpack, sf6 and m<=8)
                self.assertEqual(actual.scatter_bf16, sf6 and m<=8)
                self.assertEqual({k:v for k,v in vars(actual).items() if k not in ('a_ring','word_unpack','scatter_bf16','ep_num_tokens','ep_max_rows')},
                                 {k:v for k,v in vars(reference).items() if k != 'a_ring'})
        with self.assertRaises(ValueError): ns['MoEStaticKernelV5'](16, 16, reform_sf_pack=True, a_ring=True)

    def test_actual_transaction_counts_match_only_completed_copies(self):
        body = function('kernel').body
        start = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='a1_smem_one')
        stop = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='smem')
        ns = dict(cutlass=types.SimpleNamespace(const_expr=bool),
                  cute=types.SimpleNamespace(slice_=lambda layout, _:layout,
                                            size_in_bytes=lambda dtype, layout:layout),
                  a1_smem_staged=2048, sfa1_smem_staged=2048, b1_smem_staged=16384,
                  sfb1_smem_staged=2048, b2_smem_staged=16384, sfb2_smem_staged=2048)
        run = compile_function(body[start:stop] + ast.parse('return fc1_tma_bytes, a_tma_bytes, fc2_tma_bytes').body, ns)
        for enabled in (False, True):
            ns['self'] = types.SimpleNamespace(a_ring=enabled, skip_a=False, skip_sf=False,
                reform_sf_pack=True, sf_pack=False, sf1_packed_blocks=1, sf2_stage_bytes=1552,
                a_dtype='fp4', b_dtype='fp4', sf_dtype='fp8')
            self.assertEqual(run(), (17936 if enabled else 22032, 4096, 17936))
        self.assertEqual(16 * 4096, 65536)  # requested A/SFA savings/task, not measured traffic.

    def test_actual_two_stage_schedule_preserves_gate_up_reads_across_tasks(self):
        events = traces(tasks=3)
        for seed in range(32):
            self.assertEqual(schedule(events, seed), [(k, gu, k % 16) for k in range(48) for gu in (0,1)])
        producer = events[0]
        self.assertEqual(sum(op[0]=='write' and op[4]=='A' for op in producer), 48)
        self.assertEqual(sum(op[0]=='write' and op[4]=='SFA' for op in producer), 48)
        self.assertEqual(sum(op[0]=='write' and op[4]=='B' for op in producer), 96)
        self.assertEqual(sum(op[0]=='write' and op[4]=='SF6' for op in producer), 96)

    def test_schedule_oracle_rejects_using_weight_stage_for_shared_a(self):
        with self.assertRaises(AssertionError): schedule(traces(wrong_a_slot=True), 0)

    def test_runtime_hot_cache_key_matches_ring_admission_without_recompile(self):
        ns = constants(); ns['_EP_TILED_KERNEL_CACHE'] = {}
        extract('ep_tiled_geometry', ns); extract('ep_tiled_scale_mode', ns)
        t = fake_torch()
        ns['ep_tiled_compile_spec'] = lambda **_: self.fail('hot specialization compiled again')
        core = types.ModuleType('flashinfer.jit.cute_dsl_core')
        core.build_and_load_cute_dsl_kernel = lambda *_a, **_kw: self.fail('unexpected build')
        package = types.ModuleType('ring_fixture'); package.moe_dispatch = object()
        ns['__package__'] = 'ring_fixture'
        get = extract('get_ep_tiled_decode_kernel', ns)
        with patch.dict('sys.modules', {'torch':t, 'flashinfer.jit.cute_dsl_core':core, 'ring_fixture':package}):
            for m in range(1, 33):
                for sf6 in (False, True):
                    geom = ns['ep_tiled_geometry'](m,256,48)
                    key = (ns['EP_TILED_CACHE_TAG'],m,256,48,'int32',False,True,
                           geom['fc1'],geom['fc2'],'nvfp4','sf6_v1' if sf6 else 'raw_mma_scales',
                           'swigluoai_uninterleave',1.,0.,10.,
                           'bf16_scatter' if sf6 and m<=8 else 'fp32_scatter')
                    if sf6 and m<=8:
                        key += (ns['EP_TILED_A_RING_CACHE_TAG'], ns['EP_TILED_SF6_WORD_CACHE_TAG'],
                                ns['EP_TILED_BF16_SCATTER_CACHE_TAG'])
                    sentinel = object(); ns['_EP_TILED_KERNEL_CACHE'][key] = sentinel
                    self.assertEqual(get(num_tokens=m,reform_sf_pack=sf6), (sentinel,48))


if __name__ == '__main__': unittest.main()
