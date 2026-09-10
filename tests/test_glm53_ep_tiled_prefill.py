"""CPU execution of the EP tiled entry and unchanged small-batch producer.

These checks cover descriptor coordinates, raw-scale identity, output clearing,
and task publication. They do not emulate TMA/CuTe lowering or GPU ordering.
"""
import ast
import copy
import gzip
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import test_moe_dynamic_sf6 as sf6_oracle
from test_glm53_ep_tiled_static import shard_helpers
from test_glm53_ep_route_scale_cache import single_warp_projection


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'overlay/modules/glm53_moe/moe_dynamic_ep_local.py'
ORACLE = ROOT / 'measurements/glm53_ep_local_20260908/onepass20-completed/source/moe_dynamic_ep_local.py.gz'


def method(name, text=None, class_name='MoEGatedEPLocalKernel'):
    tree = ast.parse(SOURCE.read_text() if text is None else text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def compile_method(node, namespace):
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return namespace[node.name]


def size(value):
    return math.prod(size(v) for v in value) if isinstance(value, tuple) else value


class Tensor:
    def __init__(self, shape, *, dtype='fp4', strides=None, backing=None):
        self.shape, self.element_type = shape, dtype
        self.strides = strides
        self.backing = object() if backing is None else backing

    def offset(self, coordinates):
        def dot(c, s):
            return sum(dot(a, b) for a, b in zip(c, s)) if isinstance(s, tuple) else c * s
        return dot(coordinates, self.strides)


class Entry:
    def __init__(self, *, sf6=False):
        self.groups, self.scales, self.descriptors = [], [], []
        self.launch = None
        self.sf6 = sf6
        self.entry = method('__call__', class_name='MoEGatedEPLocalKernelSF6' if sf6 else 'MoEGatedEPLocalKernel')
        self.fn = compile_method(self.entry, dict(
            cutlass=SimpleNamespace(const_expr=bool, Float32='f32', Int16='i16'),
            cute=SimpleNamespace(group_modes=self.group, size=size, make_tensor=self.scale_tensor),
            blockscaled_utils=SimpleNamespace(tile_atom_to_shape_SF=lambda shape, sf: (shape, sf)),
            utils=SimpleNamespace(LayoutEnum=SimpleNamespace(ROW_MAJOR='row', from_tensor=lambda t: 'layout')),
            Int32=int, DynamicLaunchParams=lambda rows, gates: (rows, gates)))

    def group(self, tensor, begin, end):
        self.groups.append((tensor, begin, end))
        assert (begin, end) == (1, 3)
        return Tensor((tensor.shape[0], tensor.shape[1:3], tensor.shape[3]),
                      dtype=tensor.element_type,
                      strides=(tensor.strides[0], tensor.strides[1:3], tensor.strides[3]),
                      backing=tensor.backing)

    def scale_tensor(self, pointer, layout):
        tensor = SimpleNamespace(pointer=pointer, layout=layout)
        self.scales.append(tensor)
        return tensor

    def descriptor(self, tensor, layout, tile, cluster, **kwargs):
        self.descriptors.append((tensor, layout, tile, cluster, kwargs))
        return ('tma', len(self.descriptors)), tensor

    def capture_kernel(self, *args):
        self.kernel_args = args
        return SimpleNamespace(launch=lambda **kw: setattr(self, 'launch', kw))

    def call(self, *, tiled, tokens=33, overrides=None):
        attrs = dict(tile_shape_mnk=(128, 128, 128), fc1_tile_shape_mnk=(128, 64, 128),
                     fc1_sfb_tile_shape_nk=(128, 128), sf_vec_size=16,
                     cluster_shape_mn=(1, 1), threads_per_cta=288,
                     _setup_attributes=lambda **kw: None,
                     _dense_cls=SimpleNamespace(_make_tma_atoms_and_tensors=self.descriptor),
                     kernel=self.capture_kernel)
        for name in ('a_smem_layout_staged', 'sfa_smem_layout_staged',
                     'fc1_b_smem_layout_staged', 'fc1_sfb_smem_layout_staged',
                     'b_smem_layout_staged', 'sfb_smem_layout_staged',
                     'tiled_mma', 'fc1_tiled_mma', 'mma_atom', 'cta_layout_mnk',
                     'phase2_b_smem_layout_staged', 'phase2_sfb_smem_layout_staged',
                     'fc1_sfb_smem_layout_storage', 'epi_smem_layout_staged'):
            attrs[name] = name
        shard_check = compile_method(method('_check_ep_shard_weights'), shard_helpers({}))
        attrs['_check_ep_shard_weights'] = lambda *args, **kw: shard_check(
            SimpleNamespace(**attrs), *args, **kw)
        if self.sf6:
            checker = sf6_oracle.functions({'_check_sf6_shapes'})['_check_sf6_shapes']
            attrs['_check_sf6_shapes'] = lambda *args: checker(SimpleNamespace(**attrs), *args)
        signature = self.entry.args.args
        args = {arg.arg: object() for arg in signature if arg.arg != 'self'}
        args.update(a_input=Tensor((tokens, 4096), dtype='bf16'),
                    topk_ids=Tensor((tokens * 8,), dtype='i32'),
                    topk_weights=Tensor((tokens * 8,), dtype='f32'),
                    packed_a=Tensor((128, 4096, 1)), sfa_ptr=SimpleNamespace(dtype='fp8'),
                    scatter_output=Tensor((tokens, 4096), dtype='f32'),
                    row_counts=Tensor((72,), dtype='i32'), max_active_clusters=48)
        for key, rows, inner, tiles in (('b_w13', 4096, 512, 8), ('b_down', 4096, 128, 16)):
            if tiled:
                args[key] = Tensor((rows, inner, tiles, 72),
                    strides=(inner, 1, rows * inner, rows * inner * tiles))
            else:
                args[key] = Tensor((rows, inner * tiles, 72),
                    strides=(inner * tiles, 1, rows * inner * tiles))
        if self.sf6:
            args.update(sfb1_packed=Tensor((72, 512, 1552), dtype='uint8'),
                        sfb2_packed=Tensor((72, 256, 1552), dtype='uint8'))
        args.update(overrides or {})
        self.args = args
        self.fn(SimpleNamespace(**attrs), **args)
        return self


class TiledEntryTests(unittest.TestCase):
    def test_real_entry_groups_only_weights_preserving_raw_scales_stream_and_output(self):
        for tokens in (1, 6, 32, 33, 127, 128, 4095, 4096, 8192, 16384):
            with self.subTest(tokens=tokens):
                old, new = Entry().call(tiled=False, tokens=tokens), Entry().call(tiled=True, tokens=tokens)
                self.assertEqual(old.groups, [])
                self.assertEqual(len(new.groups), 2)
                self.assertEqual([s.layout for s in new.scales], [s.layout for s in old.scales])
                for index, pointer_name in ((1, 'sfb_w13_ptr'), (2, 'sfb_down_ptr')):
                    self.assertIs(new.scales[index].pointer, new.args[pointer_name])
                self.assertEqual([d[2:] for d in new.descriptors], [d[2:] for d in old.descriptors])
                self.assertEqual(new.launch['grid'], (1, 1, 48))
                self.assertEqual(new.launch['block'], [288, 1, 1])
                self.assertTrue(new.launch['cooperative'])
                self.assertIs(new.launch['stream'], new.args['stream'])
                self.assertIn(new.args['scatter_output'], new.kernel_args)
                kernel_call = next(n for n in ast.walk(method('__call__'))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'kernel')
                params = next(i for i, n in enumerate(kernel_call.args)
                              if isinstance(n, ast.Name) and n.id == 'launch_params')
                self.assertEqual(new.kernel_args[params][1], 16)

    def test_tma_coordinates_match_v5_permutation_across_inner_tiles_and_experts(self):
        entry = Entry().call(tiled=True)
        for descriptor_index, key, inner, k in ((2, 'b_w13', 512, 4096), (4, 'b_down', 128, 2048)):
            grouped = entry.descriptors[descriptor_index][0]
            original = entry.args[key]
            self.assertIs(grouped.backing, original.backing)
            # Independent inverse of [E,K_tiles,rows,K_inner] physical storage.
            for expert in (0, 1, 71):
                for row in (0, 63, 64, 127, 2047, 2048, 4095):
                    for column in (0, 1, inner - 1, inner, k - 1):
                        offset = grouped.offset((row, (column % inner, column // inner), expert))
                        e, rest = divmod(offset, 4096 * k)
                        kt, rest = divmod(rest, 4096 * inner)
                        r, ki = divmod(rest, inner)
                        self.assertEqual((e, r, kt * inner + ki), (expert, row, column))
            # The native K128 TMA boxes never straddle a hierarchical K chunk.
            for k_start in range(0, k, 128):
                self.assertEqual(k_start // inner, (k_start + 127) // inner)

    def test_mixed_or_wrong_tiled_geometry_and_bf16_output_fail_before_tma(self):
        bad = [dict(b_w13=Tensor((4096, 4096, 72))),
               dict(b_down=Tensor((4096, 2048, 72))),
               dict(b_w13=Tensor((4096, 256, 16, 72))),
               dict(b_down=Tensor((4096, 128, 4, 72))),
               dict(b_w13=Tensor((4096, 512, 8, 288))),
               dict(scatter_output=Tensor((33, 4096), dtype='bf16'))]
        for override in bad:
            with self.subTest(override=override):
                entry = Entry()
                with self.assertRaises(ValueError):
                    entry.call(tiled=True, overrides=override)
                self.assertEqual(entry.descriptors, [])
                self.assertIsNone(entry.launch)

    def test_q0_task_and_fp32_scatter_bodies_remain_the_measured_ep_implementation(self):
        archived = gzip.decompress(ORACLE.read_bytes()).decode()
        for name in ('_setup_attributes', 'initialize_route_q0_and_publish',
                     'publish_ep_local_uniform_tasks', 'scatter_sC_to_gmem'):
            actual = method(name)
            if name == 'initialize_route_q0_and_publish':
                actual = single_warp_projection(actual)
            self.assertEqual(ast.dump(actual, include_attributes=False),
                             ast.dump(method(name, archived), include_attributes=False), name)

    def test_actual_small_batch_cooperative_init_zeros_entire_fp32_output(self):
        node = copy.deepcopy(method('initialize_route_q0_and_publish'))
        # Execute the unchanged producer through its first CTA barrier: unlike
        # checking an expression, this covers vector bounds and scalar tail.
        boundary = next(i for i, n in enumerate(node.body)
                        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and ast.unparse(n.value.func) == 'cute.arch.sync_threads')
        node.body = node.body[:boundary]
        base = 1 << 24
        for tokens in (1, 6, 33, 127, 129):
            words = bytearray(tokens * 4096)
            scalar = []
            class Plane:
                def __setitem__(self, index, value):
                    scalar.append((index, value))
                    words[index[0] * 4096 + index[1]] += 1
            def store(address, *values):
                self.assertEqual(values, (0, 0, 0, 0))
                offset = (address - base) // 4
                self.assertEqual((address - base) % 16, 0)
                self.assertTrue(0 <= offset <= len(words) - 4)
                for i in range(offset, offset + 4):
                    words[i] += 1
            ns = dict(Int32=int, Int64=int, Uint32=int,
                cutlass=SimpleNamespace(Uint32='u32'),
                cute=SimpleNamespace(recast_tensor=lambda *a: Plane()),
                _TASK_SLICE_CHUNK=4, st_global_v4_u32=store)
            fn = compile_method(node, ns)
            class Histogram(list):
                shape = (72,)
            counts, cursors, prefixes = Histogram([-1] * 72), [-1] * 72, [-1] * 73
            heads = [[-1] for _ in range(3)]
            for tid in range(288):
                fn(SimpleNamespace(threads_per_cta=288), (tid, 0, 1, tid // 32, int(tid == 0)),
                   (Tensor((tokens, 4096)), Tensor((tokens * 8,)), object(), object()),
                   (object(), object(), SimpleNamespace(iterator=SimpleNamespace(toint=lambda: base)), object(), object()),
                   (cursors, prefixes, heads[0]), (heads[1], heads[2], object(), object()),
                   (object(), object()), (0, 1152, 2304, 4096, 8192),
                   SimpleNamespace(row_counts=counts, gate_tile_cnt=16))
            self.assertEqual(words, b'\x01' * len(words))
            self.assertEqual(scalar, [])
            self.assertEqual(heads, [[0], [0], [0]])
            self.assertEqual(counts, [0] * 72)
            self.assertEqual(cursors, [0] * 72)
            self.assertEqual(prefixes, [0] * 73)

    def test_actual_task_publication_retains_four_slices_for_small_partial_tiles(self):
        stores = {}
        fn = compile_method(method('publish_ep_local_uniform_tasks'), dict(
            Int32=int, Int64=int, Uint32=int,
            get_ptr_as_int64=lambda pointer, offset: pointer + offset * 4,
            st_global_v4_u32=lambda pointer, *values: stores.update(
                {pointer + i * 4: value for i, value in enumerate(values)})))
        def unexpected(*args):
            self.fail('aligned E72/I2048 descriptors took the scalar fallback')
        for rows in (1, 6, 32, 33, 127, 128):
            for expert in (0, 71):
                for tile in (0, 1, 8):
                    stores.clear()
                    fn(SimpleNamespace(publish_uniform_deferred_tasks=unexpected),
                       0x1000, 0x2000, 16, 4, expert, tile, rows)
                    self.assertEqual(len(stores), 8)
                    for group in range(4):
                        slot = tile * 4 + group
                        word = stores[0x1000 + slot * 4]
                        valid = stores[0x2000 + slot * 4]
                        self.assertEqual((word & 0xffff, word >> 16), (expert, tile))
                        self.assertEqual((valid & 255, (valid >> 8) & 0xfff, valid >> 20),
                                         (rows, group * 4, 4))


class SF6EntryTests(unittest.TestCase):
    def test_packed_entry_has_no_raw_scale_descriptors_and_preserves_ep_launch(self):
        entry_node = method('__call__', class_name='MoEGatedEPLocalKernelSF6')
        reads = {n.id for n in ast.walk(entry_node) if isinstance(n, ast.Name)}
        self.assertFalse(reads & {'sfb_w13_ptr', 'sfb_down_ptr'})
        for tokens in (1, 6, 32, 33, 127, 4096, 8192, 16384):
            with self.subTest(tokens=tokens):
                old = Entry().call(tiled=True, tokens=tokens)
                new = Entry(sf6=True).call(tiled=True, tokens=tokens)
                self.assertEqual(len(new.scales), 1)  # Only activation SFA.
                self.assertEqual(len(new.descriptors), 4)
                for actual, original in zip(new.descriptors, [old.descriptors[i] for i in (0, 1, 2, 4)]):
                    self.assertEqual(actual[0].shape if isinstance(actual[0], Tensor) else actual[0].layout,
                                     original[0].shape if isinstance(original[0], Tensor) else original[0].layout)
                    self.assertEqual(actual[1:], original[1:])
                self.assertIn(new.args['sfb1_packed'], new.kernel_args)
                self.assertIn(new.args['sfb2_packed'], new.kernel_args)
                self.assertIn(new.args['scatter_output'], new.kernel_args)
                self.assertIn((new.args['row_counts'], 16), new.kernel_args)
                self.assertEqual(new.launch['grid'], (1, 1, 48))
                self.assertEqual(new.launch['block'], [288, 1, 1])
                self.assertTrue(new.launch['cooperative'])
                self.assertIs(new.launch['stream'], new.args['stream'])

    def test_wrong_packed_type_shape_geometry_or_output_rejected_before_tma(self):
        for override in (
            {'sfb1_packed': Tensor((72, 128, 1552), dtype='uint8')},
            {'sfb2_packed': Tensor((72, 256, 2048), dtype='uint8')},
            {'sfb1_packed': Tensor((72, 512, 1552), dtype='fp8')},
            {'sfb2_packed': Tensor((288, 256, 1552), dtype='uint8')},
            {'b_w13': Tensor((4096, 4096, 72))},
            {'b_down': Tensor((4096, 128, 4, 72))},
            {'row_counts': Tensor((288,), dtype='i32')},
            {'scatter_output': Tensor((33, 4096), dtype='bf16')},
        ):
            with self.subTest(override=override):
                entry = Entry(sf6=True)
                with self.assertRaises(ValueError):
                    entry.call(tiled=True, overrides=override)
                self.assertEqual(entry.descriptors, [])
                self.assertIsNone(entry.launch)

    def test_actual_mro_selects_ep_producer_scatter_and_sf6_pipeline_without_raw_fallback(self):
        class Stock:
            def __init__(self, *args, **kwargs):
                self.tile_shape_mnk = (128, 128, 128)
                self.share_input_across_experts = False
            def _setup_attributes(self, hidden_size): self.hidden = hidden_size
        class Tiled(Stock): pass
        sf6_class = copy.deepcopy(sf6_oracle.CLASS)
        ep_classes = [copy.deepcopy(n) for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef)]
        ns = dict(MoEGatedDynamicKernel=Stock, MoEGatedDynamicKernelTiled=Tiled,
                  stock_contract_matches=lambda: True,
                  cute=SimpleNamespace(jit=lambda fn: fn, kernel=lambda fn: fn))
        module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], sf6_class, *ep_classes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
        cls, ep, sf6 = (ns[n] for n in ('MoEGatedEPLocalKernelSF6', 'MoEGatedEPLocalKernel', 'MoEGatedDynamicKernelSF6'))
        instance = cls(reform_sf_pack=True)
        self.assertEqual(instance.load_register_requirement, 64)
        instance._setup_attributes(4096)
        self.assertEqual(instance.hidden, 4096)
        for name in ('initialize_route_q0_and_publish', 'publish_ep_local_uniform_tasks', 'scatter_sC_to_gmem'):
            self.assertIs(getattr(cls, name), getattr(ep, name))
        for name in ('kernel', 'load_fc1_tma_slice', 'load_fc2_tma_tile', '_check_sf6_shapes'):
            self.assertIs(getattr(cls, name), getattr(sf6, name))
        for value in (False, None, 1, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError): cls(reform_sf_pack=value)

    def test_existing_sf6_loaders_cover_all_sixteen_ep_slices_and_last_expert(self):
        harness = sf6_oracle.ProducerTests()
        ns, kernel, events = harness.namespace()
        T, State, Pipe = sf6_oracle.Tensor, sf6_oracle.State, sf6_oracle.FakePipeline
        base = 1 << 35
        smem = tuple(T(name, address) for name, address in
                     (('a', 10000), ('sfa', 20000), ('gate_b', 30000), ('up_b', 40000),
                      ('gate_sf', 50000), ('up_sf', 60000), ('up_extra', 70000)))
        for expert in (0, 71):
            for intermediate in range(16):
                events.clear()
                state, up, pipe = State(), State(), Pipe(events)
                ns['load_fc1_tma_slice'](kernel, intermediate, 1, expert, 16, 32,
                    state, pipe, up, pipe, ('a', 'b', 'sfa'),
                    (T('a'), T('sfa'), T('b'), T('packed', base, (72, 512, 1552))), smem)
                expansions = [e for e in events if e[0] == 'expand']
                self.assertEqual(len(expansions), 128)
                for native_half in range(2):
                    for kt in range(32):
                        gate, up = expansions[(native_half * 32 + kt) * 2:][:2]
                        for event, row in ((gate, intermediate + 16), (up, intermediate)):
                            self.assertEqual(event[1], base + (expert * 512 + row * 16 + kt // 2) * 1552)
                            self.assertTrue(base <= event[1] <= base + (72 * 512 - 1) * 1552)
                            self.assertEqual(event[3:], (kt % 2, 0, False))
                harness.check_publication(events, scale_copies=2, dma_copies=4)
                events.clear()
                for output in range(32):
                    ns['load_fc2_tma_tile'](kernel, intermediate, output, expert, state, pipe,
                        ('b',), (T('b'), T('packed', base, (72, 256, 1552))),
                        (T('b_smem'), T('b_extra'), T('sf_smem', 80000)))
                    expansion = next(e for e in reversed(events) if e[0] == 'expand')
                    self.assertEqual(expansion[1], base + (expert * 256 + (output // 2) * 16 + intermediate) * 1552)
                    self.assertTrue(base <= expansion[1] <= base + (72 * 256 - 1) * 1552)
                    self.assertEqual(expansion[3:], (output % 2, 0, True))
                harness.check_publication(events, scale_copies=1, dma_copies=1)

    def test_sf6_cache_suffix_cannot_alias_raw_ep_or_tp_and_old_keys_are_unchanged(self):
        source = ROOT / 'overlay/modules/glm53_moe/moe_dispatch.py'
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == '_dynamic_kernel_cache_key')
        key = compile_method(node, {})
        args = dict(activation_precision='fp4', quant_mode='nvfp4', E=72, k=4096, n=2048,
                    num_topk=8, mac=48, mma_tiler_mn=(128,128), topk_ids_dtype='i32',
                    input_scales_are_reciprocal=False, fast_math=True,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                    swiglu_limit=10., share_input_across_experts=False, tiled=True)
        raw = key(**args, ep_local_prefill=True)
        packed = key(**args, ep_local_prefill=True, reform_sf_pack=True)
        self.assertEqual(raw[-1], 'glm53_ep_prefill_local_fp32_v2')
        self.assertEqual(packed, raw + ('glm53_ep_tiled_sf6_v1',))
        stock = key(**args)
        tp = key(**(args | dict(E=288, n=512)), reform_sf_pack=True)
        self.assertEqual(stock, raw[:-1])
        self.assertEqual(tp[-1], 'sf6_direct_prefill_v1')
        self.assertNotEqual(tp, packed)
        self.assertEqual(key(**(args | dict(E=288, n=512)), reform_sf_pack=True, tp_sf6_q0=True),
                         tp + ('glm53_tp_sf6_q0_v1',))

    def test_dispatcher_selects_packed_ep_class_and_passes_sf6_constructor_flag(self):
        source = ROOT / 'overlay/modules/glm53_moe/moe_dispatch.py'
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_dynamic_kernel')
        select = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'ep_local_cls is not None'
                      and any(isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ep_kwargs'
                          for t in child.targets) for child in n.body))
        calls = []
        def raw(**kwargs): calls.append(('raw', kwargs)); return object()
        def packed(**kwargs): calls.append(('sf6', kwargs)); return object()
        ns = dict(ep_local_cls=raw, sf_vec_size=16, mma_tiler_mn=(128,128),
                  ep_hybrid_q0_dual_warp=False,
                  input_scales_are_reciprocal=False, fast_math=True, activation='swigluoai_uninterleave',
                  swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10., __package__='_ep_test')
        for enabled in (False, True):
            with patch.dict('sys.modules', {'_ep_test.moe_dynamic_ep_local': SimpleNamespace(MoEGatedEPLocalKernelSF6=packed)}):
                actual = dict(ns, reform_sf_pack=enabled)
                exec(compile(ast.Module(body=[select], type_ignores=[]), str(source), 'exec'), actual)
            self.assertEqual(calls[-1][0], 'sf6' if enabled else 'raw')
            self.assertIs(calls[-1][1]['share_input_across_experts'], False)
            self.assertEqual(calls[-1][1].get('reform_sf_pack', False), enabled)


if __name__ == '__main__':
    unittest.main()
