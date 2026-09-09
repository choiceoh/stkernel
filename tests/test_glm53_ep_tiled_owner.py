"""Execute the EP tile-major owner with allocation/launch-recording CPU fakes.

No Torch/CUDA import is needed. These are ownership/admission contracts, not
arithmetic, CuTe compilation, CUDA-graph replay or performance evidence.
"""
import ast
import copy
import gc
import importlib.util
import itertools
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'overlay/modules/glm53_moe/glm53_ep_tiled.py'
WRAPPER = ROOT / 'overlay/modules/glm53_moe/flashinfer_b12x_moe.py'
DISPATCH = ROOT / 'overlay/modules/glm53_moe/moe_dispatch.py'
ADDRESSES = itertools.count(1 << 32, 1 << 30)


class Tensor:
    def __init__(self, shape, dtype='f32', *, pointer=None, device='cuda:0', contiguous=True, base=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self._pointer = next(ADDRESSES) if pointer is None else pointer
        self._version = 0
        self._contiguous, self.base = contiguous, base
        self.ndim = len(shape)
        self.is_cuda = str(device).startswith('cuda:')
        self.record_stream = Mock()

    def data_ptr(self): return self._pointer
    def numel(self): return math.prod(self.shape)
    def element_size(self): return {'f32': 4, 'bf16': 2, 'u8': 1, 'i32': 4, 'i64': 8}[self.dtype]
    def is_contiguous(self): return self._contiguous
    def stride(self): return tuple(math.prod(self.shape[i + 1:]) for i in range(self.ndim))
    def __getitem__(self, index):
        if not isinstance(index, slice) or index.start not in (None, 0) or index.step is not None:
            raise AssertionError('unexpected tensor operation')
        return Tensor((min(index.stop, self.shape[0]), *self.shape[1:]), self.dtype,
                      pointer=self.data_ptr(), device=self.device, base=self)


def extract_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], node], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


class Harness:
    def __init__(self, test, capacity=8192):
        self.test, self.events = test, []
        package = '_glm53_ep_tiled_owner_test'
        self.package = package
        self.torch = ModuleType('torch')
        self.torch.Tensor = Tensor
        for key, value in dict(float32='f32', bfloat16='bf16', uint8='u8', int32='i32', int64='i64').items():
            setattr(self.torch, key, value)
        self.torch.cuda = SimpleNamespace(get_device_capability=Mock(return_value=(12, 1)),
            is_current_stream_capturing=Mock(return_value=False), current_stream=Mock(return_value='stream'))
        self.torch.empty = Mock(side_effect=lambda shape, *, dtype, device: Tensor(shape, dtype, device=device))
        self.md = ModuleType(package + '.moe_dispatch')
        self.md._FORCED_BACKEND = None
        self.md._GLM53_B12X_FORCE_BACKEND = None
        self.md._STATIC_V2_OVERRIDE = None
        self.md._FORCE_MOE_W4A16_ENV = 'FLASHINFER_B12X_FORCE_MOE_W4A16'
        self.md.invalidate_tile_major_if_reloaded = Mock()
        self.md.tile_expert_weights_inplace = Mock(side_effect=self.relayout)
        self.md._get_weight_views = Mock(side_effect=self.views)
        self.md.launch_sm120_dynamic_moe = Mock(side_effect=lambda **kw: self.events.append(('dynamic', kw)))
        self.static = ModuleType(package + '.moe_static_ep_tiled')
        self.static.launch_ep_tiled_decode = Mock(side_effect=lambda **kw: self.events.append(('static', kw)))
        self.remap = ModuleType(package + '.glm53_ep_route_remap')
        self.remap.try_remap_ep_local = Mock(side_effect=self.remap_call)
        self.canary = ModuleType(package + '.glm53_ep_tiled_selftest')
        self.canary.before_relayout = Mock(side_effect=lambda *a: self.events.append(('before', a)))
        self.canary.after_relayout = Mock(side_effect=lambda *a: self.events.append(('after', a)))
        self.modules = {'torch': self.torch, package: ModuleType(package),
            self.md.__name__: self.md, self.static.__name__: self.static,
            self.remap.__name__: self.remap, self.canary.__name__: self.canary}
        self.context = patch.dict(sys.modules, self.modules)
        self.context.start()
        test.addCleanup(self.context.stop)
        spec = importlib.util.spec_from_file_location(package + '.glm53_ep_tiled', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        test.addCleanup(lambda: sys.modules.pop(spec.name, None))
        spec.loader.exec_module(self.module)
        self.real_shared_workspace = self.module._shared_workspace
        self.workspace = SimpleNamespace(static=object(),
            dynamic=SimpleNamespace(ep_tiled=True, ep_scatter_fp32=Tensor((capacity, 4096))),
            scratch=SimpleNamespace(scatter_fp32=Tensor((32, 4096))))
        self.module._shared_workspace = Mock(return_value=self.workspace)
        self.owner = SimpleNamespace(_use_ep=True, _ep_no_dummy=True, global_num_experts=288,
            num_local_experts=72, hidden_dim=4096, intermediate_size_per_partition=2048,
            topk=8, _activation_str='swigluoai_uninterleave', _swiglu_alpha=1.,
            _swiglu_beta=0., _swiglu_limit=10., max_num_tokens=capacity,
            _ep_stock_topk_micro=False, _ep_disable_micro=False, out_dtype='bf16',
            local_expert_offset=72, _ep_tiled=True, _sf6_weight_views=None,
            w1_sf_mma=Tensor((72, 4096, 256), 'u8'), w2_sf_mma=Tensor((72, 4096, 128), 'u8'),
            w1_scale=Tensor((72, 4096, 256), 'u8'), w2_scale=Tensor((72, 4096, 128), 'u8'),
            g1_alphas=Tensor((72,)), g2_alphas=Tensor((72,)), _fc2_input_scale=Tensor((72,)))
        self.owner._ensure_ep_scratch = Mock(side_effect=self.scratch)
        self.layer = SimpleNamespace(w13_weight=Tensor((72, 4096, 2048), 'u8'),
                                     w2_weight=Tensor((72, 4096, 1024), 'u8'))

    def relayout(self, w1, w2):
        self.events.append(('relayout', (w1, w2)))
        for tensor in (w1, w2):
            tensor._b12x_tile_major = 'plain'
            tensor._version += 1

    def views(self, w1, s1, w2, s2, a1, a2, **kwargs):
        self.events.append(('views', kwargs))
        converted1, converted2 = Tensor(s1.shape, 'u8'), Tensor(s2.shape, 'u8')
        return SimpleNamespace(tiled=True, packed_only=False,
            w13_tiled_storage=w1, w2_tiled_storage=w2,
            w13_fp4=w1, down_fp4=w2, w1_scale_storage=converted1, w2_scale_storage=converted2,
            _w13_sf_storage=converted1, _down_sf_storage=converted2, w1_alpha=a1, w2_alpha=a2,
            sfb_w13_ptr=converted1, sfb_down_ptr=converted2)

    def scratch(self, device, scale_dtype, map_dtype):
        self.events.append(('scratch', (device, scale_dtype, map_dtype)))
        self.owner._ep_ids = Tensor((self.owner.max_num_tokens, 8), 'i32', device=device)
        self.owner._ep_scales = Tensor((self.owner.max_num_tokens, 8), scale_dtype, device=device)

    def remap_call(self, ids, scales, **kwargs):
        self.events.append(('remap', (ids, scales, kwargs)))
        return True

    def prepare(self):
        self.module.prepare_ep_tiled(self.owner, self.layer)
        return self

    def inputs(self, tokens):
        return dict(output=Tensor((tokens, 4096), 'bf16'), x=Tensor((tokens, 4096), 'bf16'),
                    w1=self.layer.w13_weight, w2=self.layer.w2_weight,
                    ids=Tensor((tokens, 8), 'i64'), scales=Tensor((tokens, 8)), expert_map=None)


class OwnerTests(unittest.TestCase):
    def test_prepare_preserves_single_weight_storage_and_raw_scale_owners(self):
        h = Harness(self).prepare()
        names = [name for name, _ in h.events]
        self.assertLess(names.index('before'), names.index('relayout'))
        self.assertLess(names.index('relayout'), names.index('views'))
        self.assertLess(names.index('scratch'), names.index('after'))
        views = h.owner._ep_tiled_weight_views
        self.assertIs(views.w13_tiled_storage, h.layer.w13_weight)
        self.assertIs(views.w2_tiled_storage, h.layer.w2_weight)
        self.assertIs(views._w13_sf_storage, views.w1_scale_storage)
        self.assertIs(views._down_sf_storage, views.w2_scale_storage)
        self.assertIsNot(views._w13_sf_storage, h.owner.w1_sf_mma)
        self.assertIsNot(views._down_sf_storage, h.owner.w2_sf_mma)
        self.assertIsNotNone(h.owner.w1_scale)
        self.assertIsNotNone(h.owner.w2_scale)
        h.owner._ensure_ep_scratch.assert_called_once_with('cuda:0', 'f32', 'i32')
        self.assertFalse(h.owner._ep_tiled_canary_active)

    def test_all_supported_tokens_dispatch_once_and_return_original_output_without_allocation(self):
        h = Harness(self, 16384).prepare()
        h.torch.cuda.is_current_stream_capturing.return_value = True
        h.torch.empty.side_effect = AssertionError('inference allocation')
        h.owner._ensure_ep_scratch.side_effect = AssertionError('inference scratch replacement')
        scratch_ids, scratch_weights = h.owner._ep_ids, h.owner._ep_scales
        for tokens in (1, 6, 12, 18, 24, 32, 33, 127, 4095, 4096, 8192, 16384):
            with self.subTest(tokens=tokens):
                h.events.clear()
                args = h.inputs(tokens)
                result = h.module.launch_ep_tiled(h.owner, **args)
                self.assertIs(result, args['output'])
                self.assertEqual([event[0] for event in h.events],
                                 ['remap', 'static' if tokens <= 32 else 'dynamic'])
                _, (_, _, remap) = h.events[0]
                self.assertTrue(remap['_tiled_owner'])
                self.assertEqual(remap['out_ids'].shape, (tokens, 8))
                self.assertIs(remap['out_ids'].base, scratch_ids)
                self.assertIs(remap['out_scales'].base, scratch_weights)
                sent = h.events[1][1]
                self.assertIs(sent['topk_ids'], remap['out_ids'])
                self.assertIs(sent['topk_weights'], remap['out_scales'])
                self.assertIs(sent['weights'], h.owner._ep_tiled_weight_views)
                self.assertIs(sent['a'], args['x'])
        self.assertEqual(h.module._LAUNCHED, set())
        h.torch.empty.assert_not_called()

    def test_unsupported_input_or_changed_weight_generation_never_reaches_remap(self):
        cases = [dict(x=Tensor((6, 4096), 'bf16', contiguous=False)),
                 dict(output=Tensor((6, 4096), 'f32')),
                 dict(ids=Tensor((6, 8), 'f32')),
                 dict(scales=Tensor((6, 8), 'bf16')),
                 dict(expert_map=object())]
        # expert_map is validated by the actual remapper, covered separately.
        for override in cases[:-1]:
            h = Harness(self).prepare()
            args = h.inputs(6)
            args.update(override)
            with self.assertRaises(ValueError):
                h.module.launch_ep_tiled(h.owner, **args)
            h.remap.try_remap_ep_local.assert_not_called()
        for mutation in ('version', 'storage', 'marker'):
            h = Harness(self).prepare()
            args = h.inputs(6)
            if mutation == 'version': args['w1']._version += 1
            if mutation == 'storage': args['w2'] = Tensor(args['w2'].shape, 'u8')
            if mutation == 'marker': args['w2']._b12x_tile_major = False
            with self.assertRaises(RuntimeError):
                h.module.launch_ep_tiled(h.owner, **args)
            h.remap.try_remap_ep_local.assert_not_called()

    def test_remap_decline_or_launch_error_cannot_fall_back_to_row_major(self):
        for tokens in (6, 33):
            h = Harness(self).prepare()
            h.remap.try_remap_ep_local.side_effect = None
            h.remap.try_remap_ep_local.return_value = False
            with self.assertRaisesRegex(ValueError, 'remap'):
                h.module.launch_ep_tiled(h.owner, **h.inputs(tokens))
            h.static.launch_ep_tiled_decode.assert_not_called()
            h.md.launch_sm120_dynamic_moe.assert_not_called()
            h.remap.try_remap_ep_local.side_effect = h.remap_call
            error = RuntimeError('device launch failure')
            target = h.static.launch_ep_tiled_decode if tokens <= 32 else h.md.launch_sm120_dynamic_moe
            target.side_effect = error
            with self.assertRaises(RuntimeError) as got:
                h.module.launch_ep_tiled(h.owner, **h.inputs(tokens))
            self.assertIs(got.exception, error)
            self.assertEqual(h.module._LAUNCHED, set())

    def test_tiled_dynamic_scatter_never_allocates_or_replaces_graph_storage(self):
        h = Harness(self)
        fn = extract_function(DISPATCH, '_ep_local_scatter_buffer', {'torch': h.torch})
        workspace = SimpleNamespace(device='cuda:0', ep_tiled=True,
                                    ep_scatter_fp32=Tensor((64, 4096), 'f32'))
        storage = workspace.ep_scatter_fp32
        for tokens in (1, 6, 33, 64):
            output = Tensor((tokens, 4096), 'bf16')
            result = fn(workspace, output, tokens, 4096)
            self.assertIs(result.base, storage)
            self.assertIs(workspace.ep_scatter_fp32, storage)
            self.assertEqual(result.data_ptr(), storage.data_ptr())
        h.torch.empty.assert_not_called()

        for replacement in (None, Tensor((32, 4096), 'f32')):
            workspace.ep_scatter_fp32 = replacement
            with self.assertRaises(RuntimeError): fn(workspace, Tensor((33, 4096), 'bf16'), 33, 4096)
        h.torch.empty.assert_not_called()

    def test_failed_startup_cannot_launch_or_take_repeated_finalisation_shortcut(self):
        h = Harness(self)
        failure = RuntimeError('canary numerics failure')
        h.canary.after_relayout.side_effect = failure
        with self.assertRaises(RuntimeError) as got:
            h.prepare()
        self.assertIs(got.exception, failure)
        self.assertFalse(h.owner._ep_tiled_ready)
        self.assertFalse(h.owner._ep_tiled_canary_active)
        with self.assertRaisesRegex(RuntimeError, 'startup validation'):
            h.module.launch_ep_tiled(h.owner, **h.inputs(6))
        process = wrapper_method('process_weights_after_loading')
        with patch.dict(sys.modules, {FULL_OWNER_MODULE: h.module}):
            with self.assertRaisesRegex(RuntimeError, 'startup validation'):
                process(h.owner, h.layer)
        h.md.tile_expert_weights_inplace.assert_called_once()
        h.remap.try_remap_ep_local.assert_not_called()

    def test_actual_canary_may_launch_but_cannot_emit_production_witness(self):
        h = Harness(self)
        def after(*args):
            self.assertFalse(h.owner._ep_tiled_ready)
            self.assertTrue(h.owner._ep_tiled_canary_active)
            inputs = h.inputs(6)
            self.assertIs(h.module.launch_ep_tiled(h.owner, **inputs), inputs['output'])
        h.canary.after_relayout.side_effect = after
        with patch('builtins.print') as printed:
            h.prepare()
        self.assertTrue(h.owner._ep_tiled_ready)
        printed.assert_not_called()
        self.assertEqual(h.module._LAUNCHED, set())

    def test_scale_source_view_and_alpha_generation_changes_refuse_before_remap(self):
        fields = ('w1_scale', 'w2_scale', 'w1_sf_mma', 'w2_sf_mma',
                  'g1_alphas', 'g2_alphas', '_fc2_input_scale',
                  '_w13_sf_storage', '_down_sf_storage', 'w1_alpha', 'w2_alpha')
        for field in fields:
            for mode in ('mutate', 'replace'):
                with self.subTest(field=field, mode=mode):
                    h = Harness(self).prepare()
                    owner = h.owner if hasattr(h.owner, field) else h.owner._ep_tiled_weight_views
                    old = getattr(owner, field)
                    if mode == 'mutate': old._version += 1
                    else: setattr(owner, field, Tensor(old.shape, old.dtype))
                    with self.assertRaises(RuntimeError):
                        h.module.launch_ep_tiled(h.owner, **h.inputs(6))
                    h.remap.try_remap_ep_local.assert_not_called()

    def test_output_aliases_decline_including_partial_byte_overlap_and_scale_planes(self):
        for name in ('x', 'ids', 'scales', 'w1', 'w2', 'remap', 'scatter', 'raw_scale', 'converted_scale'):
            h = Harness(self).prepare()
            args = h.inputs(6)
            tensor = dict(remap=h.owner._ep_ids, scatter=h.workspace.scratch.scatter_fp32,
                          raw_scale=h.owner.w1_scale,
                          converted_scale=h.owner._ep_tiled_weight_views._w13_sf_storage).get(name, args.get(name))
            args['output'] = Tensor((6, 4096), 'bf16', pointer=tensor.data_ptr() + 2)
            with self.subTest(name=name), self.assertRaises(ValueError):
                h.module.launch_ep_tiled(h.owner, **args)
            h.remap.try_remap_ep_local.assert_not_called()
        h = Harness(self)
        adjacent = Tensor((1, 4096), 'bf16')
        output = Tensor((1, 4096), 'bf16', pointer=adjacent.data_ptr() + adjacent.numel() * 2)
        h.module._require_output_disjoint(output, (adjacent,))

    def test_off_wrapper_prefix_preserves_legacy_tp_and_ep_dispatch(self):
        h = Harness(self)
        launch = Mock(return_value='tiled output')
        fake_module = SimpleNamespace(launch_ep_tiled=launch)
        apply = wrapper_method('apply', prefix_only=True)
        with patch.dict(sys.modules, {FULL_OWNER_MODULE: fake_module}):
            for use_ep in (False, True):
                owner = SimpleNamespace(_sf6_weight_views=None, _use_ep=use_ep, _ep_tiled=False)
                self.assertEqual(invoke_apply(apply, owner), 'legacy continuation')
            launch.assert_not_called()
            owner._ep_tiled = True
            self.assertEqual(invoke_apply(apply, owner), 'tiled output')
            launch.assert_called_once()

    def test_explicit_backend_forcing_refuses_before_in_place_weight_transform(self):
        for name, value in (('_FORCED_BACKEND', 'dynamic'),
                            ('_GLM53_B12X_FORCE_BACKEND', 'micro'),
                            ('_STATIC_V2_OVERRIDE', {'tiled': False})):
            h = Harness(self)
            setattr(h.md, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                h.prepare()
            h.md.tile_expert_weights_inplace.assert_not_called()
        h = Harness(self)
        with patch.dict(os.environ, {h.md._FORCE_MOE_W4A16_ENV: '1'}):
            with self.assertRaises(ValueError): h.prepare()
        h.md.tile_expert_weights_inplace.assert_not_called()

    def test_configuration_rejects_other_models_modes_and_unwitnessed_capacity(self):
        h = Harness(self)
        good = vars(h.owner).copy()
        for key, value in dict(_use_ep=False, _ep_no_dummy=False, num_local_experts=73,
                               global_num_experts=72, intermediate_size_per_partition=512,
                               hidden_dim=2048, topk=1, _swiglu_limit=7.,
                               _ep_stock_topk_micro=True, _ep_disable_micro=True).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.module.validate_configuration(SimpleNamespace(**dict(good, **{key: value})))
        for capacity in (0, 32, 8191, 16385, True, 8192.):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                h.module.validate_configuration(SimpleNamespace(**dict(good, max_num_tokens=capacity)))
        for capacity in (8192, 9216, 16384):
            h.module.validate_configuration(SimpleNamespace(**dict(good, max_num_tokens=capacity)))

    def test_real_shared_factory_prewarms_once_and_owner_reference_keeps_graph_scratch_alive(self):
        h = Harness(self)
        h.md.allocate_sm120_static_workspace = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
        h.md.allocate_sm120_dynamic_workspace = Mock(side_effect=lambda **kw: SimpleNamespace(
            device=kw['device'], max_rows=kw['routed_rows']))
        h.static.allocate_ep_tiled_decode_scratch = Mock(side_effect=lambda **kw: SimpleNamespace(
            scatter_fp32=Tensor((32, 4096))))
        h.static.warm_ep_tiled_decode = Mock()
        h.md._get_dynamic_kernel = Mock()
        first = h.real_shared_workspace('cuda:0', 8192)
        second = h.real_shared_workspace('cuda:0', 8192)
        self.assertIs(first, second)
        self.assertIs(first.scratch.scatter_fp32, second.scratch.scatter_fp32)
        self.assertIs(first.dynamic.ep_scatter_fp32, second.dynamic.ep_scatter_fp32)
        h.md.allocate_sm120_static_workspace.assert_called_once()
        h.md.allocate_sm120_dynamic_workspace.assert_called_once()
        h.static.warm_ep_tiled_decode.assert_called_once_with()
        h.md._get_dynamic_kernel.assert_called_once()
        self.assertTrue(h.md._get_dynamic_kernel.call_args.kwargs['tiled'])
        self.assertEqual(first.dynamic.ep_scatter_fp32.shape, (8192, 4096))
        self.assertTrue(first.dynamic.ep_tiled)
        owner = SimpleNamespace(workspace=first)
        weak = weakref.ref(first)
        del first, second
        gc.collect()
        self.assertIs(weak(), owner.workspace)
        del owner
        gc.collect()
        self.assertIsNone(weak())
        self.assertEqual(len(h.module._WORKSPACES), 0)

    def test_dynamic_selector_leaves_tp_and_off_paths_unmodified(self):
        h = Harness(self)
        kernel = ModuleType(h.package + '.moe_dynamic_ep_local')
        kernel.MoEGatedEPLocalKernel = object()
        kernel.stock_contract_matches = lambda: True
        namespace = dict(__package__=h.package, _GLM53_EP_TILED=False,
            _GLM53_EP_PREFILL_LOCAL=False, _FORCED_BACKEND=None,
            _FORCE_MOE_W4A16_ENV=h.md._FORCE_MOE_W4A16_ENV, os=os, torch=h.torch,
            select_sm120_moe_backend=Mock(return_value='static'))
        fn = extract_function(DISPATCH, '_ep_local_prefill_kernel', namespace)
        arguments = dict(E=72, m=33, k=4096, n=2048, num_topk=8, tile_m=128,
            activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
            swiglu_limit=10., quant_mode='nvfp4', tiled=True)
        with patch.dict(sys.modules, {kernel.__name__: kernel}):
            self.assertIsNone(fn(**arguments))
            namespace['_GLM53_EP_TILED'] = True
            for tokens in (1, 32, 33, 4095, 4096, 16384):
                self.assertIs(fn(**dict(arguments, m=tokens)), kernel.MoEGatedEPLocalKernel)
                self.assertIsNone(fn(**dict(arguments, E=288, n=512, m=tokens)))
            for tokens in (0, True, 33., 16385):
                with self.assertRaises(ValueError): fn(**dict(arguments, m=tokens))
            self.assertIsNone(fn(**dict(arguments, tiled=False)))
            namespace['_GLM53_EP_PREFILL_LOCAL'] = True
            self.assertIsNone(fn(**dict(arguments, tiled=False, m=4095)))
            namespace['select_sm120_moe_backend'].return_value = 'dynamic'
            self.assertIs(fn(**dict(arguments, tiled=False, m=4096)), kernel.MoEGatedEPLocalKernel)


FULL_OWNER_MODULE = 'flashinfer.fused_moe.cute_dsl.blackwell_sm12x.glm53_ep_tiled'


def wrapper_method(name, prefix_only=False):
    cls = next(n for n in ast.parse(WRAPPER.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'FlashInferB12xExperts')
    node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    if prefix_only:
        # Exercise the actual new branch, then stop at the unchanged legacy
        # continuation: this test does not emulate all older TP/EP kernels.
        branch = next(i for i, n in enumerate(node.body) if isinstance(n, ast.If)
                      and any(isinstance(a, ast.Constant) and a.value == '_ep_tiled' for a in ast.walk(n.test)))
        node.body = node.body[:branch + 1] + [ast.Return(value=ast.Constant('legacy continuation'))]
    namespace = {}
    module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(WRAPPER), 'exec'), namespace)
    return namespace[name]


def invoke_apply(fn, owner):
    return fn(owner, *[object() for _ in range(fn.__code__.co_argcount - 1)])


if __name__ == '__main__':
    unittest.main()
