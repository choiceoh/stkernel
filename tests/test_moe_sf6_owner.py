"""CPU checks for packed-only model ownership and wrapper calls."""
from __future__ import annotations

import ast
import gc
import importlib
import logging
import os
from pathlib import Path
import sys
import types
from typing import Any, Optional, Tuple
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'overlay/modules/glm53_moe'


def functions(filename, names, namespace):
    tree = ast.parse((MODULE/filename).read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError((filename, names, [node.name for node in selected]))
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def wrapper_run():
    tree = ast.parse((MODULE/'b12x_moe.py').read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'B12xMoEWrapper')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    method.decorator_list = []
    namespace = dict(torch=torch, Any=Any, Optional=Optional, Tuple=Tuple,
                     _is_cuda_graph_capturing=lambda: True, __package__='_test_sf6_wrapper')
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'b12x_moe.py', 'exec'), namespace)
    return namespace['run']


def owner_namespace():
    names = {'_b12x_sf6_requested', '_b12x_sf6_generation', '_b12x_require_packed_owner',
             '_b12x_release_raw_scales', 'finalize_packed_scale_owners'}
    return functions('flashinfer_b12x_moe.py', names,
                     dict(torch=torch, os=os, _SF6_PENDING=weakref.WeakKeyDictionary(),
                          logger=logging.getLogger('sf6-owner-test')))


def expert_method(name, namespace):
    tree = ast.parse((MODULE/'flashinfer_b12x_moe.py').read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'FlashInferB12xExperts')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'flashinfer_b12x_moe.py', 'exec'), namespace)
    return namespace[name]


class FakeExperts:
    @property
    def w1_scale(self):
        return self.quant_config._w1.scale

    @property
    def w2_scale(self):
        return self.quant_config._w2.scale


def model_layer():
    layer = torch.nn.Module()
    for name, shape in (('w13_weight', (2,256,256)), ('w2_weight', (2,512,64)),
                        ('w13_weight_scale', (2,256,32)), ('w2_weight_scale', (2,512,8))):
        layer.register_parameter(name, torch.nn.Parameter(torch.full(shape, 17, dtype=torch.uint8),
                                                         requires_grad=False))
    expert = FakeExperts()
    expert.quant_config = types.SimpleNamespace(
        _w1=types.SimpleNamespace(scale=layer.w13_weight_scale),
        _w2=types.SimpleNamespace(scale=layer.w2_weight_scale))
    expert.w1_sf_mma = layer.w13_weight_scale.view(-1)
    expert.w2_sf_mma = layer.w2_weight_scale.view(-1)
    expert._sf6_weight_views = expert._sf6_generation = None
    expert._sf6_finalized = False
    expert._use_ep = False
    expert._wrapper = None
    expert.g1_alphas = torch.ones(2)
    expert.g2_alphas = torch.ones(2)
    expert.hidden_dim = 512
    expert.global_num_experts = expert.num_local_experts = 2
    expert.topk = 2
    expert._activation_str = 'swigluoai_uninterleave'
    expert._swiglu_alpha, expert._swiglu_beta, expert._swiglu_limit = 1., 0., 10.
    return layer, expert


class ModelOwnership(unittest.TestCase):
    def setUp(self):
        self.ns = owner_namespace()
        package = types.ModuleType('_test_sf6_model_owner')
        package.__path__ = [str(MODULE)]
        self.modules = patch.dict(sys.modules, {package.__name__: package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.pack = importlib.import_module(package.__name__ + '.moe_reform_sf_pack')
        self.dispatch = types.ModuleType('flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch')
        self.dispatch.prepare_packed_only_weight_views = Mock(side_effect=self.prepare)
        parent = types.ModuleType('flashinfer.fused_moe.cute_dsl.blackwell_sm12x')
        parent.moe_dispatch = self.dispatch
        self.imports = patch.dict(sys.modules, {parent.__name__: parent,
                                               self.dispatch.__name__: self.dispatch})
        self.imports.start()
        self.addCleanup(self.imports.stop)

    def prepare(self, **kwargs):
        scales = self.pack.prepare_reform_scales(
            kwargs['w1_blockscale'], kwargs['w2_blockscale'], experts=2, n=128, k=512)
        if not scales.enabled:
            return None
        return types.SimpleNamespace(packed_only=True, reform_scales=scales,
            sfb1_packed=scales.fc1, sfb2_packed=scales.fc2,
            sfb_w13_ptr=None, sfb_down_ptr=None, w1_scale_storage=None, w2_scale_storage=None,
            _w13_sf_storage=None, _down_sf_storage=None,
            w1_storage=kwargs['w1_fp4'], w2_storage=kwargs['w2_fp4'],
            w1_alpha=kwargs['w1_alphas'], w2_alpha=kwargs['w2_alphas'])

    def register(self, layer, expert):
        self.ns['_SF6_PENDING'][layer] = weakref.ref(expert)

    def test_real_lossless_owner_releases_all_raw_aliases_and_survives(self):
        layer, expert = model_layer()
        self.register(layer, expert)
        model = torch.nn.Sequential(layer)
        refs = [weakref.ref(value) for value in (layer.w13_weight_scale, layer.w2_weight_scale,
                                               expert.w1_sf_mma, expert.w2_sf_mma)]
        self.assertEqual(self.ns['finalize_packed_scale_owners'](model), 24576)
        self.dispatch.prepare_packed_only_weight_views.reset_mock(side_effect=True)
        self.dispatch.prepare_packed_only_weight_views.side_effect = AssertionError('repacked sealed owner')
        with patch.object(torch, 'empty', side_effect=AssertionError('allocation after finalisation')):
            self.assertEqual(self.ns['finalize_packed_scale_owners'](model), 0)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertIsNone(layer.w13_weight_scale)
        self.assertIsNone(layer.w2_weight_scale)
        self.assertIsNone(expert.w1_scale)
        self.assertIsNone(expert.w2_scale)
        self.assertIsNone(expert.w1_sf_mma)
        self.assertIsNone(expert.w2_sf_mma)
        owner = expert._sf6_weight_views
        self.assertEqual(owner.sfb1_packed.numel() + owner.sfb2_packed.numel(), 18624)
        from _test_sf6_model_owner.moe_sf_pack import unpack_sf_inline
        for packed in (owner.sfb1_packed, owner.sfb2_packed):
            self.assertTrue(bool((unpack_sf_inline(packed, 2048) == 17).all()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_nonrepresentable_layer_keeps_both_original_planes(self):
        layer, expert = model_layer()
        layer.w2_weight_scale.view(-1)[0] = 100
        originals = (layer.w13_weight_scale, layer.w2_weight_scale, expert.w1_sf_mma, expert.w2_sf_mma)
        self.register(layer, expert)
        model = torch.nn.Sequential(layer)
        self.assertEqual(self.ns['finalize_packed_scale_owners'](model), 0)
        self.assertEqual(self.ns['finalize_packed_scale_owners'](model), 0)
        self.assertEqual(self.dispatch.prepare_packed_only_weight_views.call_count, 1)
        for actual, original in zip((layer.w13_weight_scale, layer.w2_weight_scale,
                                     expert.w1_sf_mma, expert.w2_sf_mma), originals):
            self.assertIs(actual, original)
        self.assertIsNone(expert._sf6_weight_views)

    def test_only_requested_model_layers_and_non_ep_are_finalised(self):
        layer, expert = model_layer()
        other, other_expert = model_layer()
        ep, ep_expert = model_layer()
        ep_expert._use_ep = True
        for pair in ((layer, expert), (other, other_expert), (ep, ep_expert)):
            self.register(*pair)
        self.assertEqual(self.ns['finalize_packed_scale_owners'](torch.nn.Sequential(layer, ep)), 24576)
        self.assertIsNotNone(other.w13_weight_scale)
        self.assertIsNotNone(ep.w13_weight_scale)
        self.assertEqual(self.dispatch.prepare_packed_only_weight_views.call_count, 1)

    def test_pending_registry_never_owns_model_or_experts(self):
        layer, expert = model_layer()
        self.register(layer, expert)
        model_ref, expert_ref = weakref.ref(layer), weakref.ref(expert)
        del layer, expert
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertIsNone(expert_ref())
        self.assertEqual(len(self.ns['_SF6_PENDING']), 0)

    def test_bad_owner_or_alias_fails_before_source_release(self):
        for corruption in ('raw_field', 'different_storage', 'wrapper_started'):
            layer, expert = model_layer()
            self.register(layer, expert)
            if corruption == 'raw_field':
                def bad_prepare(**kwargs):
                    views = self.prepare(**kwargs)
                    views.w1_scale_storage = kwargs['w1_blockscale']
                    return views
                self.dispatch.prepare_packed_only_weight_views.side_effect = bad_prepare
            elif corruption == 'different_storage':
                self.dispatch.prepare_packed_only_weight_views.side_effect = self.prepare
                expert.quant_config._w1.scale = expert.w1_scale.clone()
            else:
                expert._wrapper = object()
            with self.subTest(corruption=corruption), self.assertRaises(RuntimeError):
                self.ns['finalize_packed_scale_owners'](torch.nn.Sequential(layer))
            self.assertIsNotNone(layer.w13_weight_scale)
            self.assertIsNotNone(layer.w2_weight_scale)
            self.assertIsNone(expert._sf6_weight_views)

    def test_requested_flag_has_no_substring_or_default_activation(self):
        for value, wanted in (('', False), ('t,r', False), ('t,r,sf60', False),
                              ('t,r,sf6', True), ('t,r, sf6 ', True)):
            with patch.dict(os.environ, {'VLLM_GLM53_B12X_STATIC_V2': value}):
                self.assertEqual(self.ns['_b12x_sf6_requested'](), wanted)

    def test_model_apply_and_repeated_postload_never_rebuild_raw(self):
        layer, expert = model_layer()
        self.register(layer, expert)
        self.ns['finalize_packed_scale_owners'](torch.nn.Sequential(layer))
        postload = expert_method('process_weights_after_loading', self.ns)
        apply = expert_method('apply', self.ns)
        expert._wrapper = types.SimpleNamespace(run=Mock())
        expert._ensure_wrapper = Mock()
        expert._direct_out = True
        expert._fc2_input_scale = torch.ones(2)
        arguments = dict(output=torch.empty((6,512), dtype=torch.bfloat16),
            hidden_states=torch.ones((6,512), dtype=torch.bfloat16),
            w1=layer.w13_weight, w2=layer.w2_weight,
            topk_weights=torch.ones((6,2)), topk_ids=torch.zeros((6,2), dtype=torch.int32),
            activation=None, global_num_experts=2, expert_map=None, a1q_scale=None, a2_scale=None,
            workspace13=None, workspace2=None, expert_tokens_meta=None, apply_router_weight_on_input=None)
        with patch.object(torch, 'empty', side_effect=AssertionError('allocation in sealed path')), \
             patch.object(torch, 'zeros', side_effect=AssertionError('allocation in sealed path')):
            postload(expert, layer)
            for _ in range(3):
                apply(expert, **arguments)
        self.assertEqual(expert._wrapper.run.call_count, 3)
        for call in expert._wrapper.run.call_args_list:
            self.assertIs(call.kwargs['_weight_views'], expert._sf6_weight_views)
            self.assertIsNone(call.kwargs['w1_weight_sf'])
            self.assertIsNone(call.kwargs['w2_weight_sf'])
        layer.w13_weight.add_(1)
        with self.assertRaises(RuntimeError):
            apply(expert, **arguments)
        with self.assertRaises(RuntimeError):
            postload(expert, layer)


class PackedWrapper(unittest.TestCase):
    def setUp(self):
        self.run = wrapper_run()
        self.dispatch = types.ModuleType('_test_sf6_wrapper.blackwell_sm12x.moe_dispatch')
        for name in ('_get_weight_views', 'static_v2_weights_layout', 'static_v2_weights_sf_pack',
                     'static_v2_weights_reform_sf_pack', '_sf6_tensor_version',
                     '_pad_intermediate_to_tile', 'is_gated_activation'):
            setattr(self.dispatch, name, Mock(side_effect=AssertionError('raw path called: ' + name)))
        self.dispatch._LEVEL_TILE_N = 128
        self.dispatch.select_sm120_moe_backend = Mock(return_value='static')
        self.dispatch.launch_sm120_moe = Mock(side_effect=lambda **kwargs: kwargs['scatter_output'])
        self.imports = patch.dict(sys.modules, {self.dispatch.__name__: self.dispatch})
        self.imports.start()
        self.addCleanup(self.imports.stop)
        self.wrapper = types.SimpleNamespace(use_cuda_graph=True, max_num_tokens=128,
            hidden_size=16, output_dtype=torch.bfloat16, quant_mode='nvfp4',
            _dynamic_workspace=None, _static_workspace=object(),
            num_experts=2, num_local_experts=2, top_k=2, activation='swigluoai_uninterleave',
            swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10., activation_precision='fp4',
            source_format='modelopt', intermediate_size=128)
        self.owner = types.SimpleNamespace(packed_only=True,
            reform_scales=types.SimpleNamespace(enabled=True))
        self.args = dict(x=torch.zeros((6,16), dtype=torch.bfloat16),
            w1_weight=torch.zeros((2,256,8), dtype=torch.uint8), w1_weight_sf=None,
            w2_weight=torch.zeros((2,16,64), dtype=torch.uint8), w2_weight_sf=None,
            w1_alpha=torch.ones(2), w2_alpha=torch.ones(2), fc2_input_scale=torch.ones(2),
            token_selected_experts=torch.zeros((6,2), dtype=torch.int32),
            token_final_scales=torch.ones((6,2)), out=torch.empty((6,16), dtype=torch.bfloat16),
            _weight_views=self.owner)

    def test_repeated_captured_calls_use_owner_without_allocating_or_raw_cache(self):
        with patch.object(torch, 'empty', side_effect=AssertionError('allocation in packed run')), \
             patch.object(torch, 'zeros', side_effect=AssertionError('allocation in packed run')):
            for _ in range(4):
                result = self.run(self.wrapper, **self.args)
                self.assertEqual(result.data_ptr(), self.args['out'].data_ptr())
        self.assertEqual(self.dispatch.launch_sm120_moe.call_count, 4)
        for call in self.dispatch.launch_sm120_moe.call_args_list:
            self.assertIs(call.kwargs['_weight_views'], self.owner)
            self.assertIsNone(call.kwargs['w1_weight_sf'])
            self.assertIsNone(call.kwargs['w2_weight_sf'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_dynamic_prefill_keeps_explicit_owner(self):
        self.wrapper._dynamic_workspace = object()
        self.dispatch.select_sm120_moe_backend.return_value = 'dynamic'
        self.run(self.wrapper, **self.args)
        call = self.dispatch.launch_sm120_moe.call_args.kwargs
        self.assertIs(call['_workspace'], self.wrapper._dynamic_workspace)
        self.assertIs(call['_weight_views'], self.owner)

    def test_invalid_or_raw_mixed_owner_is_rejected(self):
        for replacement in ({'_weight_views': types.SimpleNamespace()},
                            {'_weight_views': types.SimpleNamespace(packed_only=True)},
                            {'w1_weight_sf': torch.ones(1)},
                            {'w2_weight_sf': torch.ones(1)},
                            {'input_global_scale': torch.ones(1)},
                            {'_weight_views': None}):
            with self.subTest(replacement=list(replacement)):
                with self.assertRaises(ValueError):
                    self.run(self.wrapper, **dict(self.args, **replacement))
        self.dispatch.launch_sm120_moe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
