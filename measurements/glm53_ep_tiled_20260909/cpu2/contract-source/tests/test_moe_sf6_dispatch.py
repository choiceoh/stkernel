"""CPU ownership and direct-launch contracts over the actual dispatcher body."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import gc
import importlib
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
import weakref
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'overlay/modules/glm53_moe'
PACKAGE = '_sf6_dispatch_contract'
pkg = types.ModuleType(PACKAGE)
pkg.__path__ = [str(MODULE)]
sys.modules.setdefault(PACKAGE, pkg)
sf6 = importlib.import_module(PACKAGE+'.moe_reform_sf_pack')


def namespace():
    tree = ast.parse((MODULE/'moe_dispatch.py').read_text())
    names = {'_WeightViews', '_get_weight_views', '_prepared_reform_scales',
             '_sf6_tensor_version', '_register_cache_eviction', '_tile_expert_weights',
             '_scale_runtime_addresses', 'prepare_packed_only_weight_views',
             'launch_sm120_dynamic_moe', '_dynamic_kernel_cache_key',
             '_dynamic_workspace_tile_m'}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert len(nodes) == len(names)
    ns = dict(torch=torch, dataclass=dataclass, weakref=weakref, logging=logging,
        __package__=PACKAGE, _WEIGHT_CACHE={}, _REFORM_SF_CACHE={},
        _normalize_activation_precision=lambda x:x, _normalize_quant_mode=lambda x,a:x,
        _sf_params_for_quant_mode=lambda q:(16, 'sf8'), _level_tile_n=lambda a:128,
        _is_cuda_graph_capturing=lambda:False, prepare_reform_scales=sf6.prepare_reform_scales,
        TILED_W13_K_IN=512, TILED_W2_K_IN=128, _TILE_MAJOR_ATTR='_tile_major',
        make_ptr=lambda *a, **k: (a,k), cute=types.SimpleNamespace(AddressSpace=types.SimpleNamespace(gmem=1)),
        convert_sf_from_mma_layout=lambda sf,**kw:sf,
        _FORCED_BACKEND=None, _GLM53_B12X_FORCE_BACKEND=None,
        _ep_local_prefill_kernel=lambda **kw: None, _TP_SF6_Q0_ENABLED=False,
        _GLM53_B12X_PREFILL_REUSE=False, _GLM53_B12X_PREFILL_FC1_N128=False,
        _static_v2_config_for=lambda **kw:dict(tiled=True,reform_sf_pack=True),
        _select_dynamic_tile_m=lambda rows,experts,activation:16,
        _check_memref_limit=lambda *a:None, _expand_to_experts=lambda t,n:t,
        _sf_pack_dummy=Mock(side_effect=AssertionError('dummy requested on direct path')))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-sf6-dispatch', 'exec'), ns)
    return ns


class PackedViews(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        self.args = dict(w1_fp4=torch.zeros((2,512,256),dtype=torch.uint8),
            w2_fp4=torch.zeros((2,512,128),dtype=torch.uint8),
            w1_blockscale=torch.arange(2*512*512//16).remainder(64).to(torch.uint8),
            w2_blockscale=torch.arange(2*512*256//16).remainder(64).to(torch.uint8),
            w1_alphas=torch.ones(2),w2_alphas=torch.ones(2),n=256,k=512,
            tiled=True,reform_sf_pack=True)

    def test_final_owner_releases_raw_aliases_and_only_its_cache(self):
        get = self.ns['_get_weight_views']
        old = get(**self.args)
        other_key = ('fp4','nvfp4',True,False,999,1000,1,1001,1002,1)
        self.ns['_WEIGHT_CACHE'][other_key] = ('unrelated',)
        raw_refs = [weakref.ref(self.args[name]) for name in ('w1_blockscale','w2_blockscale')]
        owner = get(**self.args, packed_only=True)
        self.assertTrue(owner.packed_only)
        for name in ('sfb_w13_ptr','sfb_down_ptr','w1_scale_storage','w2_scale_storage',
                     '_w13_sf_storage','_down_sf_storage'):
            self.assertIsNone(getattr(owner,name),name)
        self.assertEqual(set(self.ns['_WEIGHT_CACHE']),{other_key})
        packed_before = owner.sfb1_packed.clone()
        del old, self.args
        gc.collect()
        self.assertTrue(all(ref() is None for ref in raw_refs))
        self.assertEqual(self.ns['_REFORM_SF_CACHE'],{})
        self.assertTrue(torch.equal(owner.sfb1_packed,packed_before))
        self.assertEqual(owner.sfb1_packed.numel()+owner.sfb2_packed.numel(),
                         (32768+16384)//2048*1552)

    def test_unrepresentable_layer_retains_both_raw_planes(self):
        self.args['w2_blockscale'][1] = 255
        refs = [weakref.ref(self.args[name]) for name in ('w1_blockscale','w2_blockscale')]
        owner = self.ns['_get_weight_views'](**self.args,packed_only=True)
        self.assertFalse(owner.packed_only)
        self.assertFalse(owner.reform_scales.enabled)
        self.assertIsNone(owner.sfb1_packed)
        self.assertIsNone(owner.sfb2_packed)
        del self.args
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertIsNotNone(owner._w13_sf_storage)
        self.assertIsNotNone(owner._down_sf_storage)

    def test_dead_pointer_slots_use_packed_addresses_and_reject_raw_launch(self):
        owner = self.ns['_get_weight_views'](**self.args,packed_only=True)
        ptrs = self.ns['_scale_runtime_addresses'](owner,direct_sf6=True)
        self.assertEqual(ptrs,(owner.sfb1_packed.data_ptr(),owner.sfb2_packed.data_ptr()))
        with self.assertRaisesRegex(RuntimeError,'raw-scale kernel'):
            self.ns['_scale_runtime_addresses'](owner,direct_sf6=False)

    def test_dynamic_prefill_launch_passes_compressed_planes_without_raw_tensors(self):
        owner = self.ns['_get_weight_views'](**self.args,packed_only=True)
        tensor = torch.zeros(16,dtype=torch.int32)
        names = ('packed_a_view','packed_input_scale','packed_a_flat','scale_flat','barrier_count',
                 'barrier_epoch','pair_head','task_head','task_tail','task_expert','task_valid_rows',
                 'row_counts','expert_write_rows','expert_tile_base','token_map','token_weights')
        ws = types.SimpleNamespace(**{n:tensor for n in names},max_rows=512,tile_m=32,
                                   physical_tiles_capacity=16,task_capacity=64)
        compiled = Mock()
        compiler = self.ns['_get_dynamic_kernel'] = Mock(return_value=(compiled,48))
        contract = types.ModuleType(PACKAGE+'.moe_dynamic_gated_sf6')
        contract.stock_contract_matches = lambda:True
        with patch.dict(sys.modules,{contract.__name__:contract}):
            self.ns['launch_sm120_dynamic_moe'](workspace=ws,weights=owner,
                a=torch.zeros((16,512),dtype=torch.bfloat16),
                topk_ids=torch.zeros((16,2),dtype=torch.int32),topk_weights=torch.ones((16,2)),
                input_gs=torch.ones(2),down_input_scale=torch.ones(2),
                scatter_output=torch.empty((16,512),dtype=torch.bfloat16),
                num_experts=2,num_tokens=16,k=512,n=256,top_k=2)
        self.assertTrue(compiler.call_args.kwargs['reform_sf_pack'])
        args = compiled.call_args.args
        self.assertEqual(args[15],owner.sfb1_packed.data_ptr())
        self.assertEqual(args[17],owner.sfb2_packed.data_ptr())
        self.assertIs(args[28],owner.sfb1_packed)
        self.assertIs(args[29],owner.sfb2_packed)
        self.assertEqual(args[30:],(16,512,512,64))
        self.assertFalse(torch.cuda.is_initialized())

    def test_ineligible_paths_decline_before_packing(self):
        prepare = self.ns['prepare_packed_only_weight_views']
        args = {key:value for key,value in self.args.items() if key not in ('tiled','reform_sf_pack')}
        args.update(num_experts=2,num_local_experts=2,num_topk=2,activation='swigluoai_uninterleave')
        self.ns['_get_weight_views'] = Mock(side_effect=AssertionError('ineligible pack'))
        for changes in (dict(num_local_experts=1),dict(activation_precision='bf16'),dict(quant_mode='mxfp4')):
            self.assertIsNone(prepare(**dict(args,**changes)))
        for setting in ('_FORCED_BACKEND','_GLM53_B12X_FORCE_BACKEND',
                        '_GLM53_B12X_PREFILL_REUSE','_GLM53_B12X_PREFILL_FC1_N128'):
            original = self.ns[setting]
            self.ns[setting] = 'micro' if 'BACKEND' in setting else True
            self.assertIsNone(prepare(**args))
            self.ns[setting] = original
        self.ns['_static_v2_config_for'] = lambda **kw:dict(tiled=True,reform_sf_pack=False)
        self.assertIsNone(prepare(**args))

    def test_sf6_workspace_uses_supported_gated_tile_for_prefill_tails(self):
        choose = self.ns['_dynamic_workspace_tile_m']
        args = dict(state_E=288,weight_E=288,k=4096,n=512,num_topk=8,
                    quant_mode='nvfp4',activation='swigluoai_uninterleave',swiglu_limit=10.)
        for rows in (648,4096,8192,65536):
            self.assertEqual(choose(routed_rows=rows,**args),128)
        self.ns['_static_v2_config_for'] = lambda **kw:dict(tiled=True,reform_sf_pack=False)
        self.assertEqual(choose(routed_rows=648,**args),16)


if __name__ == '__main__':
    unittest.main()
