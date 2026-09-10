"""Host admission/cache/owner tests; no CUDA numerics or timing claims."""
import ast
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_glm53_ep_hybrid_owner as hybrid
import test_glm53_ep_tiled_owner as base
import test_glm53_ep_tiled_selftest as canary

ROOT = Path(__file__).resolve().parents[1]
FLAG = 'VLLM_GLM53_EP_HYBRID_Q0_DUAL_WARP'
TAG = 'glm53_ep2tp2_q0_dual_warp_v1'
C = canary.canary


def dispatch_functions():
    path = ROOT/'overlay/modules/glm53_moe/moe_dispatch.py'
    tree = ast.parse(path.read_text())
    names = {'_ep_hybrid_q0_dual_warp_eligible', '_dynamic_kernel_cache_key'}
    body = [ast.parse('from __future__ import annotations').body[0]]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def key_kwargs():
    return dict(activation_precision='fp4', quant_mode='nvfp4', E=144,
        k=4096, n=1024, num_topk=8, mac=48, mma_tiler_mn=(128,128),
        topk_ids_dtype='torch.int32', input_scales_are_reciprocal=False,
        fast_math=True, activation='swigluoai_uninterleave', swiglu_alpha=1.,
        swiglu_beta=0., swiglu_limit=10., share_input_across_experts=False,
        ep_local_prefill=True, tiled=True, reform_sf_pack=True)


class DualWarpDispatch(unittest.TestCase):
    def test_exact_selector_and_rejected_domains(self):
        select = dispatch_functions()['_ep_hybrid_q0_dual_warp_eligible']
        args = dict(enabled=True,E=144,m=8192,k=4096,n=1024,num_topk=8,
            tile_m=128,quant_mode='nvfp4',tiled=True,reform_sf_pack=True,
            ep_local=True,activation='swigluoai_uninterleave',swiglu_alpha=1.,
            swiglu_beta=0.,swiglu_limit=10.)
        for rows in (33,2128,4096,8192,16384):
            self.assertTrue(select(**dict(args,m=rows)))
        for change in ({'m':32},{'m':16385},{'m':8192.},{'m':True},
                       {'enabled':False},{'E':72},{'n':2048},{'tile_m':64},
                       {'tiled':False},{'reform_sf_pack':False},{'ep_local':False},
                       {'activation':'silu'},{'quant_mode':'mxfp4'},{'num_topk':4}):
            with self.subTest(change=change):
                self.assertFalse(select(**(args|change)))

    def test_cache_separation_preserves_baseline_and_rejects_wrong_lane(self):
        key = dispatch_functions()['_dynamic_kernel_cache_key']
        args = key_kwargs()
        baseline = key(**args)
        candidate = key(**args, ep_hybrid_q0_dual_warp=True)
        self.assertEqual(len(baseline),21)
        self.assertEqual(candidate,baseline+(TAG,))
        self.assertEqual(key(**args,ep_hybrid_q0_dual_warp=False),baseline)
        self.assertEqual(len(key(**(args|dict(E=72,n=2048)))),20)
        for invalid in (1,0,'1',None):
            with self.assertRaises(TypeError):key(**args,ep_hybrid_q0_dual_warp=invalid)
        for change in ({'E':72,'n':2048},{'ep_local_prefill':False},
                       {'reform_sf_pack':False},{'tiled':False},{'mma_tiler_mn':(64,128)},
                       {'share_input_across_experts':True},{'tp_sf6_q0':True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                key(**(args|change),ep_hybrid_q0_dual_warp=True)

    def test_owner_opt_in_freeze_and_decode_unchanged(self):
        with patch.dict(os.environ,{FLAG:'1'}):
            old = base.Harness(self)
            with self.assertRaises(ValueError):old.prepare()
            old.md.tile_expert_weights_inplace.assert_not_called()
            h = hybrid.hybrid_harness(self).prepare()
            self.assertTrue(h.owner._ep_tiled_q0_dual_warp)
            self.assertIs(h.module._shared_workspace.call_args.kwargs['q0_dual_warp'],True)
            with redirect_stdout(io.StringIO()) as log:
                h.module.launch_ep_tiled(h.owner,**h.inputs(6))
                native_args = h.static.launch_ep_tiled_decode.call_args.kwargs
                self.assertNotIn('_ep_hybrid_q0_dual_warp_override',native_args)
                self.assertIs(native_args['decode_opt'],False)
                h.owner._ep_tiled_canary_active = True
                h.module.launch_ep_tiled(h.owner,**h.inputs(33))
                self.assertNotIn('[ep-hybrid-q0-dual-warp]',log.getvalue())
                h.owner._ep_tiled_canary_active = False
                for rows in (33,4096,8192):
                    h.module.launch_ep_tiled(h.owner,**h.inputs(rows))
                    self.assertIs(h.md.launch_sm120_dynamic_moe.call_args.kwargs[
                        '_ep_hybrid_q0_dual_warp_override'],True)
            self.assertEqual(log.getvalue().count('[ep-hybrid-q0-dual-warp] LAUNCHED'),1)
            h.md.launch_sm120_dynamic_moe.reset_mock()
            with patch.dict(os.environ,{FLAG:'0'}),self.assertRaisesRegex(RuntimeError,'mode changed'):
                h.module.launch_ep_tiled(h.owner,**h.inputs(4096))
            h.md.launch_sm120_dynamic_moe.assert_not_called()
        for value in ('true','yes','2',''):
            with patch.dict(os.environ,{FLAG:value}), self.assertRaises(ValueError):
                h.module.ep_hybrid_q0_dual_warp_enabled(hybrid.H)

    def test_workspace_isolation_and_explicit_baseline_compile(self):
        h = base.Harness(self)
        h.md.allocate_sm120_static_workspace = Mock(side_effect=lambda **kw:SimpleNamespace(**kw))
        h.md.allocate_sm120_dynamic_workspace = Mock(side_effect=lambda **kw:SimpleNamespace(
            device=kw['device'],max_rows=kw['routed_rows']))
        h.static.allocate_ep_tiled_decode_scratch = Mock(return_value=SimpleNamespace())
        h.md._get_dynamic_kernel = Mock()
        baseline = h.real_shared_workspace('cuda:0',8192,shard=hybrid.H)
        self.assertIs(h.md._get_dynamic_kernel.call_args.kwargs['_ep_hybrid_q0_dual_warp_override'],False)
        candidate = h.real_shared_workspace('cuda:0',8192,shard=hybrid.H,q0_dual_warp=True)
        self.assertIsNot(baseline,candidate)
        self.assertIs(h.md._get_dynamic_kernel.call_args.kwargs['_ep_hybrid_q0_dual_warp_override'],True)
        self.assertIs(candidate,h.real_shared_workspace('cuda:0',8192,shard=hybrid.H,q0_dual_warp=True))
        self.assertEqual(h.md._get_dynamic_kernel.call_count,2)
        self.assertIs(h.static.warm_ep_tiled_decode.call_args.kwargs['decode_opt'],False)
        with self.assertRaises(ValueError):h.real_shared_workspace('cuda:0',8192,q0_dual_warp=True)

    def test_canary_requires_selected_dynamic_key_and_retains_native_keys(self):
        key = dispatch_functions()['_dynamic_kernel_cache_key'](**key_kwargs())
        context = dict(shard=hybrid.H, q0_dual_warp=True,
                       md=SimpleNamespace(_DYNAMIC_KERNEL_CACHE={key+(TAG,):object()}))
        self.assertEqual(C._cache_evidence(context,object(),8192)['keys'],[repr(key+(TAG,))])
        context['md']._DYNAMIC_KERNEL_CACHE = {key:object()}
        with self.assertRaises(AssertionError):C._cache_evidence(context,object(),8192)
        decode = hybrid.actual_native_key_fixture()
        context['decode'] = decode
        owner = SimpleNamespace(_ep_tiled_workspace=SimpleNamespace(
            static=SimpleNamespace(max_rows=256),scratch=SimpleNamespace(max_active_clusters=48)))
        for rows in (4,6,8,12,16,24,32):
            native = decode.native_key(rows)
            decode._EP_TILED_KERNEL_CACHE = {native:object()}
            evidence = C._cache_evidence(context,owner,rows)
            self.assertEqual(evidence['keys'],[repr(native)])
            self.assertIs(evidence['decode_opt'],False)

    def test_canary_mode_binding_and_separate_once_cache(self):
        context = canary.runtime()|dict(shard=hybrid.H,loader_identity=hybrid.identity(0),
                                      rank_geometry={'physical_rank':0})
        self.assertNotEqual(C._key(context),C._key(context|{'q0_dual_warp':True}))
        C._STATES.clear()
        with redirect_stdout(io.StringIO()),patch.object(C,'_runtime',return_value=context|{'q0_dual_warp':True}), \
             patch.object(C,'_memory',return_value={}),patch.object(C,'_capture_references',return_value={}), \
             patch.object(C,'_validate_candidate'):
            owner,layer=object(),object()
            handle=C.before_relayout(owner,layer)
            receipt=C.after_relayout(owner,layer,handle)
            self.assertIs(receipt['q0_dual_warp'],True)
            self.assertEqual(receipt['schema'],2)
        C._STATES.clear()


if __name__ == '__main__':
    unittest.main()
