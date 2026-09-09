"""CPU admission/cache/dispatch contracts for full-token expert-local prefill."""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
from typing import Tuple
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT/'overlay/modules/glm53_moe/moe_dispatch.py'
WR = ROOT/'overlay/modules/glm53_moe/flashinfer_b12x_moe.py'
MODEL = ROOT/'overlay/modules/glm53_model/glm5next_model.py'


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=[ast.parse('from __future__ import annotations').body[0],*body],type_ignores=[]), str(path),'exec'),namespace)
    return namespace


def extract_method(name, namespace):
    tree = ast.parse(WR.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'FlashInferB12xExperts')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.parse('from __future__ import annotations').body[0]
    exec(compile(ast.Module(body=[future, method], type_ignores=[]), str(WR), 'exec'), namespace)
    return namespace[name]


class AdmissionTests(unittest.TestCase):
    def test_wide_workspace_covers_concentrated_and_tail_routes(self):
        ns=extract(MD, {'_dynamic_task_geometry'}, dict(
            _LEVEL_TILE_M=128, _LEVEL_TILE_N=128, _DYNAMIC_SLICE_CHUNK=4,
            _align_up=lambda n, align: (n+align-1)//align*align))
        geometry=ns['_dynamic_task_geometry']
        for tokens in (4096,4097,6912,8192,16384):
            pairs=tokens*8
            histograms=([pairs]+[0]*71, [0]*72,
                        [pairs//72+(i<pairs%72) for i in range(72)],
                        [pairs-71]+[1]*71)
            tiles,slices,tasks=geometry(72,2048,pairs)
            self.assertEqual(slices,16)
            for counts in histograms:
                actual_tiles=sum(math.ceil(count/128) for count in counts)
                self.assertLessEqual(actual_tiles,tiles)
                self.assertLessEqual(actual_tiles*4,tasks)

    def test_wrapper_admits_only_exact_eager_local_geometry(self):
        fn = extract(WR, {'ep_local_prefill_eligible'}, {})['ep_local_prefill_eligible']
        capture = Mock(return_value=False)
        good=dict(enabled=True,use_ep=True,no_dummy=True,experts=72,hidden=4096,intermediate=2048,
                  tokens=6912,topk=8,activation='swigluoai_uninterleave',alpha=1.0,beta=0.0,limit=10.0,is_capturing=capture)
        self.assertTrue(fn(**good))
        capture.assert_called_once_with()
        capture.reset_mock()
        for k,v in dict(enabled=False,use_ep=False,no_dummy=False,experts=288,hidden=2048,intermediate=512,
                        tokens=4095,topk=1,activation='silu',alpha=1.7,beta=1.0,limit=None).items():
            self.assertFalse(fn(**dict(good,**{k:v})),k)
        capture.assert_not_called()
        for rows in (4096,8192,16384):self.assertTrue(fn(**dict(good,tokens=rows)))
        capture.reset_mock()
        for rows in (True,4096.0,16385):self.assertFalse(fn(**dict(good,tokens=rows)))
        capture.assert_not_called()
        capture.return_value=True
        self.assertFalse(fn(**good))
        capture.assert_called_once_with()

    def test_apply_defers_the_capture_query_until_candidate_admission(self):
        capture = Mock(return_value=False)
        ns = extract(WR, {'ep_local_prefill_eligible'}, dict(
            _EP_LOCAL_PREFILL_ENABLED=False,
            torch=SimpleNamespace(int32='i32', cuda=SimpleNamespace(is_current_stream_capturing=capture)),
            b12x_ep_zero_weight_micro_chunks=lambda *a, **kw: 0,
            b12x_ep_stock_topk_micro_chunks=lambda *a, **kw: 0,
            b12x_ep_should_compact=lambda *a, **kw: True))
        apply = extract_method('apply', ns)
        for enabled, rows in ((False, 6912), (True, 8), (True, 6912)):
            with self.subTest(enabled=enabled, rows=rows):
                ns['_EP_LOCAL_PREFILL_ENABLED'] = enabled
                capture.reset_mock()
                ids = SimpleNamespace(device='cuda', shape=(rows,8), size=lambda i: (rows,8)[i])
                weights = SimpleNamespace(dtype='f32')
                alpha = SimpleNamespace(numel=lambda:72)
                fake = SimpleNamespace(
                    w1_scale=object(), w2_scale=object(), g1_alphas=alpha, g2_alphas=alpha,
                    _fc2_input_scale=object(), w1_sf_mma=object(), w2_sf_mma=object(),
                    _use_ep=True, _kernel_num_experts=72, _ep_no_dummy=True,
                    hidden_dim=4096, intermediate_size_per_partition=2048,
                    _activation_str='swigluoai_uninterleave', _swiglu_alpha=1.,
                    _swiglu_beta=0., _swiglu_limit=10., _ensure_ep_scratch=Mock(),
                    _remap_ep_tensors=Mock(return_value=(ids,weights)),
                    _ep_zero_weight_micro=False, _ep_stock_topk_micro=False,
                    _ep_compact_enabled=True, _apply_ep_local_prefill=Mock(return_value='local'),
                    _apply_ep_compact=Mock(return_value='compact'))
                result = apply(fake, object(), SimpleNamespace(shape=(rows,4096)),
                               SimpleNamespace(size=lambda i:72), object(), weights, ids,
                               None, 288, None, None, None, None, None, None, False)
                expected = enabled and rows >= 4096
                self.assertEqual(result, 'local' if expected else 'compact')
                self.assertEqual(capture.call_count, int(expected))
                self.assertEqual(fake._remap_ep_tensors.call_args.kwargs,
                                 dict(fuse_local_prefill=expected))

    def test_dispatch_never_silently_falls_back_with_sentinel_geometry(self):
        ns=extract(MD,{'_ep_local_prefill_kernel'},dict(_GLM53_EP_PREFILL_LOCAL=False,_FORCED_BACKEND=None,_FORCE_MOE_W4A16_ENV="test_force_w4",
                   os=SimpleNamespace(environ={}),
                   select_sm120_moe_backend=Mock(return_value='dynamic'),
                   torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda:(12,1)))))
        good=dict(E=72,m=6912,k=4096,n=2048,num_topk=8,tile_m=128,activation='swigluoai_uninterleave',
                  swiglu_alpha=1.0,swiglu_beta=0.0,swiglu_limit=10.0,quant_mode='nvfp4',tiled=False)
        fn=ns['_ep_local_prefill_kernel'];self.assertIsNone(fn(**good));ns['_GLM53_EP_PREFILL_LOCAL']=True
        self.assertIsNone(fn(**dict(good,E=288,n=512)))
        for change in ({'tiled':True},{'tile_m':64}):
            with self.assertRaises(ValueError):fn(**dict(good,**change))
        module=ModuleType('test_ep.moe_dynamic_ep_local');module.MoEGatedEPLocalKernel=object()
        module.stock_contract_matches=lambda:True;ns['__package__']='test_ep'
        with patch.dict(sys.modules,{'test_ep':ModuleType('test_ep'),'test_ep.moe_dynamic_ep_local':module}):
            self.assertIs(fn(**good),module.MoEGatedEPLocalKernel)
            ns['select_sm120_moe_backend'].return_value='static'
            with self.assertRaisesRegex(ValueError,'dynamic backend selection'):fn(**good)
            ns['select_sm120_moe_backend'].return_value='dynamic'
            module.stock_contract_matches=lambda:False
            with self.assertRaisesRegex(RuntimeError,'drifted'):fn(**good)

    def test_ep_and_stock_artifacts_have_distinct_keys(self):
        fn=extract(MD,{'_dynamic_kernel_cache_key'}, {})['_dynamic_kernel_cache_key']
        args=dict(activation_precision='fp4',quant_mode='nvfp4',E=72,k=4096,n=2048,num_topk=8,mac=48,
                  mma_tiler_mn=(128,128),topk_ids_dtype='int32',input_scales_are_reciprocal=False,
                  fast_math=True,activation='swigluoai_uninterleave',swiglu_alpha=1.0,swiglu_beta=0.0,
                  swiglu_limit=10.0,share_input_across_experts=False)
        self.assertNotEqual(fn(**args),fn(**args,ep_local_prefill=True))
        self.assertEqual(fn(**args,ep_local_prefill=True)[-1],'glm53_ep_prefill_local_fp32_v2')

    def test_sp_reduction_gate_requires_opt_in_for_ep(self):
        ns=extract(MODEL,{'_prefill_sp_layer_reduction_ok'},dict(_EP_PREFILL_LOCAL=False))
        config=SimpleNamespace(tp_size=4,ep_size=1,dp_size=1,is_sequence_parallel=False,
                               skip_final_all_reduce=False,moe_backend='flashinfer_b12x')
        runner=SimpleNamespace(moe_config=config,routed_input_transform=None,routed_output_transform=None,
                               _fused_output_is_reduced=False,router=object())
        layer=SimpleNamespace(mhc=True,is_mtp_layer=False,is_sequence_parallel=False,_mlp_is_moe=True,
               self_attn=SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True)),
               mlp=SimpleNamespace(experts=runner,shared_experts=SimpleNamespace(down_proj=SimpleNamespace(reduce_results=False))))
        fn=ns['_prefill_sp_layer_reduction_ok'];self.assertTrue(fn(layer))
        config.tp_size,config.ep_size=1,4;self.assertFalse(fn(layer))
        ns['_EP_PREFILL_LOCAL']=True;self.assertTrue(fn(layer))
        for key,value in (('ep_size',8),('dp_size',2),('is_sequence_parallel',True),('skip_final_all_reduce',True)):
            old=getattr(config,key);setattr(config,key,value);self.assertFalse(fn(layer));setattr(config,key,old)
        runner.routed_input_transform=object();self.assertFalse(fn(layer))

    def test_wrapper_passes_original_token_and_output_storage(self):
        tree=ast.parse(WR.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashInferB12xExperts')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_ep_local_prefill')
        ns=dict(logger=SimpleNamespace(info_once=Mock()))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(WR),'exec'),ns)
        fused=ModuleType('flashinfer.fused_moe');fused.b12x_fused_moe=Mock()
        md=SimpleNamespace(_ep_local_prefill_kernel=Mock(return_value=object()))
        package=ModuleType('flashinfer.fused_moe.cute_dsl.blackwell_sm12x');package.moe_dispatch=md
        fake=SimpleNamespace(_activation_str='swigluoai_uninterleave',_swiglu_alpha=1.0,_swiglu_beta=0.0,
              _swiglu_limit=10.0,w1_sf_mma='s1',w2_sf_mma='s2',g1_alphas='a1',g2_alphas='a2',_fc2_input_scale='scale')
        x=SimpleNamespace(shape=(6912,4096));out=object();ids=object();weights=object()
        with patch.dict(sys.modules,{'flashinfer.fused_moe':fused,'flashinfer.fused_moe.cute_dsl.blackwell_sm12x':package}):
            result=ns['_apply_ep_local_prefill'](fake,out,x,object(),object(),ids,weights)
            self.assertIs(result,out)
            sent=fused.b12x_fused_moe.call_args.kwargs
            for k,v in (('x',x),('output',out),('token_selected_experts',ids),('token_final_scales',weights)):
                self.assertIs(sent[k],v)
            self.assertEqual((sent['top_k'],sent['num_experts']), (8,72))
            md._ep_local_prefill_kernel.return_value=None
            with self.assertRaises(RuntimeError):ns['_apply_ep_local_prefill'](fake,out,x,object(),object(),ids,weights)
            self.assertEqual(fused.b12x_fused_moe.call_count,1)


if __name__=='__main__':unittest.main()
