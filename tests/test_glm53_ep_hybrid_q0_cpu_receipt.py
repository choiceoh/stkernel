"""Pure contract checks; actual CuTe lowering remains the isolated CPU gate."""
import ast
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT/'probes/glm53_ep_tiled_compile.py'


def extracted(path, names, constants=(), namespace=None):
    body = []
    found = set()
    for node in ast.parse(path.read_text()).body:
        selected = isinstance(node, ast.FunctionDef) and node.name in names
        if selected:
            found.add(node.name)
        if isinstance(node, ast.Assign):
            selected = any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        if selected:
            body.append(copy.deepcopy(node))
    assert found == set(names)
    ns = dict(json=json, re=re) if namespace is None else namespace
    prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix]+body,type_ignores=[])),
                 str(path),'exec'), ns)
    return ns


def probe_contracts():
    return extracted(PROBE, {
        'expected_native_key','expected_native_specialization',
        'expected_dynamic_key','expected_dynamic_specialization',
        'dynamic_actual_specialization','compiler_resource_summary','validate_compile_matrix',
    }, {'BASELINE_STATIC_CASES','HYBRID_STATIC_CASES','COMPILE_GROUPS',
        'HYBRID_TAG','HYBRID_Q0_TAG'})


class HybridQ0CPUReceiptTests(unittest.TestCase):
    def setUp(self):
        self.ns = probe_contracts()

    def test_matrix_preserves_eight_controls_and_requires_distinct_ninth_candidate(self):
        ns = self.ns
        result = dict(fresh_lowerings=9,same_source_baseline_lowerings=3,
                      hybrid_lowerings=6,hybrid_q0_control_lowerings=1,
                      hybrid_q0_candidate_lowerings=1)
        for kind,e,n,cases in ns['COMPILE_GROUPS']:
            dual = kind == 'hybrid_q0_dynamic'
            result[kind+'_passes'] = []
            for name,rows,mode in cases:
                dynamic = mode == 'dynamic'
                passed = dict(arm=kind.replace('_','-')+'/'+name,
                    compiled_in_this_run=True,candidate=e==144,q0_dual_warp=dual,
                    cache_key=ns['expected_dynamic_key'](e,n,dual) if dynamic
                        else ns['expected_native_key'](rows,mode,e,n),
                    specialization=ns['expected_dynamic_specialization'](e,n,dual) if dynamic
                        else ns['expected_native_specialization'](rows,mode,e,n),
                    resources=[dict(resources='Function fixture\nREG:168 STACK:112 SHARED:0 LOCAL:0\n')])
                passed['resource_summary'] = ns['compiler_resource_summary'](
                    passed,None if dynamic else 98304,None if dynamic else 101376)
                result[kind+'_passes'].append(passed)
        ns['validate_compile_matrix'](json.loads(json.dumps(result)))
        self.assertEqual([len(result[k+'_passes']) for k,_,_,_ in ns['COMPILE_GROUPS']],
                         [2,1,4,1,1])
        control = result['hybrid_dynamic_passes'][0]
        candidate = result['hybrid_q0_dynamic_passes'][0]
        self.assertEqual(candidate['cache_key'][:-1], control['cache_key'])
        self.assertEqual(len(control['cache_key']),21)
        self.assertEqual(len(candidate['cache_key']),22)
        mutations = (
            lambda r:r.pop('hybrid_q0_dynamic_passes'),
            lambda r:r.update(fresh_lowerings=8),
            lambda r:r.update(hybrid_q0_control_lowerings=0),
            lambda r:r['hybrid_q0_dynamic_passes'][0].update(q0_dual_warp=False),
            lambda r:r['hybrid_q0_dynamic_passes'][0].update(q0_dual_warp=1),
            lambda r:r['hybrid_q0_dynamic_passes'][0]['cache_key'].pop(),
            lambda r:r['hybrid_dynamic_passes'][0]['cache_key'].append(ns['HYBRID_Q0_TAG']),
            lambda r:r['hybrid_q0_dynamic_passes'][0]['specialization'].update(q0_dual_warp=False),
            lambda r:r['hybrid_q0_dynamic_passes'][0]['specialization'].update(q0_dual_warp=1),
            lambda r:r['hybrid_q0_dynamic_passes'][0]['specialization'].update(q0_batch_tokens=8),
            lambda r:r['hybrid_q0_dynamic_passes'][0]['resource_summary'].update(total_shared_bytes=0),
        )
        for mutation in mutations:
            changed=copy.deepcopy(result);mutation(changed)
            with self.assertRaises((AssertionError,KeyError)):
                ns['validate_compile_matrix'](changed)

    def test_actual_observer_requires_bool_pointer_abi_and_lowered_geometry(self):
        ns = self.ns
        cutlass = SimpleNamespace(Float32=object())
        cls = type('MoEGatedEPLocalKernelSF6', (), {})
        for e,n,dual in ((72,2048,False),(144,1024,False),(144,1024,True)):
            kernel=cls();kernel.q0_dual_warp=dual;kernel.reform_sf_pack=True
            args=[None]*36;args[34]=48
            expected=ns['expected_dynamic_specialization'](e,n,dual)
            for idx,shape in expected['tensor_shapes'].items():
                args[int(idx)]=SimpleNamespace(shape=tuple(shape))
            # Deliberately no element_type: this is the real Pointer API boundary.
            args[25]=SimpleNamespace(dtype=cutlass.Float32)
            actual=lambda lowered: ns['dynamic_actual_specialization'](
                kernel,args,e,n,dual,cutlass,lowered=lowered)
            self.assertEqual(actual(False),expected)
            with self.assertRaises(AttributeError):actual(True)
            kernel.num_mma_warps=8;kernel.threads_per_cta=288
            kernel.tile_shape_mnk=(128,128,128)
            self.assertEqual(actual(True),expected)
            for bad in (not dual,1,None):
                kernel.q0_dual_warp=bad
                with self.assertRaises(AssertionError):actual(False)
            kernel.q0_dual_warp=dual
            args[25].dtype=object()
            with self.assertRaises(AssertionError):actual(False)
            args[25].dtype=cutlass.Float32
            kernel.threads_per_cta=256
            with self.assertRaises(AssertionError):actual(True)

    def test_expected_cache_matches_actual_dispatch_factory(self):
        factory=extracted(ROOT/'overlay/modules/glm53_moe/moe_dispatch.py',
                          {'_dynamic_kernel_cache_key'})['_dynamic_kernel_cache_key']
        class DType:
            def __str__(self):return 'torch.int32'
        for e,n,dual in ((72,2048,False),(144,1024,False),(144,1024,True)):
            kwargs=dict(activation_precision='fp4',quant_mode='nvfp4',E=e,k=4096,n=n,
                num_topk=8,mac=48,mma_tiler_mn=(128,128),topk_ids_dtype=DType(),
                input_scales_are_reciprocal=False,fast_math=True,
                activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,
                swiglu_limit=10.,share_input_across_experts=False,ep_local_prefill=True,
                tiled=True,reform_sf_pack=True,ep_hybrid_q0_dual_warp=dual)
            actual=json.loads(json.dumps(factory(**kwargs),default=str))
            self.assertEqual(actual,self.ns['expected_dynamic_key'](e,n,dual))
            if e==72:
                kwargs['ep_hybrid_q0_dual_warp']=True
                with self.assertRaises(ValueError):factory(**kwargs)

    def test_compile_observation_surrounds_actual_lowering_and_forwards_explicit_override(self):
        tree=ast.parse(PROBE.read_text())
        compile_body=next(n for n in tree.body
                          if isinstance(n,ast.FunctionDef) and n.name=='compile_candidate')
        observer=next(n for n in ast.walk(compile_body)
                      if isinstance(n,ast.FunctionDef) and n.name=='observe_compile')
        checks=sorted((n for n in ast.walk(observer) if isinstance(n,ast.Call)
                       and ast.unparse(n.func)=='dynamic_actual_specialization'),key=lambda n:n.lineno)
        self.assertEqual([next(k.value.value for k in c.keywords if k.arg=='lowered')
                          for c in checks],[False,True])
        real=next(n for n in ast.walk(observer) if isinstance(n,ast.Call)
                  and ast.unparse(n.func)=='real_compile')
        self.assertLess(checks[0].lineno,real.lineno)
        self.assertLess(real.lineno,checks[1].lineno)
        dispatch=next(n for n in ast.walk(compile_body) if isinstance(n,ast.Call)
                      and ast.unparse(n.func)=='md._get_dynamic_kernel')
        selector=next(k.value for k in dispatch.keywords
                      if k.arg=='_ep_hybrid_q0_dual_warp_override')
        self.assertEqual(ast.unparse(selector),'q0_dual_warp')
        selections=[ast.unparse(n.value) for n in ast.walk(compile_body)
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)
                    and t.id=='q0_dual_warp' for t in n.targets)]
        self.assertEqual(selections,["kind == 'hybrid_q0_dynamic'"])


if __name__=='__main__':unittest.main()
