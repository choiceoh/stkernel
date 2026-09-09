"""CPU-only owner, route and sticky-completion contracts for the TP canary."""
import ast
from collections import Counter
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get("GLM53_TP_Q0_SELFTEST_REPO", Path(__file__).resolve().parents[1]))
SOURCE = Path(os.environ.get("GLM53_TP_Q0_SELFTEST_SOURCE", ROOT / "overlay/modules/glm53_moe/glm53_tp_sf6_q0_selftest.py"))
PACKAGE = "_tp_q0_selftest_cpu"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = []
    sys.modules[PACKAGE] = package
for name, source in (("glm53_ep_local_selftest", ROOT/"overlay/modules/glm53_moe/glm53_ep_local_selftest.py"),
                     ("glm53_tp_sf6_q0_selftest", SOURCE)):
    spec = importlib.util.spec_from_file_location(PACKAGE+"."+name, source)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
mod = sys.modules[PACKAGE+".glm53_tp_sf6_q0_selftest"]


def owner():
    views = SimpleNamespace(tiled=True,packed_only=True,reform_scales=SimpleNamespace(enabled=True))
    experts = SimpleNamespace(_use_ep=False,_sf6_finalized=True,_sf6_weight_views=views,
        global_num_experts=288,num_local_experts=288,hidden_dim=4096,
        intermediate_size_per_partition=512,topk=8)
    layer = SimpleNamespace(w13_weight=SimpleNamespace(shape=(288,1024,2048)),
                            w2_weight=SimpleNamespace(shape=(288,4096,256)))
    return experts,layer


def keys():
    baseline = ("dynamic","fp4","nvfp4",288,4096,512,8,48,(128,128),"torch.int32",
                False,True,"swigluoai_uninterleave",1.,0.,10.,False,True,"sf6_direct_prefill_v1")
    return baseline,baseline+("glm53_tp_sf6_q0_v1",)


class CanaryTests(unittest.TestCase):
    def setUp(self):
        mod._STATES.clear()

    def test_owner_selection_accepts_already_finalized_and_rejects_other_geometry(self):
        experts,layer=owner()
        self.assertTrue(mod.tp_sf6_q0_selftest_eligible(experts,layer))
        self.assertTrue(mod.tp_sf6_q0_selftest_eligible(experts,layer))
        for name,value in (("_use_ep",True),("_sf6_finalized",False),("num_local_experts",72),
                           ("intermediate_size_per_partition",2048),("_sf6_weight_views",None)):
            old=getattr(experts,name);setattr(experts,name,value)
            self.assertFalse(mod.tp_sf6_q0_selftest_eligible(experts,layer))
            setattr(experts,name,old)
        experts._sf6_weight_views.reform_scales.enabled=False
        self.assertFalse(mod.tp_sf6_q0_selftest_eligible(experts,layer))

    def test_all_fixture_ids_are_valid_and_duplicates_are_retained(self):
        self.assertEqual([row[1] for row in mod.CASES],[4096,6912,4097,8192])
        for _,rows,kind in mod.CASES:
            first=mod._route_rows(rows,kind)
            second=mod._route_rows(rows,kind,True)
            self.assertNotEqual(first,second)
            self.assertEqual(len(first),rows)
            self.assertTrue(all(len(route)==8 and all(0<=e<288 for e in route) for route in first+second))
            if kind=="duplicate":self.assertTrue(all(route==[5]*8 for route in first))
        with self.assertRaises(ValueError):mod._route_rows(4,"remote")

    def test_route_oracle_keeps_zero_signed_zero_and_duplicate_multiplicity(self):
        ids=[[5]*8,[5,0,287,0,5,287,1,1]]
        weights=[[0.,-0.,.25,.125,.25,.125,0.,.125],[.125]*8]
        grouped=[[] for _ in range(288)]
        for token,(route,values) in enumerate(zip(ids,weights)):
            for expert,weight in zip(route,values):grouped[expert].append((token,weight))
        counts=list(map(len,grouped));bases=[0]
        for count in counts:bases.append(bases[-1]+(count+127)//128)
        tokens=[-77]*(bases[-1]*128);stored=[float("nan")]*len(tokens)
        for e,rows in enumerate(grouped):
            for j,(token,weight) in enumerate(reversed(rows)):
                tokens[bases[e]*128+j]=token;stored[bases[e]*128+j]=weight
        sample=mod._route_metadata(counts,bases,tokens,stored,ids,weights)
        self.assertEqual(len(sample),16)
        first_zero=next(p for (_,_,bits),p in sample if bits==mod._weight_bits(-0.))
        stored[first_zero]=0.
        with self.assertRaisesRegex(AssertionError,"multiset"):
            mod._route_metadata(counts,bases,tokens,stored,ids,weights)

    def test_cache_pair_accepts_19_field_stock_key_and_rejects_missing_baseline(self):
        baseline,candidate=keys()
        md=SimpleNamespace(_DYNAMIC_KERNEL_CACHE={baseline:object(),candidate:object()})
        self.assertEqual(mod._selected_keys(md),[dict(baseline=repr(baseline),candidate=repr(candidate))])
        del md._DYNAMIC_KERNEL_CACHE[baseline]
        with self.assertRaisesRegex(AssertionError,"differ beyond"):
            mod._selected_keys(md)
        md._DYNAMIC_KERNEL_CACHE={baseline:object()}
        with self.assertRaisesRegex(AssertionError,"absent"):
            mod._selected_keys(md)

    def test_inference_tensor_state_does_not_require_version_counter(self):
        class Tensor:
            shape=(288,);dtype="float32";device="cuda:0"
            @property
            def _version(self):raise RuntimeError("Inference tensors do not track version counter.")
            def data_ptr(self):return 123
            def stride(self):return (1,)
        state=mod._tensor_state(Tensor())
        self.assertEqual(state[2],"inference-no-version-counter")

    def test_actual_schedule_uses_existing_launcher_graphs_and_unchanged_threshold_helpers(self):
        tree=ast.parse(SOURCE.read_text())
        case=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_case")
        text=ast.unparse(case)
        self.assertIn('_tp_sf6_q0_override=candidate',text)
        self.assertIn('C1-eager',text);self.assertIn('C2-graph-current',text);self.assertIn('C3-graph-side',text)
        self.assertIn('with torch.cuda.graph(graph, stream=side)',text)
        self.assertLess(text.index('b1 = eager(False)'),text.index('for label in'))
        self.assertIn('b2, b3 = (eager(False), eager(False))',text)
        self.assertIn('compare(candidate, b1, b2, failure_context=context)',text)
        imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='glm53_ep_local_selftest']
        self.assertEqual(len(imports),1)
        self.assertIn('compare',[a.name for a in imports[0].names])
        self.assertNotIn('ensure_ep_local_selftest',SOURCE.read_text())
        self.assertNotIn('_weights(',SOURCE.read_text())

    def _run(self,*,case_failure=False,cleanup_failure=False,weight_mutation=False):
        experts,layer=owner();calls=[];state={"owner":1};identity_calls=[]
        class Rng:
            def clone(self):return self
        cpu_rng,cuda_rng=Rng(),Rng()
        torch=SimpleNamespace(get_rng_state=lambda:cpu_rng,equal=lambda a,b:a is b,
            cuda=SimpleNamespace(get_rng_state=lambda _:cuda_rng,synchronize=lambda _:None))
        baseline,candidate=keys()
        md=SimpleNamespace(_DYNAMIC_KERNEL_CACHE={baseline:None,candidate:None},
                           allocate_sm120_dynamic_workspace=lambda **kw:object())
        def case(*args):
            calls.append(args[-2][0]);args[-1].update(verdict="PASS",phase="complete")
            if case_failure:raise AssertionError("original numerical FAIL")
        def restore(*args):
            if cleanup_failure:raise RuntimeError("cleanup also failed")
            return {}
        def identity(value):
            identity_calls.append(value)
            return {"sha256":str(len(identity_calls)) if weight_mutation else "same"}
        patches=dict(_runtime=lambda *a:(torch,md,"cuda:0",{"source":{}}),
                     _caller_state=lambda *a:dict(state),_backings=lambda *a:{"w13":object()},
                     _tensor_identity=identity,_cache_snapshot=lambda _: {},
                     _restore_scratch_caches=restore,_memory=lambda *a:{},_case=case)
        return experts,layer,calls,patch.multiple(mod,**patches)

    def test_success_and_cached_success_run_once_and_preserve_owner(self):
        experts,layer,calls,patches=self._run()
        with patches,contextlib.redirect_stdout(io.StringIO()) as output:
            first=mod.ensure_tp_sf6_q0_selftest(experts,layer=layer)
            second=mod.ensure_tp_sf6_q0_selftest(experts,layer=layer)
        self.assertIs(first,second)
        self.assertEqual(len(calls),4)
        self.assertTrue(first['caller_preserved'])
        self.assertEqual(first['phase'],'complete')
        self.assertEqual(output.getvalue().count('[tp-sf6-q0-selftest] PASS '),1)

    def test_first_failure_survives_cleanup_failure_and_blocks_repeat(self):
        experts,layer,calls,patches=self._run(case_failure=True,cleanup_failure=True)
        with patches,contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(RuntimeError) as caught:
                mod.ensure_tp_sf6_q0_selftest(experts,layer=layer)
            self.assertIn('original numerical FAIL',str(caught.exception.__cause__))
            with self.assertRaisesRegex(RuntimeError,'previously failed'):
                mod.ensure_tp_sf6_q0_selftest(experts,layer=layer)
        self.assertEqual(len(calls),1)
        self.assertNotIn('[tp-sf6-q0-selftest] PASS ',output.getvalue())
        receipt=next(iter(mod._STATES.values()))
        self.assertIn('cleanup also failed',receipt['cleanup_error'])
        self.assertIn('original numerical FAIL',receipt['error'])

    def test_weight_mutation_prevents_complete_marker(self):
        experts,layer,calls,patches=self._run(weight_mutation=True)
        with patches,contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(RuntimeError) as caught:
                mod.ensure_tp_sf6_q0_selftest(experts,layer=layer)
        self.assertIn('actual weight/scale',str(caught.exception.__cause__))
        self.assertNotIn('[tp-sf6-q0-selftest] PASS ',output.getvalue())


if __name__ == '__main__':
    unittest.main()
