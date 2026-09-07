"""CPU evidence contracts; tensor tests run in the pinned runtime, without GPU."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probes'))
import glm53_prefill_observer as observer
import glm53_prefill_trace as trace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import prefill_observation as client

try:
    import torch
except ImportError:
    torch = None


class EvidenceTests(unittest.TestCase):
    def test_rpc_failure_and_request_failure_always_attempt_both_cleanups(self):
        for failed in ('begin','start','request','stop'):
            calls=[]
            class API:
                def post(self,path,payload=None):
                    op=(payload or {}).get('op',path)
                    calls.append(op)
                    if op==failed or (op=='/start_profile' and failed=='start') or (op=='/stop_profile' and failed=='stop'):
                        raise RuntimeError('injected failure')
                    if op=='status':return {'ranks':[dict(rank=r,active=False,source_sha256='a'*64) for r in range(4)]}
                    if op=='begin':return {'ranks':[dict(rank=r,active=True,source_sha256='a'*64,request_id='cpu',mode='profile') for r in range(4)]}
                    if op=='end':return {'ranks':[]} # incomplete request cannot receive a completion verdict
                    return {'status':200}
            def request():
                calls.append('request')
                if failed=='request':raise RuntimeError('request failed')
                return {'wire_sha256':'b'*64}
            result=client.collect_request(api=API(),mode='profile',request_id='cpu',source_sha256='a'*64,request=request)
            self.assertFalse(result['complete']);self.assertTrue(result['errors'])
            self.assertIn('end',calls);self.assertEqual(calls[-1],'status')
            if failed!='begin':self.assertIn('/stop_profile',calls)
            if failed in ('begin','start'):self.assertNotIn('request',calls)

    def test_private_endpoint_and_idle_source_guard(self):
        with self.assertRaises(ValueError):client.PrivateObserverAPI('http://127.0.0.1:8000')
        for ranks in ([],[dict(rank=r,active=False,source_sha256='b'*64) for r in range(4)]):
            with self.assertRaises(ValueError):client.idle_observers({'ranks':ranks},'a'*64)

    def test_profile_and_routes_collect_separately_with_complete_rank_cleanup(self):
        for mode in ('profile','routes'):
            calls=[]
            class API:
                def post(self,path,payload=None):
                    op=(payload or {}).get('op',path);calls.append(op)
                    if op=='status':return {'ranks':[dict(rank=r,active=False,source_sha256='a'*64) for r in range(4)]}
                    if op=='begin':return {'ranks':[dict(rank=r,active=True,source_sha256='a'*64,request_id='cpu',mode=mode) for r in range(4)]}
                    if op=='end':
                        reports=[]
                        for rank in range(4):
                            layer='model.layers.2.experts.w13_weight'
                            record=dict(call=0,rank=rank,layer=layer,rows=2,request_id='cpu',mode=mode)
                            if mode=='routes':record.update(zero_weight_slots=0,**observer.histogram(list(range(16)),2))
                            reports.append(dict(rank=rank,request_id='cpu',mode=mode,source_sha256='a'*64,complete=True,
                                hook_restored=True,errors=[],performance_acceptance=False,numerical_acceptance=False,
                                records=[record],layers=[layer],moe_forward_groups=observer.forward_groups([record],[layer])))
                        return {'ranks':reports}
                    return {'status':200}
            result=client.collect_request(api=API(),mode=mode,request_id='cpu',source_sha256='a'*64,request=lambda:{'wire_sha256':'b'*64})
            self.assertTrue(result['complete']);self.assertFalse(result['performance_acceptance'])
            self.assertEqual('/start_profile' in calls,mode=='profile')
            self.assertEqual('/stop_profile' in calls,mode=='profile')
            self.assertEqual(calls[-2:],['end','status'])

    def test_all_rank_source_and_route_accounting_are_required(self):
        layer='model.layers.2.experts.w13_weight'
        reports=[]
        for rank in range(4):
            record=dict(call=0,layer=layer,rows=2,rank=rank,request_id='cpu',mode='routes',zero_weight_slots=0,
                        **observer.histogram(list(range(16)),2))
            reports.append(dict(rank=rank,request_id='cpu',mode='routes',source_sha256='a'*64,complete=True,
                                hook_restored=True,errors=[],performance_acceptance=False,numerical_acceptance=False,
                                records=[record],layers=[layer],moe_forward_groups=observer.forward_groups([record],[layer])))
        check=lambda rows:observer.validate_ranks(rows,request_id='cpu',mode='routes',source_sha256='a'*64)
        self.assertEqual(check(list(reversed(reports))),reports)
        wrong_hash=copy.deepcopy(reports);wrong_hash[1]['source_sha256']='b'*64
        wrong_count=copy.deepcopy(reports);wrong_count[2]['records'][0]['expert_counts'][0]=0
        wrong_cleanup=copy.deepcopy(reports);wrong_cleanup[3]['hook_restored']=False
        wrong_ratio=copy.deepcopy(reports);wrong_ratio[1]['records'][0]['padded_work_reduction_pct']=float('nan')
        for bad in (reports[:3],reports[:3]+[reports[0]],wrong_hash,wrong_count,wrong_cleanup,wrong_ratio):
            with self.assertRaises(ValueError):check(bad)

    def test_real_routing_budget_and_invalid_slots(self):
        result = observer.histogram(list(range(8)) * 8, 8)
        self.assertEqual(result['routed_slots'], 64)
        self.assertEqual(result['padded_rows'], {'64':512, '128':1024})
        self.assertEqual(result['padded_work_reduction_pct'], 50)
        for ids in ([-1] * 64, [288] * 64, [True] * 64, [0] * 63):
            with self.assertRaises(ValueError): observer.histogram(ids, 8)

    def test_forward_coverage_uses_observed_rows_and_rejects_partial_layers(self):
        layers = ['model.layers.2.experts.w13_weight', 'model.layers.3.experts.w13_weight']
        records = [dict(layer=layer, rows=rows) for rows in (6912, 137) for layer in layers]
        self.assertEqual([g['executed_moe_rows'] for g in observer.forward_groups(records,layers)], [6912,137])
        for bad in (records[:-1], list(reversed(records[:2])), [records[0],records[3]]):
            with self.assertRaises(ValueError): observer.forward_groups(bad,layers)

    def test_trace_freshness_rejects_overwrite_and_ambiguous_new_files(self):
        state = dict(device=1,inode=10,mtime_ns=90,size=100,regular=True,symlink=False)
        old = {'old.pt.trace.json.gz':state}
        new = {**old,'new.pt.trace.json.gz':dict(state,inode=11)}
        self.assertEqual(trace.fresh_trace(old,new),'new.pt.trace.json.gz')
        for bad in ({**new,'second.pt.trace.json.gz':state},
                    {**new,'old.pt.trace.json.gz':dict(state,size=101)},
                    {**old,'new.pt.trace.json.gz':dict(state,symlink=True)}, old):
            with self.assertRaises(ValueError): trace.fresh_trace(old,bad)

    def test_trace_overlap_missing_calls_and_foreign_rank(self):
        record = dict(call=0,layer='model.layers.2.experts.w13_weight',rows=6912,
                      rank=2,request_id='request1',mode='profile')
        obs = dict(mode='profile',complete=True,hook_restored=True,errors=[],records=[record],
                   moe_forward_groups=[dict(executed_moe_rows=6912)],rank=2,request_id='request1',source_sha256='a'*64)
        events = [dict(ph='X',name=trace.PREFIX+json.dumps(record),cat='user_annotation',ts=0,dur=1),
                  dict(ph='X',name='ncclKernel',cat='kernel',ts=0,dur=10,args={'device':0}),
                  dict(ph='X',name='moe_kernel',cat='kernel',ts=5,dur=10,args={'device':0}),
                  dict(ph='X',name='unknown_kernel',cat='kernel',ts=20,dur=5,args={'device':0})]
        out = trace.analyze(events,obs)
        self.assertEqual((out['kernel_and_memop_sum_us'],out['gpu_interval_union_us'],out['gap_inside_gpu_span_us']), (25,20,5))
        self.assertEqual(out['categories']['unknown']['calls'],1)
        self.assertFalse(out['performance_acceptance'])
        for bad in (events[1:],events+[events[0]],events+[dict(events[-1],args={'device':1})]):
            with self.assertRaises(ValueError): trace.analyze(bad,obs)
        foreign = copy.deepcopy(obs);foreign['records'][0]['rank']=1
        with self.assertRaises(ValueError): trace.analyze(events,foreign)


@unittest.skipIf(torch is None, 'real CPU tensor contracts require pinned runtime torch')
class HookTests(unittest.TestCase):
    def setUp(self):
        class Wrapper:
            num_experts = num_local_experts = 288
            def run(self,x,w1_weight,token_selected_experts,token_final_scales,*,out=None):
                self.seen = (x,token_selected_experts,token_final_scales)
                return x if out is None else out
        self.cls=Wrapper;self.wrapper=Wrapper();self.original=Wrapper.run
        self.capturing=False;self.annotations=[]
        def annotate(name):self.annotations.append(name);return nullcontext()
        self.api=SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:self.capturing),
                                 int64=torch.int64,profiler=SimpleNamespace(record_function=annotate))
        self.weight=torch.ones(1)
        self.x=torch.zeros(2,4096,dtype=torch.bfloat16)
        self.ids=torch.arange(16).reshape(2,8).to(torch.int32)
        self.scales=torch.ones(2,8)
        self.name='model.layers.2.experts.w13_weight'

    def session(self,mode,**kwargs):
        return observer.Observation(torch=self.api,wrapper_class=self.cls,weights={self.weight.data_ptr():self.name},
                                    rank=0,request_id='cpu_request',mode=mode,**kwargs)

    def call(self):return self.wrapper.run(self.x,self.weight,self.ids,self.scales)

    def test_routes_preserve_inputs_outputs_and_uninstall_exact_hook(self):
        session=self.session('routes');session.install()
        self.assertIs(self.call(),self.x)
        self.assertIs(self.wrapper.seen[1],self.ids)
        self.ids.fill_(7)  # Saved counts must not alias the route storage.
        result=session.finish()
        self.assertTrue(result['complete']);self.assertIs(self.cls.run,self.original)
        self.assertEqual(result['records'][0]['expert_counts'][:16],[1]*16)
        self.assertEqual(self.annotations,[])
        with self.assertRaises(RuntimeError):session.finish()

    def test_profile_has_no_histogram_copy_and_records_exact_call(self):
        session=self.session('profile');session.install()
        self.assertIs(self.call(),self.x);result=session.finish()
        self.assertTrue(result['complete'])
        self.assertNotIn('expert_counts',result['records'][0])
        self.assertEqual(json.loads(self.annotations[0][len(trace.PREFIX):]),result['records'][0])

    def test_budget_capture_and_unknown_layer_invalidate_without_changing_output(self):
        for cause in ('budget','capture','weight'):
            session=self.session('routes',limit=1);session.install()
            if cause=='budget':self.call()
            if cause=='capture':self.capturing=True
            if cause=='weight':session.weights={}
            self.assertIs(self.call(),self.x)
            result=session.finish();self.capturing=False
            self.assertFalse(result['complete']);self.assertTrue(result['errors'])
            self.assertIs(self.cls.run,self.original)


if __name__ == '__main__': unittest.main()
