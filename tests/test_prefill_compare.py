import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('prefill_compare', Path(__file__).resolve().parents[1]/'bench/prefill_compare.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    """Synthetic contract data only; never a GPU/serving evidence artifact."""
    arms = []
    knob = 'VLLM_GLM53_PREFILL_MOE_OVERLAP'
    for a, enabled in enumerate((False, True, False)):
        nodes = {n: dict(id=f'id-{a}-{n}', started_at=f'time-{a}', image='sha256:'+'a'*64,
            args={'host':'127.0.0.1','port':'18000','max-model-len':'262144','num-gpu-blocks-override':'415',
                  'gpu-memory-utilization':'0.6229','max-num-batched-tokens':'8192',
                  'max-num-seqs':'4','max-cudagraph-capture-size':'32'},
            env={knob:str(int(enabled)),'VLLM_OTHER':'0'}, mounts={'source':'hash'},
            manifest_sha='b'*64, model={'revision':'model'}, hardware={'uuid':n}) for n in m.NODES}
        arm = dict(revision='c'*40, knob=knob, enabled=enabled, before=nodes,
                   after=copy.deepcopy(nodes), launch_proof=dict.fromkeys(m.NODES, enabled))
        for phase in ('priming', 'measured'):
            records, fresh = [], []
            for q, (ctx, question) in enumerate(m.REQUESTS):
                digest = hashlib.sha256(str((ctx, question)).encode()).hexdigest()
                timing = dict(ctx=ctx, question=question, request_sha256=digest,
                    prompt_tokens=ctx+q, min_tokens=0, max_tokens=400, seed=None,
                    ttft_s=(.8 if enabled else 1.)*(q+1))
                records.append(timing)
                salt=f'{a}-{phase}-{q}'
                counters=dict(prefix_hits=0, traffic=dict(running=0, waiting=0))
                fresh.append(dict(cache_salt=salt, before=copy.deepcopy(counters),
                    after=copy.deepcopy(counters), issues=[], unsalted_sha256=digest,
                    wire_sha256=hashlib.sha256(salt.encode()).hexdigest(),
                    prompt_tokens=timing['prompt_tokens'],ttft_s=timing['ttft_s']))
            name=f'arm-{a}-{phase}'
            record=dict(name=name, boot_id=nodes['10.10.10.2']['id']+'|'+nodes['10.10.10.2']['started_at'],
                git='c'*7, overlay='b'*12, quality=dict(ok=9,total=9), korean=dict(dirty=0,n=5),
                traffic=dict(issues=[]), workload=dict(ctx=m.CONTEXTS,require_exclusive=True,fixed_decode_tokens=0),
                requests=records,harness=40,endpoint={'completion':'loopback'},doc_lang='ko',thinking=True,
                cold_compile=phase=='priming')
            arm[phase]=dict(record=record,fresh=dict(name=name,schema=1,requests=fresh))
        arms.append(arm)
    return arms


class CompareTests(unittest.TestCase):
    def reject(self, arms):
        result=m.compare(arms)
        self.assertTrue(result['issues'])
        self.assertEqual(result['comparison'], [])

    def test_matched_fresh_bracket_reports_latency_and_inverse_rate_separately(self):
        result=m.compare(fixture())
        self.assertFalse(result['issues'])
        self.assertEqual(len(result['comparison']), 3)
        for row in result['comparison']:
            self.assertAlmostEqual(row['latency_reduction_pct'],20)
            self.assertAlmostEqual(row['throughput_gain_pct'],25)

    def test_cache_salt_reuse_and_prefix_hits_invalidate(self):
        for change in ('salt','hits','missing'):
            arms=fixture()
            fresh=arms[1]['measured']['fresh']['requests'][0]
            if change=='salt':fresh['cache_salt']=arms[0]['priming']['fresh']['requests'][0]['cache_salt']
            elif change=='hits':fresh['after']['prefix_hits']=1
            else:del fresh['before']['prefix_hits']
            with self.subTest(change=change):self.reject(arms)

    def test_different_build_node_setting_and_midrun_source_invalidate(self):
        for change in ('revision','image','env','after','proof'):
            arms=fixture()
            arm=arms[1]
            if change=='revision':arm['revision']='d'*40
            elif change=='proof':arm['launch_proof']['10.10.10.3']=False
            else:
                node=arm['after' if change=='after' else 'before']['10.10.10.3']
                if change=='env':node['env']['VLLM_OTHER']='1'
                elif change=='after':node['mounts']['source']='different'
                else:node['image']='sha256:'+'d'*64
            with self.subTest(change=change):self.reject(arms)

    def test_missing_priming_and_compile_cold_measurement_invalidate(self):
        arms=fixture();del arms[1]['priming'];self.reject(arms)
        arms=fixture();arms[1]['measured']['record']['cold_compile']=True;self.reject(arms)

    def test_different_request_tokens_and_quality_invalidate(self):
        for change in ('hash','tokens','quality','traffic'):
            arms=fixture();record=arms[1]['measured']['record']
            if change=='hash':record['requests'][0]['request_sha256']='d'*64
            elif change=='tokens':record['requests'][0]['prompt_tokens']+=1
            elif change=='quality':record['quality']['ok']=8
            else:record['traffic']['issues'].append('external request')
            with self.subTest(change=change):self.reject(arms)

    def test_invalid_times_cannot_become_speedups(self):
        for value in (0,-1,None,float('nan'),float('inf')):
            arms=fixture();arms[1]['measured']['record']['requests'][0]['ttft_s']=value
            with self.subTest(value=value):self.reject(arms)

    def test_consistently_mixed_worker_build_and_wrong_capacity_invalidate(self):
        for change in ('manifest_sha','capacity'):
            arms=fixture()
            for arm in arms:
                for phase in ('before','after'):
                    node=arm[phase]['10.10.10.3']
                    if change=='manifest_sha':node['manifest_sha']='d'*64
                    else:node['args']['num-gpu-blocks-override']='1056'
            with self.subTest(change=change):self.reject(arms)


if __name__ == '__main__':
    unittest.main()
