"""Private memory-pair configuration, evidence and failure contracts."""
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'bench'), str(ROOT/'probes')]
import glm53_cpu_vote_host as host
import glm53_cpu_vote_memory as memory
import glm53_cpu_vote_memory_pair as pair
from test_glm53_observation_runner import original


def incoming():
    value = original()
    value['Config']['Env'].append('VLLM_GLM53_RANK_CACHE=/cache/glm53-ranks')
    value['HostConfig']['Binds'] += ['/original/'+n+':'+host.PREFIX+n+':ro' for n in host.MODULES]
    return value


def receipt(policy='0'):
    values = dict(pid=20, start_ticks='5', process_kib={'Pss':100}, host_kib={'MemAvailable':200},
                  vmpin_kib=10, cuda_initialized=True,
                  zero_mappings=memory.mapping_summary('10-20 rw-s 0 00:00 0 /dev/zero (deleted)\nSize: 8 kB\nRss: 8 kB\nPss: 4 kB'))
    ranks = [dict(rank=i, active=False, policy=policy, source_sha256='probe', rank_cache_sha256='cache',
                  memory=copy.deepcopy(values)) for i in range(4)]
    api = dict(active=False, source_sha256='probe', memory=copy.deepcopy(values))
    api['memory']['cuda_initialized'] = False
    return dict(ranks=ranks, api=api)


class ConfigurationTests(unittest.TestCase):
    def test_only_vote_differs_between_clones_and_original_is_untouched(self):
        before = incoming(); saved = copy.deepcopy(before)
        kwargs = dict(directory='/out', source=ROOT, session='cpu')
        a = host.clone_payload(before, policy='0', **kwargs)
        b = host.clone_payload(before, policy='1', **kwargs)
        self.assertEqual(before, saved)
        aenv = dict(e.split('=', 1) for e in a.pop('Env'))
        benv = dict(e.split('=', 1) for e in b.pop('Env'))
        self.assertEqual(a, b)
        self.assertEqual(aenv.pop(host.KNOB), '0'); self.assertEqual(benv.pop(host.KNOB), '1')
        self.assertEqual(aenv, benv)
        self.assertEqual(aenv['SECRET'], 'keep-private')
        self.assertEqual({k:aenv[k] for k in host.FIXED_ENV}, host.FIXED_ENV)
        command = base64.b64decode(a['Cmd'][1].split()[1]).decode()
        self.assertIn('glm53_cpu_vote_memory.WorkerExtension', command)
        self.assertIn('--max-model-len 1048576', command)
        self.assertIn('--host 127.0.0.1 --port 18000', command)
        for name in host.MODULES:
            self.assertIn(str(ROOT/'build/glm53'/name)+':'+host.PREFIX+name+':ro', a['HostConfig']['Binds'])

    def test_missing_duplicate_writable_bind_and_unsupported_policy_fail(self):
        variants = []
        c = incoming(); c['HostConfig']['Binds'].pop(); variants.append(c)
        c = incoming(); c['HostConfig']['Binds'].append(c['HostConfig']['Binds'][-1]); variants.append(c)
        c = incoming(); c['HostConfig']['Binds'][-1] = c['HostConfig']['Binds'][-1][:-2]+'rw'; variants.append(c)
        for entry in ('VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1', host.KNOB+'=1',
                      'VLLM_GLM53_RANK_CACHE=0', 'NCCL_DEBUG_FILE=/tmp/shared'):
            c = incoming(); c['Config']['Env'].append(entry); variants.append(c)
        for c in variants:
            with self.assertRaises(ValueError):
                host.clone_payload(c, directory='/out', source=ROOT, session='cpu', policy='1')
        for policy in ('2', 'true', 1, None):
            with self.assertRaises(ValueError):
                host.clone_payload(incoming(), directory='/out', source=ROOT, session='cpu', policy=policy)

    def test_paused_original_requires_exact_identity_before_create(self):
        c = incoming(); expected = host.original_identity(c); c['State']['Running'] = False
        args = dict(action='prepare', name='glm53-observe-cpu', session='cpu', directory='/out',
                    source=str(ROOT), original='glm53', policy='1')
        for identity in (None, {**expected, 'id':'replaced'}):
            with patch.object(host.host, 'inspect', side_effect=[None, c]), patch.object(host.host, 'create') as create:
                with self.assertRaises(RuntimeError): host.dispatch(**args, expected_original=identity)
                create.assert_not_called()
        with patch.object(host.host, 'inspect', side_effect=[None, c]), \
                patch.object(host.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=127*2**30)), \
                patch.object(host.host, 'create') as create:
            with self.assertRaisesRegex(RuntimeError, '128 GiB'): host.dispatch(**args, expected_original=expected)
            create.assert_not_called()


class EvidenceTests(unittest.TestCase):
    def test_zero_mapping_totals_exclude_other_anonymous_mappings(self):
        text = '10-20 rw-s 0 00:00 0 /dev/zero (deleted)\nSize: 8 kB\nRss: 8 kB\nPss: 4 kB\n'
        text += '20-30 rw-p 0 00:00 0\nSize: 16 kB\nRss: 16 kB\nPss: 16 kB\n'
        row = memory.mapping_summary(text)
        self.assertEqual(row, dict(count=1, size_histogram_kib={'8':1}, totals_kib={'Size':8,'Rss':8,'Pss':4}))
        with self.assertRaises(ValueError): memory.mapping_summary(text.replace('Pss: 4 kB', ''))

    def test_rank_policy_source_and_memory_completeness_are_required(self):
        memory.validate(receipt(), 'probe', 'cache', '0')
        for mutate in (
                lambda r:r['ranks'].pop(),
                lambda r:r['ranks'][2].update(rank=0),
                lambda r:r['ranks'][0].update(policy='1'),
                lambda r:r['ranks'][0].update(rank_cache_sha256='old'),
                lambda r:r['api'].update(source_sha256='old'),
                lambda r:r['ranks'][0]['memory']['zero_mappings'].update(count=2),
                lambda r:r['ranks'][0]['memory']['process_kib'].update(Pss=-1)):
            value=receipt(); mutate(value)
            with self.assertRaises(ValueError):memory.validate(value, 'probe', 'cache', '0')

    def test_warm_logs_reject_cold_fallback_fp8_errors_and_missing_nccl(self):
        logs = {node:f'[rank-cache] hit rank={rank} bytes=123\n'+
                '[fp8-cache] enabled=True hit=5 miss=0 errors=0\n'*2+'NCCL INFO Init COMPLETE\n'
                for rank,node in enumerate(pair.base.lifecycle.NODES)}
        pair.validate_logs(logs, warm=True)
        for text in (logs['local'].replace('hit rank', 'saved rank'),
                     logs['local'].replace('errors=0', 'errors=1'),
                     logs['local'].replace('NCCL INFO Init COMPLETE', ''),
                     logs['local']+'[rank-cache] another rank missed\n'):
            with self.assertRaises(ValueError): pair.validate_logs({**logs, 'local':text}, warm=True)
        pair.validate_logs({n:t.replace('hit rank', 'saved rank') for n,t in logs.items()}, warm=False)

    def test_memory_only_capture_cannot_be_analyzed_as_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'completion.json').write_text(json.dumps(dict(experiment='glm53-cpu-vote-memory',
                complete=True, restored_original=True)))
            with self.assertRaisesRegex(ValueError, 'not a prefill'): pair.base.analyze(Path(tmp))

    def test_arm_identity_rejects_capacity_and_non_vote_environment_drift(self):
        row=dict(image='image', args={'max-model-len':'1048576'}, mounts={'a':'sha'},
                 manifest_sha='manifest', model='model', hardware='gpu', env={host.KNOB:'0', 'OTHER':'same'})
        a={n:copy.deepcopy(row) for n in pair.base.lifecycle.NODES}; b=copy.deepcopy(a)
        for r in b.values():r['env'][host.KNOB]='1'
        pair.matched(a,b)
        for key,value in (('args',{}), ('mounts',{}), ('env',{'OTHER':'changed'})):
            changed=copy.deepcopy(b); changed['local'][key]=value
            with self.assertRaises(ValueError):pair.matched(a,changed)


class LifecycleTests(unittest.TestCase):
    def test_failed_second_arm_cleans_before_resetting_root_and_never_requests(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'FLEET_SESSION':'cpu'}):
            run=pair.Run(ROOT, 'sha', Path(tmp)); events=[]
            prepared={n:dict(original={'id':n}) for n in pair.base.lifecycle.NODES}
            def ready(_):
                events.append(('ready',run.arm))
                if run.arm=='BASE0':raise RuntimeError('failed warm boot')
            api=Mock(); api.post.side_effect=lambda path,body: {} if 'prefill-observe' in path else receipt()
            with (patch.object(run,'all',return_value=prepared),patch.object(run,'host'),
                  patch.object(run,'ready',side_effect=ready),patch.object(run,'snapshot',return_value={}),
                  patch.object(run,'attest_clone'),patch.object(run,'logs',return_value={}),
                  patch.object(run,'cleanup',side_effect=lambda:events.append(('cleanup',run.arm))),
                  patch.object(run,'phase') as phase,patch.object(pair,'validate_logs',return_value={}),patch.object(pair,'idle_observers'),
                  patch.object(pair.memory,'validate'),patch.object(pair,'PrivateObserverAPI',return_value=api),
                  patch.object(pair.base.lifecycle,'idle'),patch.object(pair.time,'sleep')):
                with self.assertRaisesRegex(RuntimeError, 'failed warm boot'):run.collect(prepared,{})
                phase.assert_not_called()
            self.assertEqual(run.out,Path(tmp))
            self.assertEqual(events,[('ready','PRIME'),('cleanup','PRIME'),('ready','BASE0'),('cleanup','BASE0')])

    def test_cleanup_failure_prevents_original_restart_until_supervisor_recovery(self):
        events=[]
        def transition(before,action):events.append(action);return {}
        with patch.object(pair.base.lifecycle,'transition_all',side_effect=transition), \
                patch.object(pair.base.lifecycle,'wait_restore') as restored:
            with self.assertRaisesRegex(RuntimeError,'clone still alive'):
                pair.base.lifecycle.with_paused({},lambda:None,lambda *_:None,
                    before_restore=Mock(side_effect=RuntimeError('clone still alive')))
            self.assertEqual(events,['stop']);restored.assert_not_called()

    def test_actual_cpu_snapshot_does_not_initialize_cuda(self):
        try:import torch
        except ImportError:self.skipTest('CPU torch required')
        if not Path('/proc/self/smaps').is_file():self.skipTest('Linux proc required')
        self.assertFalse(torch.cuda.is_initialized())
        result=memory.snapshot(torch)
        self.assertFalse(result['cuda_initialized']);self.assertFalse(torch.cuda.is_initialized())
        self.assertGreater(result['process_kib']['Pss'],0)
        self.assertEqual(result['pid'],os.getpid())

    def test_memory_endpoint_rejects_remote_and_options_before_rpc(self):
        try:import fastapi
        except ImportError:self.skipTest('FastAPI required')
        for address, body in (('10.0.0.1',{}),('127.0.0.1',{'change':True})):
            client=Mock()
            async def read():return body
            request=types.SimpleNamespace(url=types.SimpleNamespace(path='/glm53/cpu-vote-memory'),
                method='POST',client=types.SimpleNamespace(host=address),json=read,
                app=types.SimpleNamespace(state=types.SimpleNamespace(engine_client=client)))
            result=asyncio.run(memory.middleware(request,Mock()))
            self.assertIn(result.status_code,(400,403));client.collective_rpc.assert_not_called()


if __name__=='__main__':unittest.main()
