"""CPU-only lifecycle, cloned configuration and request evidence contracts."""
import base64
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'probes'),str(ROOT/'bench')]
import glm53_observation_host as host
import prefill_observation_run as runner
import prefill_observation_requests as requests
import fleet_boot
import fleet_observation_cleanup


def original():
    script='set -e\nvllm serve /models/model --host 0.0.0.0 --port 8000 --headless '
    script+='--max-model-len 1048576 --profiler-config \'{"profiler":"torch"}\' > /glmlogs/glm53.log 2>&1'
    encoded=base64.b64encode(script.encode()).decode()
    return dict(Id='original-id',Image='sha256:pinned',State={'Running':True},
        Config=dict(Entrypoint=['/bin/bash'],Cmd=['-c','echo '+encoded+' | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh'],
            Image='tag',Env=['SECRET=keep-private','PYTHONPATH=/original','VLLM_GLM53_DEV_LAB=0'],Labels=None),
        HostConfig=dict(NetworkMode='host',AutoRemove=False,DeviceRequests=[{'Count':-1}],
            Binds=['/original/logs:/glmlogs:rw','/original/prof:/prof','/model:/models/model:ro',
                   '/overlay:/overlay:ro'],Memory=1234,IpcMode='host'))


class CloneTests(unittest.TestCase):
    def test_clone_preserves_model_capacity_environment_and_original(self):
        before=original();saved=copy.deepcopy(before)
        result=host.clone_payload(before,directory='/out',source='/source',session='cpu')
        self.assertEqual(before,saved)
        script=base64.b64decode(result['Cmd'][1].split()[1]).decode()
        for text in ('--host 127.0.0.1','--port 18000','--headless','--max-model-len 1048576',
                     '--worker-extension-cls glm53_prefill_observer.WorkerExtension',
                     '--middleware glm53_prefill_observer.middleware'):
            self.assertIn(text,script)
        env=dict(e.split('=',1) for e in result['Env'])
        self.assertEqual(env,dict(SECRET='keep-private',PYTHONPATH='/observer:/original',VLLM_GLM53_DEV_LAB='0'))
        expected=copy.deepcopy(before['HostConfig'])
        expected['Binds']=['/out/glmlogs:/glmlogs:rw','/out/prof:/prof','/model:/models/model:ro',
                           '/overlay:/overlay:ro','/source/probes:/observer:ro']
        self.assertEqual(result['HostConfig'],expected)
        self.assertEqual(result['Image'],before['Image'])
        self.assertEqual(result['Labels'][host.OWNER+'.original'],before['Id'])

    def test_ambiguous_or_experimental_incoming_is_rejected(self):
        variants=[]
        for key,value in [('AutoRemove',True),('NetworkMode','bridge'),('Mounts',[{}]),('Binds',[])]:
            c=original();c['HostConfig'][key]=value;variants.append(c)
        for key in ('VLLM_GLM53_B12X_PREFILL_M64','VLLM_GLM53_PREFILL_SP_RS_INT8','VLLM_GLM53_DEV_LAB'):
            c=original();c['Config']['Env'].append(key+'=1');variants.append(c)
        c=original();c['Config']['Cmd']=['-c','vllm serve unreviewed'];variants.append(c)
        for c in variants:
            with self.assertRaises(ValueError):host.clone_payload(c,directory='/out',source='/source',session='cpu')

    def test_remove_requires_exact_session_label_and_never_removes_original(self):
        args=dict(action='remove',name='glm53-observe-cpu',session='cpu',directory='/out',source='/source')
        with patch.object(host,'inspect',return_value=original()),patch.object(host.subprocess,'run') as command:
            with self.assertRaises(RuntimeError):host.dispatch(**args)
            command.assert_not_called()
        clone=original();clone['Id']='clone-id';clone['Config']['Labels']={host.OWNER:'cpu'}
        with patch.object(host,'inspect',return_value=clone),patch.object(host.subprocess,'run') as command:
            self.assertTrue(host.dispatch(**args)['removed'])
            self.assertEqual(command.call_args.args[0],['docker','rm','-f','clone-id'])


class LifecycleTests(unittest.TestCase):
    def test_parallel_partial_failure_settles_every_node(self):
        calls=[]
        def action(node):
            calls.append(node)
            if node==1:raise RuntimeError('partial failure')
            return node
        with self.assertRaises(RuntimeError):runner.settled(range(4),action)
        self.assertEqual(sorted(calls),list(range(4)))

    def exercise(self,*,prepare_failure=False):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{
            'FLEET_SESSION':'cpu','FLEET_RUNNER_REPO':str(ROOT),'FLEET_OBSERVATION_CLONES':'1'}):
            out=Path(tmp)/'capture';run=runner.Run(ROOT,'frozen',out);events=[]
            before={node:{'id':node+'-original'} for node in runner.lifecycle.NODES}
            def all_nodes(action):
                events.append(action)
                if action=='prepare' and prepare_failure:raise RuntimeError('partial prepare')
                return before
            def transition(expected,action):
                self.assertEqual(expected,before);events.append(action);return before
            def failed_collection(*_):
                try:raise RuntimeError('request failed')
                finally:run.cleanup()
            with (patch.object(runner.lifecycle,'check_holder'),patch.object(runner.lifecycle,'pinned'),
                  patch.object(runner.prefill_serving,'run_owned') as command,patch.object(runner.lifecycle,'snapshot',return_value=before),
                  patch.object(runner.lifecycle,'validate_before',return_value='present'),patch.object(runner.lifecycle,'idle'),
                  patch.object(runner.lifecycle,'remote',return_value={'disk_free_gib':200}),
                  patch.object(run,'snapshot',return_value=before),patch.object(run,'all',side_effect=all_nodes),
                  patch.object(run,'collect',side_effect=failed_collection),
                  patch.object(run,'cleanup',side_effect=lambda:events.append('remove-clones')),
                  patch.object(runner.lifecycle,'transition_all',side_effect=transition)):
                self.assertEqual(run.run(),1)
            result=json.loads((out/'completion.json').read_text())
            self.assertFalse(result['complete']);self.assertFalse(result['performance_acceptance'])
            self.assertTrue(result['cleanup_complete'])
            self.assertEqual(result['original_stop_attempted'], not prepare_failure)
            if prepare_failure:self.assertNotIn('stop',events)
            else:self.assertIn('stop',events)
            self.assertNotIn('start',events)
            self.assertIn('remove-clones',events)
            command.assert_not_called()
            return events

    def test_request_failure_cleans_clones_without_recovery_boot(self):self.exercise()
    def test_partial_prepare_failure_cleans_clones_without_stopping_originals(self):self.exercise(prepare_failure=True)
    def test_boot_failure_always_runs_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'FLEET_SESSION':'cpu'}):
            run=runner.Run(ROOT,'frozen',Path(tmp))
            with patch.object(run,'host'),patch.object(run,'ready',side_effect=RuntimeError('boot failed')),patch.object(run,'cleanup') as cleanup:
                with self.assertRaises(RuntimeError):run.collect({}, {})
                cleanup.assert_called_once()

    def test_supervisor_cleanup_precedes_handoff_and_survives_payload_cancellation(self):
        with patch.dict(os.environ,{'FLEET_DIR':'/tmp/test','FLEET_OBSERVATION_CLONES':'1'}):
            supervisor=fleet_boot.Supervisor('fleet','cpu','30','note',[])
            supervisor.stopping=143
            with (patch.object(supervisor,'held',return_value=True),patch.object(supervisor,'event') as event,
                  patch.object(supervisor,'execute',side_effect=[1,0]) as execute,patch.object(fleet_boot.time,'sleep')):
                self.assertEqual(supervisor.cleanup_observation(),0)
                self.assertEqual(execute.call_count,2)
                self.assertEqual(supervisor.stopping,143)
                self.assertIn('observation-cleanup-retry',[c.args[0] for c in event.call_args_list])
            supervisor.env.pop('FLEET_OBSERVATION_CLONES')
            with patch.object(supervisor,'execute') as execute:
                self.assertEqual(supervisor.cleanup_observation(),0);execute.assert_not_called()

    def test_detached_cleanup_settles_all_nodes_on_failure(self):
        calls=[]
        def call(command,**kwargs):
            calls.append(command)
            from types import SimpleNamespace
            return SimpleNamespace(returncode=1 if command[0]=='python3' else 0,stdout='{"removed":true}')
        with self.assertRaises(RuntimeError):fleet_observation_cleanup.cleanup('cpu',call)
        self.assertEqual(len(calls),4)

    def test_detached_cleanup_never_removes_foreign_or_public_container(self):
        clone=dict(Id='owned-id',Config={'Labels':{'codex.glm.prefill-observation':'cpu'}})
        for owner in ('cpu','someone-else'):
            clone['Config']['Labels']['codex.glm.prefill-observation']=owner
            outputs=['glm53\nglm53-observe-cpu\n',json.dumps([clone]),'glm53\n']
            with (patch.object(fleet_observation_cleanup.subprocess,'check_output',side_effect=outputs),
                  patch.object(fleet_observation_cleanup.subprocess,'run') as command,redirect_stdout(io.StringIO())):
                if owner=='cpu':
                    exec(fleet_observation_cleanup.SCRIPT,{'session':'cpu'})
                    self.assertEqual(command.call_args.args[0],['docker','rm','-f','owned-id'])
                else:
                    with self.assertRaises(RuntimeError):exec(fleet_observation_cleanup.SCRIPT,{'session':'cpu'})
                    command.assert_not_called()

    def test_phase_sets_actual_benchmark_port_and_model(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'FLEET_SESSION':'cpu'}):
            out=Path(tmp);run=runner.Run(ROOT,'frozen',out)
            def execute(command,**kwargs):
                with patch.dict(os.environ,kwargs['env']):
                    module=requests.onepass._load('bench-dec.py','observation_port_test')
                    self.assertEqual(module.URL,requests.BASE+'/v1/chat/completions')
                    self.assertEqual(module.METRICS,requests.BASE+'/metrics')
                    self.assertEqual(module.MODEL,'glm-5.3-flash')
                (out/'BASE').mkdir();(out/'BASE/result.json').write_text('{"complete":true}')
            with patch.object(runner.prefill_serving,'run_owned',side_effect=execute):run.phase('baseline','BASE')


class RequestTests(unittest.TestCase):
    def test_quality_and_short_request_coverage_are_mandatory(self):
        timing=[dict(ctx=c,question=q) for c,q in [(2000,0),(2000,1),(2000,2),(32000,'all'),(128000,'all')]]
        record=dict(requests=timing,quality={'ok':9,'total':9},korean={'dirty':0})
        requests.validate_baseline(record,[{}]*5)
        for key,value in [('requests',timing[1:]),('quality',{'ok':8,'total':9}),
                          ('korean',{'dirty':1}),('evidence_issues',['external traffic'])]:
            with self.assertRaises(RuntimeError):requests.validate_baseline(dict(record,**{key:value}),[{}]*5)

    def test_prefill_identity_permits_only_limit_and_salt_changes(self):
        body=dict(messages=[{'role':'user','content':'문서'}],seed=None,temperature=0,max_tokens=400,cache_salt='a')
        identity=lambda p:requests.prefill_identity(json.dumps(p).encode())
        self.assertEqual(identity(body),identity(dict(body,max_tokens=1,cache_salt='b')))
        for key,value in [('seed',7),('messages',[{'role':'user','content':'다른 문서'}]),('temperature',1)]:
            self.assertNotEqual(identity(body),identity(dict(body,**{key:value})))


if __name__=='__main__':unittest.main()
