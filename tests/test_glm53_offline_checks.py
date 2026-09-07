import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('offline', Path(__file__).resolve().parents[1]/'probes/glm53_offline_checks.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class OfflineTests(unittest.TestCase):
    def test_fp8_trace_is_separate_and_cannot_start_after_failed_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            argv=['offline','--out',str(Path(directory)/'evidence'),'--probe-source','/frozen',
                  '--probe-revision','a'*40,'--fp8-trace']
            with patch('sys.argv',argv),patch.dict(os.environ,OFFLINE_SOURCE_REV='a'*40), \
                 patch.object(m,'check_holder'),patch.object(m,'pinned'), \
                 patch.object(m,'probe_api_preflight',return_value={}), \
                 patch.object(m,'compile_preflight',side_effect=RuntimeError('stop before serving')), \
                 patch.object(m,'snapshot') as snapshot,patch.object(m.signal,'signal'):
                self.assertEqual(m.main(),1)
                snapshot.assert_not_called()
                self.assertEqual(m.PINS,(('moe-m64-fp8-trace','/frozen','a'*40,
                    ['bash','probes/run_glm53_moe_m64_tp4_check.sh','--fp8-trace']),))

    def test_fp8_diagnostic_uses_a_distinct_label_and_command(self):
        with tempfile.TemporaryDirectory() as directory:
            argv=['offline','--out',str(Path(directory)/'evidence'),'--probe-source','/frozen',
                  '--probe-revision','a'*40,'--fp8-diagnostic']
            with patch('sys.argv',argv),patch.dict(os.environ,OFFLINE_SOURCE_REV='a'*40), \
                 patch.object(m,'check_holder'),patch.object(m,'pinned'), \
                 patch.object(m,'probe_api_preflight',return_value={}), \
                 patch.object(m,'compile_preflight',side_effect=RuntimeError('stop before serving')), \
                 patch.object(m,'snapshot') as snapshot,patch.object(m.signal,'signal'):
                self.assertEqual(m.main(),1)
                snapshot.assert_not_called()
                self.assertEqual(m.PINS,(('moe-m64-fp8-diagnostic','/frozen','a'*40,
                    ['bash','probes/run_glm53_moe_m64_tp4_check.sh','--fp8-diagnostic']),))

    def test_failed_compile_prevents_snapshot_or_container_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'evidence'
            argv=['offline','--out',str(out),'--probe-source','/frozen','--probe-revision','a'*40]
            with patch('sys.argv',argv),patch.dict(os.environ,OFFLINE_SOURCE_REV='a'*40),patch.object(m,'check_holder'),patch.object(m,'pinned'),patch.object(m,'probe_api_preflight',return_value={}),patch.object(m,'compile_preflight',side_effect=RuntimeError('compile failed')) as compile,patch.object(m,'snapshot') as snapshot,patch.object(m,'transition_all') as transition,patch.object(m.signal,'signal'):
                self.assertEqual(m.main(),1)
                compile.assert_called_once()
                self.assertEqual(compile.call_args.args[:2],('/frozen','a'*40))
                snapshot.assert_not_called();transition.assert_not_called()
            result=json.loads((out/'completion.json').read_text())
            self.assertIn('compile failed',result['error'])
            self.assertEqual(result['probes'],{})

    def states(self):
        return {n: dict(id=n, running=True, auto_remove=False, image=m.IMAGE,
                        overlays={'file':'hash'}, manifest='hash', port=8000) for n in m.NODES}

    def test_no_partial_or_automatically_deleted_or_wrong_image_fleet(self):
        self.assertEqual(m.validate_before(self.states()), 'present')
        self.assertEqual(m.validate_before(dict.fromkeys(m.NODES)), 'absent')
        for delta in [None, {'auto_remove': True}, {'running': False}, {'image': 'other'}]:
            state = self.states()
            state['local'] = None if delta is None else dict(state['local'], **delta)
            with self.subTest(delta=delta), self.assertRaises(RuntimeError):
                m.validate_before(state)

    def test_probe_error_restores_before_propagating(self):
        events = []
        def move(before, action): events.append(action); return {}
        def run(): events.append('probe'); raise ValueError('numerics fail')
        with patch.object(m, 'transition_all', move), patch.object(m, 'wait_restore', return_value={}), self.assertRaises(ValueError):
            m.with_paused(self.states(), run, lambda *a: None)
        self.assertEqual(events, ['stop', 'probe', 'start'])

    def test_partial_stop_error_restores_without_running_gpu(self):
        events = []
        def move(before, action):
            events.append(action)
            if action == 'stop': raise RuntimeError('one node failed to stop')
            return {}
        with patch.object(m, 'transition_all', move), patch.object(m, 'wait_restore', return_value={}), self.assertRaises(RuntimeError):
            m.with_paused(self.states(), lambda: self.fail('GPU must not run'), lambda *a: None)
        self.assertEqual(events, ['stop', 'start'])

    def test_restore_failure_is_not_success(self):
        with patch.object(m, 'transition_all', return_value={}), patch.object(m, 'wait_restore', side_effect=RuntimeError('rank exited')), self.assertRaises(RuntimeError):
            m.with_paused(self.states(), lambda: 0, lambda *a: None)

    def test_lost_hold_refuses_before_remote_mutation(self):
        with patch.object(m, 'check_holder', side_effect=RuntimeError('not ours')), patch.object(m, 'remote') as remote, self.assertRaises(RuntimeError):
            m.transition('local', self.states()['local'], 'stop')
        remote.assert_not_called()

    def test_public_restore_uses_actual_fleet_decision_and_refuses_errors(self):
        with patch.object(m, 'check_holder'), patch.dict(m.os.environ, FLEET_SESSION='test'), patch.object(m.subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess([], 1, b'no (next boots next and replaces whatever is up)\n')
            result = {}
            m.restore_public(Path('/unused'), lambda *a: None, result)
            self.assertIn('queued boot', result['public_restore'])
            self.assertEqual(run.call_count, 1)
            run.return_value = subprocess.CompletedProcess([], 2, b'')
            with self.assertRaises(RuntimeError):
                m.restore_public(Path('/unused'), lambda *a: None, {})

    def test_generated_transition_executes_stop_and_start_with_unordered_mounts(self):
        self.generated_transition(change_source=False)

    def test_generated_transition_refuses_real_source_change_before_docker_mutation(self):
        self.generated_transition(change_source=True)

    def generated_transition(self, change_source):
        # Run the COMPLETE emitted Python in real child processes. Only the
        # Docker executable is substituted with a stateful temporary fake.
        # This catches quoting/name resolution and inspect serialization bugs
        # that mocking transition_all/remote cannot exercise.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state.json';docker=root/'docker'
            state.write_text(json.dumps(dict(running=True,inspects=0,mutations=[],source='/tmp/first')))
            docker.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
path=pathlib.Path(os.environ['PREFILL_FAKE_DOCKER_STATE'])
s=json.loads(path.read_text());args=sys.argv[1:];identifier='d'*64
if args==['ps','-a','--format','{{.Names}}']:
    print('glm53')
elif args==['inspect','glm53']:
    s['inspects']+=1
    mounts=[dict(Destination='/a',Source=s['source'],RW=False),dict(Destination='/b',Source='/tmp/second',RW=False)]
    if s['inspects']%2:mounts.reverse()
    print(json.dumps([dict(Id=identifier,Image='sha256:'+'a'*64,
        State=dict(Running=s['running'],StartedAt='running-at-'+str(len(s['mutations']))),
        Config=dict(Cmd=['serve --port 8000'],Env=['SETTING=value']),
        HostConfig=dict(AutoRemove=False),Mounts=mounts)]))
elif args in (['stop','--time','45',identifier],['start',identifier]):
    s['mutations'].append(args);s['running']=args[0]=='start';print(identifier)
else:
    raise SystemExit('unexpected fake Docker invocation: '+repr(args))
path.write_text(json.dumps(s))
''')
            docker.chmod(0o755)
            with patch.dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],PREFILL_FAKE_DOCKER_STATE=str(state)),patch.object(m,'check_holder'):
                before=m.remote('local',m.INSPECT+"\nprint(json.dumps(inspect('glm53')))")
                if change_source:
                    data=json.loads(state.read_text());data['source']='/tmp/different';state.write_text(json.dumps(data))
                    with self.assertRaisesRegex(RuntimeError,'identity/config/source changed'):
                        m.transition('local',before,'stop')
                    self.assertEqual(json.loads(state.read_text())['mutations'],[])
                else:
                    stopped=m.transition('local',before,'stop')
                    self.assertFalse(stopped['running'])
                    started=m.transition('local',before,'start')
                    self.assertTrue(started['running'])
                    self.assertEqual(m.identity(before),m.identity(started))
                    self.assertEqual(json.loads(state.read_text())['mutations'],
                                     [['stop','--time','45','d'*64],['start','d'*64]])


if __name__ == '__main__':
    unittest.main()
