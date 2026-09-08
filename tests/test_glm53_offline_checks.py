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

    def test_fully_stopped_predecessor_is_reused_without_boot(self):
        states = {n: dict(state, running=False) for n, state in self.states().items()}
        self.assertEqual(m.validate_before(states), 'stopped')

    def test_probe_error_propagates_without_recovery_boot(self):
        events = []
        def move(before, action): events.append(action); return {}
        def run(): events.append('probe'); raise ValueError('numerics fail')
        with patch.object(m, 'transition_all', move), self.assertRaises(ValueError):
            m.with_paused(self.states(), run, lambda *a: None)
        self.assertEqual(events, ['stop', 'probe'])

    def test_partial_stop_error_does_not_run_gpu_or_start_recovery(self):
        events = []
        def move(before, action):
            events.append(action)
            raise RuntimeError('one node failed to stop')
        with patch.object(m, 'transition_all', move), self.assertRaises(RuntimeError):
            m.with_paused(self.states(), lambda: self.fail('GPU must not run'), lambda *a: None)
        self.assertEqual(events, ['stop'])

    def test_successful_probe_returns_immediately_after_measurement(self):
        with patch.object(m, 'transition_all', return_value={}) as move:
            self.assertEqual(m.with_paused(self.states(), lambda: 7, lambda *a: None), 7)
        self.assertEqual(move.call_args.args[1], 'stop')
        self.assertEqual(move.call_count, 1)

    def test_lost_hold_refuses_before_remote_mutation(self):
        with patch.object(m, 'check_holder', side_effect=RuntimeError('not ours')), patch.object(m, 'remote') as remote, self.assertRaises(RuntimeError):
            m.transition('local', self.states()['local'], 'stop')
        remote.assert_not_called()

    def test_session_start_is_refused_before_remote_mutation(self):
        with patch.object(m, 'check_holder'), patch.object(m, 'remote') as remote:
            with self.assertRaisesRegex(ValueError, 'session recovery starts are disabled'):
                m.transition('local', self.states()['local'], 'start')
            remote.assert_not_called()

    def test_generated_transition_stops_without_restart_with_unordered_mounts(self):
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
                    self.assertEqual(m.identity(before),m.identity(stopped))
                    with self.assertRaises(ValueError):m.transition('local',before,'start')
                    self.assertEqual(json.loads(state.read_text())['mutations'],
                                     [['stop','--time','45','d'*64]])


if __name__ == '__main__':
    unittest.main()
