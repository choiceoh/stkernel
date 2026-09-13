"""Exercise the actual bracket loop when canonical quality fails after recording.

Only the consumer and fleet actions are fixtures. The shell measure/leg/probe
functions run unchanged, including their exit-status and append-offset checks.
No GPU, container, network or fleet lease is used.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHA = '0123456789abcdef0123456789abcdef01234567'
CONSUMER = '''
import json, os, pathlib, sys
run = int(os.environ['ONEPASS_RUN_INDEX'])
case = json.loads(os.environ['CASES'])[run - 1]
with open(os.environ['EVENTS'], 'a') as stream:
    stream.write('measure ' + str(run) + '\\n')
record = dict(engine='st', arm_sha=os.environ['ST_BRACKET_SHA'],
              name=sys.argv[sys.argv.index('--name') + 1], run_index=run,
              run_id='run-' + str(run), boot_id='fixed-container|started',
              session=os.environ['FLEET_SESSION'], recording=dict(status='complete'))
record.update(case.get('record', {}))
if case.get('append', True):
    with open(os.environ['ONEPASS_JSONL'], 'a') as stream:
        stream.write(case.get('raw', json.dumps(record)) + '\\n')
sys.exit(case.get('rc', 0))
'''


class CompletionTests(unittest.TestCase):
    def test_production_shape_keeps_the_selected_bracket_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            production = root / 'production.env'
            production.write_text('PORT=8000\nST_KV_GIB=24\nSTK_execution_overlap=1\n')
            source = (ROOT / 'bench/st_bracket.sh').read_text().rsplit('\ncase "${1:-}" in', 1)[0]
            source += '\nshape\n[ "$PORT" = "$EXPECTED_PORT" ] && [ "$ST_KV_GIB" = 24 ] && [ "$ST_PRODUCTION" = 1 ] && [ -z "${STK_execution_overlap+x}" ]\n'
            script = root / 'runner.sh'
            script.write_text(source)
            for port in ('8001', '8017'):
                with self.subTest(port=port):
                    env = dict(os.environ, REPO=str(root), LOGD=str(root / 'logs'),
                               ST_PRODUCTION_ENV=str(production), ST_BRACKET_PORT=port,
                               EXPECTED_PORT=port)
                    result = subprocess.run(['bash', str(script)], text=True, capture_output=True,
                                            timeout=20, env=env)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def execute(self, cases, *, verb='leg', stale=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'bench').mkdir()
            (root / 'bench/onepass.py').write_text(CONSUMER)
            source = (ROOT / 'bench/st_bracket.sh').read_text().rsplit('\ncase "${1:-}" in', 1)[0]
            source += '''
sha_of() { echo "$1"; }
boot_arm() { ARM=$1; RELEASE=/fixture; echo boot >> "$EVENTS"; }
stop_arm() { echo stop >> "$EVENTS"; }
reset_prefix() { echo reset >> "$EVENTS"; }
door_up() { return 0; }
docker() { echo "$EXPECTED_SHA"; }
'''
            source += '\nleg candidate "$EXPECTED_SHA"\n' if verb == 'leg' else '\nprobe "$EXPECTED_SHA"\n'
            script = root / 'runner.sh'
            script.write_text(source)
            ledger = root / 'records.jsonl'
            if stale:
                ledger.write_text(json.dumps(dict(engine='st', arm_sha=SHA, name='candidate',
                    run_index=1, run_id='old', boot_id='fixed-container|started', session='test',
                    recording=dict(status='complete'))) + '\n')
            events = root / 'events'
            env = dict(os.environ, REPO=str(root), LOGD=str(root / 'logs'),
                       ONEPASS_JSONL=str(ledger), EVENTS=str(events),
                       CASES=json.dumps(cases), EXPECTED_SHA=SHA,
                       FLEET_SESSION='test', FLEET_REHEARSE='0', ST_BRACKET_RUNS='2',
                       ST_PROBE_RUNS='2')
            result = subprocess.run(['bash', str(script)], text=True, capture_output=True,
                                    timeout=20, env=env)
            return result, events.read_text().splitlines()

    def test_recorded_quality_failure_keeps_one_boot_and_preserves_failure(self):
        for codes in ((2, 0), (0, 2), (2, 2)):
            with self.subTest(codes=codes):
                result, events = self.execute([dict(rc=rc) for rc in codes])
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(events, ['boot', 'measure 1', 'reset', 'measure 2', 'stop'])
                self.assertIn('recorded issues', result.stdout)

    def test_live_probe_keeps_both_runs_without_boot_or_stop(self):
        result, events = self.execute([dict(rc=2), dict(rc=0)], verb='probe')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(events, ['reset', 'measure 1', 'reset', 'measure 2'])

    def test_unrecorded_or_incomplete_exit_two_stops_after_first_run(self):
        for case in (dict(append=False), dict(record={'recording': {'status': 'incomplete'}}),
                     dict(raw='{broken'), dict(record={'rehearsal': True}),
                     dict(record={'boot_id': ''})):
            with self.subTest(case=case):
                result, events = self.execute([dict(rc=2, **case), {}])
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(events, ['boot', 'measure 1', 'stop'])

    def test_existing_record_cannot_turn_argument_failure_into_completion(self):
        result, events = self.execute([dict(rc=2, append=False), {}], stale=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(events, ['boot', 'measure 1', 'stop'])

    def test_fresh_record_must_match_the_actual_invocation(self):
        for key, value in (('engine', 'vllm'), ('arm_sha', 'bad'), ('name', 'different'),
                           ('run_index', 2), ('session', 'someone-else')):
            with self.subTest(key=key):
                result, events = self.execute([dict(rc=2, record={key: value}), {}])
                self.assertEqual(result.returncode, 2)
                self.assertEqual(events, ['boot', 'measure 1', 'stop'])

    def test_runtime_failure_is_not_reclassified_by_a_record(self):
        result, events = self.execute([dict(rc=1), {}])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(events, ['boot', 'measure 1', 'stop'])

    def test_second_pass_runtime_failure_remains_the_terminal_error(self):
        result, events = self.execute([dict(rc=2), dict(rc=1)])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(events, ['boot', 'measure 1', 'reset', 'measure 2', 'stop'])

    def test_successful_pair_is_unchanged(self):
        result, events = self.execute([{}, {}])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(events, ['boot', 'measure 1', 'reset', 'measure 2', 'stop'])


if __name__ == '__main__':
    unittest.main()
