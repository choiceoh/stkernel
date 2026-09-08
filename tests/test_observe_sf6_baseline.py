"""CPU-only admission, same-boot capture, and failed-proof preservation."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'probes'), str(ROOT / 'tests')]
import test_decode_next_runtime as fixtures
import observe_decode_next_onepass as frozen
import observe_sf6_baseline as sidecar

STARTED = 1788904382.
BOOT = '2026-09-08T22:00:00+00:00'
REVISION = 'a' * 40


class FakeReader:
    def __init__(self):
        self.calls, self.health_calls = 0, 0
        self.suffix = ''
        self.on_ranks = lambda: None

    def report(self, host):
        value = fixtures.direct_report('baseline', host)
        value['serving_argv'] += ['--host', '127.0.0.1', '--port', '18000']
        value['boot_id'] = host + self.suffix + '|' + BOOT
        return value

    def head(self):
        value = self.report('srv2')
        return {key: value[key] for key in ('boot_id', 'image', 'running', 'knobs')}

    def healthy(self):
        self.health_calls += 1
        return True

    def ranks(self, mode, expected):
        assert mode == 'baseline'
        self.calls += 1
        values = {}
        raw = '\n'.join(line for line in fixtures.COMMON_LOG.splitlines()
                        if 'AR consumer MHC' not in line).encode()
        for host in frozen.HOSTS:
            value = self.report(host)
            value['markers'] = frozen.proof.parse_markers(raw.decode())
            value['log_sha256'] = sidecar.sha(raw)
            value['errors'] = frozen.proof.validate_report(value, expected, sf6_direct=True)
            value['valid'] = not value['errors']
            values[host] = dict(report=value, log_b64=base64.b64encode(raw).decode(),
                                error='; '.join(value['errors']))
        self.on_ranks()
        return values


class BaselineObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.args = SimpleNamespace(repo=ROOT, out=self.base / 'out', name='base', session='session',
            ticket='123', pid=42, start='777', fleet_dir=self.base / 'fleet', port=18000, timeout=100.,
            revision=REVISION)
        self.reader = FakeReader()
        self.go = dict(state='GO', reservation=dict(started_at=STARTED))
        self.terminal = False

    def record(self):
        self.args.out.mkdir(parents=True, exist_ok=True)
        data = '{ "name": "base", "boot_id": ' + json.dumps(self.reader.head()['boot_id']) + ' }\n'
        (self.args.out / 'records.raw.jsonl').write_text(data)

    def run_observer(self, sleeper=None):
        def default_sleep(_):
            if self.reader.calls:
                self.record()
                self.terminal = True
        with patch.object(sidecar, 'load_frozen', return_value=(frozen, self.args.revision, fixtures.MANIFEST)), \
             patch.object(frozen, 'Reader', return_value=self.reader), \
             patch.object(sidecar, 'reservation', side_effect=lambda args: dict(state='TERMINAL') if self.terminal else self.go), \
             patch.object(sidecar.time, 'sleep', side_effect=sleeper or default_sleep), \
             patch('builtins.print'):
            code = sidecar.observe(self.args)
        return code, json.loads((self.args.out / 'baseline-observer.json').read_text())

    def test_same_boot_after_release_collects_all_evidence_without_fake_pass(self):
        code, state = self.run_observer()
        self.assertEqual(code, 1)  # Strict MHC failure remains a failure.
        self.assertEqual(state['collection_status'], 'COMPLETE')
        self.assertEqual(state['runtime_validation'], 'FAIL')
        self.assertEqual(state['source_commit'], self.args.revision)
        self.assertEqual((self.args.out / 'baseline-source.commit').read_text().strip(), self.args.revision)
        self.assertEqual(self.reader.calls, 2)
        for phase, identity in state['phases'].items():
            folder = self.args.out / identity['path']
            receipt_path = folder / 'receipt.json'
            self.assertEqual(sidecar.sha(receipt_path.read_bytes()), identity['receipt_sha256'])
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt['collection_status'], 'COMPLETE')
            self.assertEqual(receipt['runtime_validation'], 'FAIL')
            self.assertEqual(receipt['source_commit'], self.args.revision)
            self.assertTrue(any('missing AR/MHC self-test PASS' in e for e in receipt['validation_errors']))
            for filename, digest in receipt['artifacts_sha256'].items():
                self.assertEqual(sidecar.sha((folder / filename).read_bytes()), digest)
            if phase == 'runtime':
                self.assertFalse(receipt['owned_before'])
                self.assertEqual(receipt['record_sha256'], sidecar.sha((folder / 'record.raw.json').read_bytes()))
        self.assertFalse(list(self.args.out.glob('observed-*.json')))

    def test_after_boot_change_fails_before_foreign_rank_collection(self):
        def sleep(_):
            self.record()
            self.reader.suffix = '-replacement'
        code, state = self.run_observer(sleep)
        self.assertEqual(code, 1)
        self.assertEqual(state['collection_status'], 'FAILED')
        self.assertEqual(self.reader.calls, 1)
        self.assertTrue(any('not bound' in e for e in state['errors']))

    def test_record_before_prepared_never_collects(self):
        self.record()
        code, state = self.run_observer()
        self.assertEqual(code, 1)
        self.assertEqual(self.reader.calls, 0)
        self.assertIn('before any accepted prepared', state['errors'][0])

    def test_record_during_prepared_is_sealed_failure_not_promoted(self):
        self.reader.on_ranks = self.record
        code, state = self.run_observer()
        self.assertEqual(code, 1)
        self.assertEqual(state['collection_status'], 'FAILED')
        self.assertNotIn('prepared', state['phases'])
        attempt = self.args.out / state['attempts'][0]['path'] / 'receipt.json'
        self.assertIn('record appeared', ' '.join(json.loads(attempt.read_text())['collection_errors']))

    def test_log_tamper_and_source_drift_are_collection_failures(self):
        self.args.out.mkdir()
        original = self.reader.ranks
        for change in ('log', 'source'):
            def ranks(mode, expected):
                values = original(mode, expected)
                if change == 'log':
                    values['srv3']['log_b64'] = base64.b64encode(b'changed').decode()
                else:
                    values['srv3']['report']['source_sha256'] = {}
                return values
            with patch.object(sidecar, 'reservation', return_value=self.go), \
                 patch.object(self.reader, 'ranks', side_effect=ranks):
                result = sidecar.capture(self.args, frozen, self.reader, fixtures.MANIFEST,
                    self.go, 'prepared', self.args.out / change)
            self.assertEqual(result['receipt']['collection_status'], 'FAILED')

    def test_wrong_head_endpoint_is_not_accepted_as_isolated_baseline(self):
        self.args.out.mkdir()
        original = self.reader.report
        for change in ('0.0.0.0', '8000'):
            def report(host):
                value = original(host)
                if host == 'srv2':
                    argv = value['serving_argv']
                    argv[argv.index('--host' if change == '0.0.0.0' else '--port') + 1] = change
                return value
            with patch.object(sidecar, 'reservation', return_value=self.go), \
                 patch.object(self.reader, 'report', side_effect=report):
                result = sidecar.capture(self.args, frozen, self.reader, fixtures.MANIFEST,
                    self.go, 'prepared', self.args.out / change)
            self.assertIn('head serving endpoint', ' '.join(result['receipt']['collection_errors']))

    def test_queued_and_terminal_do_not_import_reader_or_observe_system(self):
        with patch.object(sidecar, 'reservation', side_effect=[{'state': 'WAIT'}, {'state': 'TERMINAL'}]), \
             patch.object(sidecar, 'load_frozen', side_effect=AssertionError('imported before GO')) as load, \
             patch.object(sidecar.time, 'sleep') as sleep, patch('builtins.print'):
            self.assertEqual(sidecar.observe(self.args), 1)
        load.assert_not_called()
        sleep.assert_called_once_with(1)

    def test_failed_collection_can_retry_before_record_but_only_three_times(self):
        self.reader.ranks = lambda mode, expected: {}
        code, state = self.run_observer(lambda _: None)
        self.assertEqual(code, 1)
        self.assertEqual(len(state['attempts']), 3)
        self.assertIn('three prepared', state['errors'][0])

    def test_reused_output_and_nonfrozen_source_are_rejected(self):
        self.run_observer()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            sidecar.observe(self.args)
        with patch.object(frozen, 'frozen_manifest', return_value=('f' * 40, fixtures.MANIFEST)), \
             patch.object(sidecar.importlib.util, 'module_from_spec', return_value=frozen), \
             patch.object(sidecar.importlib.util, 'spec_from_file_location') as spec:
            with self.assertRaisesRegex(ValueError, 'requires frozen'):
                sidecar.load_frozen(ROOT, self.args.revision)

    def test_loader_accepts_only_the_explicit_exact_revision(self):
        for revision in ('b' * 40, '0123456789abcdef' * 2 + '12345678'):
            with patch.object(frozen, 'frozen_manifest', return_value=(revision, fixtures.MANIFEST)), \
                 patch.object(sidecar.importlib.util, 'module_from_spec', return_value=frozen), \
                 patch.object(sidecar.importlib.util, 'spec_from_file_location'):
                self.assertEqual(sidecar.load_frozen(ROOT, revision)[1], revision)
                with self.assertRaisesRegex(ValueError, 'requires frozen'):
                    sidecar.load_frozen(ROOT, 'c' * 40)
        for invalid in ('main', '', 'a' * 39, 'a' * 41, 'g' * 40, 'A' * 40):
            with patch.object(sidecar.importlib.util, 'spec_from_file_location') as load:
                with self.assertRaisesRegex(ValueError, 'exact 40-character'):
                    sidecar.load_frozen(ROOT, invalid)
                load.assert_not_called()

    def test_cli_requires_valid_revision_and_forwards_it_unchanged(self):
        base = ['--repo', str(ROOT), '--out', str(self.args.out), '--name', 'base',
                '--session', 'session', '--ticket', '123', '--pid', '42', '--start', '777']
        for options in ([], ['--revision', 'main'], ['--revision', 'a' * 39],
                        ['--revision', 'g' * 40]):
            with patch.object(sidecar, 'observe') as run, patch.object(sys, 'stderr'):
                with self.assertRaises(SystemExit):
                    sidecar.main(base + options)
                run.assert_not_called()
        with patch.object(sidecar, 'observe', return_value=0) as run:
            self.assertEqual(sidecar.main(base + ['--revision', 'b' * 40]), 0)
        self.assertEqual(run.call_args.args[0].revision, 'b' * 40)

    def test_file_only_gate_requires_exact_process_reservation_and_holder(self):
        pending = self.args.fleet_dir / 'pending'
        pending.mkdir(parents=True)
        path = pending / (sidecar.sha(self.args.session.encode()) + '.json')
        row = dict(session=self.args.session, ticket=self.args.ticket, pid=self.args.pid,
                   start=self.args.start, state='queued', started_at=STARTED)
        path.write_text(json.dumps(row))
        proc = self.base / 'proc'
        (proc / '42').mkdir(parents=True)
        fields = ['S'] + ['0'] * 18 + ['777']
        stat = proc / '42/stat'
        stat.write_text('42 (supervisor) ' + ' '.join(fields))
        holder = self.args.fleet_dir / 'holder'
        holder.write_text('session|42|host|10|30|note|boot\n')
        self.assertEqual(sidecar.reservation(self.args, proc=proc)['state'], 'WAIT')
        row['state'] = 'running'; path.write_text(json.dumps(row))
        self.assertEqual(sidecar.reservation(self.args, proc=proc)['state'], 'GO')
        holder.write_text('foreign|42|host|10|30|note|boot\n')
        self.assertEqual(sidecar.reservation(self.args, proc=proc)['state'], 'WAIT')
        for key, value in (('ticket', '456'), ('pid', 43), ('start', '888'), ('session', 'other')):
            path.write_text(json.dumps(dict(row, **{key: value})))
            with self.assertRaisesRegex(ValueError, 'identity|changed'):
                sidecar.reservation(self.args, proc=proc)
        path.write_text(json.dumps(row)); fields[19] = '999'
        stat.write_text('42 (supervisor) ' + ' '.join(fields))
        with self.assertRaisesRegex(ValueError, 'stale'):
            sidecar.reservation(self.args, proc=proc)


if __name__ == '__main__':
    unittest.main()
