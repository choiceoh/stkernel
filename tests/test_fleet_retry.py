"""Explicit retries keep failed history and reuse only valid compatible evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_fleet_experiments as fixtures
import test_fleet_coalescing as coalescing
import experiments as ex
import experiment_retry as retrying
import experiment_submission as submission
import experiment_baselines as baseline


class RetryTests(unittest.TestCase):
    setUp = fixtures.SubmissionTests.setUp
    commit = fixtures.SubmissionTests.commit
    cli = fixtures.SubmissionTests.cli
    submit = fixtures.SubmissionTests.submit
    wait = fixtures.SubmissionTests.wait
    pair_context = fixtures.SubmissionTests.pair_context
    empty_commit = coalescing.CoalescingTests.empty_commit
    seed_baselines = coalescing.CoalescingTests.seed_baselines

    def store(self):
        store = ex.Store(self.jobs)
        self.addCleanup(store.db.close)
        return store

    def payload(self, **changes):
        spec = dict(kind='cpu', revision=self.sha, hypothesis='retry fixture', command=['true'])
        spec.update(changes)
        with patch.dict(os.environ, self.env, clear=True):
            return submission.Context(self.repo).payload(ex.normalize(spec, self.repo))

    def retry(self, store, source, session='owner'):
        with patch.dict(os.environ, self.env, clear=True):
            return retrying.retry(store, session, source, 'transient failure resolved', self.repo, launch=False)

    def cpu_payload(self):
        (self.repo / 'tests').mkdir(exist_ok=True)
        (self.repo / 'tests/test_retry_check.py').write_text(
            'import unittest\nclass Check(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+2,4)\n')
        self.commit()
        return self.payload(command=[sys.executable, 'tests/test_retry_check.py'])

    def test_cli_retries_saved_revision_after_agent_checkout_moves(self):
        marker = self.root / 'attempted'
        command = [sys.executable, '-c', 'from pathlib import Path; import sys; '
                   f'p=Path({str(marker)!r}); failed=not p.exists(); p.touch(); sys.exit(7 if failed else 0)']
        original = self.submit('owner', command=command)
        failed = self.wait(original['id'])
        self.assertEqual(failed['state'], 'failed')
        revision = self.sha
        self.empty_commit()
        answer = self.cli('retry', 'owner', original['id'], '--reason', 'temporary condition cleared')
        self.assertNotEqual(answer['id'], original['id'])
        self.assertEqual(answer['retry_of'], original['id'])
        result = self.wait(answer['id'])
        self.assertEqual(result['state'], 'succeeded', result)
        self.assertEqual(result['revision'], revision)
        store = self.store()
        self.assertIsNone(store.get(answer['id'])['repeat_reason'])
        self.assertEqual(store.get(original['id'])['result'], failed['result'])
        self.assertEqual(store.get(original['id'])['state'], 'failed')

    def test_historical_controller_retry_runs_pinned_current_code_with_saved_source(self):
        historical = self.repo / 'bench/experiments.py'
        historical.write_text(historical.read_text() + '\n# Historical controller revision.\n')
        self.commit()
        store = self.store(); payload = self.payload()
        payload['snapshot']['runner'] = ex.digest(historical)
        old = store.submit('owner', payload)['id']; store.state(old, 'interrupted')
        answer = self.cli('retry', 'owner', old, '--reason', 'controller interruption resolved')
        result = self.wait(answer['id'])
        self.assertEqual(result['state'], 'succeeded', result)
        current = store.get(answer['id'])['payload']
        self.assertEqual(current['spec']['revision'], self.sha)
        controller = retrying.controller_path(current)
        self.assertNotEqual(controller, Path(current['repo']))
        env = ex.child_env(current, store, answer['id'])
        self.assertEqual(env['REPO'], current['repo'])
        self.assertEqual(env['FLEET'], str(Path(current['repo']) / 'bench/fleet.sh'))
        self.assertEqual(ex.digest(Path(current['repo']) / 'bench/experiments.py'), payload['snapshot']['runner'])
        self.assertEqual(ex.digest(controller / 'bench/experiments.py'), current['snapshot']['runner'])
        self.assertNotEqual(payload['snapshot']['runner'], current['snapshot']['runner'])
        with (controller / 'bench/experiment_retry.py').open('a') as stream:
            stream.write('\n# unexpected alteration\n')
        with self.assertRaisesRegex(ValueError, 'controller changed'):
            ex.verify(current)

    def test_shared_failure_creates_one_attempt_and_keeps_each_subscription(self):
        store = self.store(); payload = self.payload()
        old = store.submit('owner', payload)['id']
        self.assertEqual(store.submit('reader', payload)['id'], old)
        store.state(old, 'failed', {'reason': 'temporary runner failure'})
        first = self.retry(store, old)
        second = self.retry(store, old, 'reader')
        self.assertEqual(first['id'], second['id'])
        self.assertEqual(second['disposition'], 'joined')
        self.assertEqual(store.get(first['id'])['payload']['spec'], payload['spec'])
        self.assertEqual(store.get(old)['result'], {'reason': 'temporary runner failure'})
        self.assertEqual(store.db.execute('SELECT count(*) FROM retry_attempts WHERE attempt=?',
                                         (first['id'],)).fetchone()[0], 2)
        self.assertEqual({row[0] for row in store.db.execute('SELECT session FROM subscribers WHERE job=?', (old,))},
                         {'owner', 'reader'})
        store.state(first['id'], 'succeeded')
        self.assertEqual(self.retry(store, old)['id'], first['id'])
        self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0], 2)

    def test_invalid_source_state_subscription_and_reason_create_nothing(self):
        store = self.store(); payload = self.payload()
        old = store.submit('owner', payload)['id']
        for state in ('queued', 'running', 'succeeded', 'incomplete'):
            store.state(old, state)
            with self.assertRaisesRegex(ValueError, 'retry requires'):
                self.retry(store, old)
        store.state(old, 'failed')
        with self.assertRaisesRegex(ValueError, 'subscription'):
            self.retry(store, old, 'stranger')
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, 'nonempty reason'):
            retrying.retry(store, 'owner', old, '  ', self.repo)
        with store.db:
            store.db.execute('INSERT INTO withdrawals VALUES(?,?,?,?)', (old, 'owner', 'replacement', 'obsolete'))
        with self.assertRaisesRegex(ValueError, 'withdrawn'):
            self.retry(store, old)
        with store.db:
            store.db.execute('DELETE FROM withdrawals')
        store.state(old, 'retired')
        with self.assertRaisesRegex(ValueError, 'retry requires'):
            self.retry(store, old)
        self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0], 1)
        self.assertEqual(store.db.execute('SELECT count(*) FROM retry_attempts').fetchone()[0], 0)

    def test_fresh_attestation_and_concurrent_withdrawal_abort_before_publication(self):
        store = self.store(); old = store.submit('owner', self.payload())['id']
        store.state(old, 'failed')
        original = submission.Context.payload; calls = []
        def changed(context, spec):
            payload = original(context, spec); calls.append(1)
            if len(calls) == 2:
                payload['snapshot'] = dict(payload['snapshot'], host='changed')
            return payload
        with patch.object(submission.Context, 'payload', new=changed), self.assertRaisesRegex(ValueError, 'changed'):
            self.retry(store, old)
        calls.clear()
        def withdrawn(context, spec):
            payload = original(context, spec); calls.append(1)
            if len(calls) == 2:
                with store.db:
                    store.db.execute('INSERT INTO withdrawals VALUES(?,?,?,?)',
                                     (old, 'owner', 'replacement', 'cancelled during attestation'))
            return payload
        with patch.object(submission.Context, 'payload', new=withdrawn), self.assertRaisesRegex(ValueError, 'withdrawn'):
            self.retry(store, old)
        self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0], 1)

    def test_failed_prerequisite_stays_failed_and_is_not_silently_retried(self):
        store = self.store()
        dependency = store.submit('gate', self.payload(command=['false']))['id']
        store.state(dependency, 'failed', {'checks': {'passed': False}})
        old = store.submit('owner', self.payload(depends_on=[dependency]))['id']
        store.state(old, 'blocked')
        with self.assertRaisesRegex(ValueError, 'prerequisite ' + dependency):
            self.retry(store, old)
        self.assertEqual(store.get(dependency)['state'], 'failed')
        self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0], 2)

    def test_valid_cpu_cache_is_reused_and_failed_cache_is_never_success(self):
        store = self.store(); payload = self.cpu_payload()
        self.assertIsNotNone(payload['cpu_identity'])
        old = store.submit('owner', payload)['id']
        store.state(old, 'failed', {'reason': 'runner interruption'})
        gate = store.submit('gate', self.payload(command=['true', 'gate']))['id']
        store.state(gate, 'succeeded')
        cached_payload = copy.deepcopy(payload)
        cached_payload['spec']['depends_on'] = [gate]
        cached = store.submit('cache-owner', cached_payload)['id']
        checks = dict(evidence='cpu-only', checks={'passed': True, 'coverage_complete': True})
        store.state(cached, 'succeeded', checks)
        with store.db:
            store.db.execute('INSERT INTO cpu_cache VALUES(?,?)', (payload['cpu_identity']['key'], cached))
        answer = self.retry(store, old)
        self.assertTrue(answer['cache_hit'])
        self.assertEqual(store.get(answer['id'])['result']['cache_source'], cached)
        self.assertFalse((self.jobs / answer['id'] / 'checkout').exists())
        self.assertIsNone(store.get(answer['id'])['repeat_reason'])
        store.state(answer['id'], 'failed')
        store.state(cached, 'failed', {'checks': {'passed': False, 'coverage_complete': True}})
        failed = self.retry(store, answer['id'])
        self.assertEqual(failed['state'], 'queued')
        self.assertNotIn('cache_hit', failed)

    def test_explicit_retry_does_not_inherit_its_failed_source_cpu_claim(self):
        store = self.store(); payload = self.cpu_payload()
        old = store.submit('owner', payload)['id']
        store.state(old, 'failed', {'reason': 'previous CPU attempt failed'})
        answer = self.retry(store, old)
        claim = tempfile.TemporaryFile(mode='w+')
        import experiment_sharing
        with patch.object(experiment_sharing, 'cpu_claim', return_value=(claim, old)):
            self.assertEqual(ex.worker(store, answer['id']), 0)
        self.assertEqual(store.get(answer['id'])['state'], 'succeeded', store.get(answer['id']))
        self.assertEqual(store.get(old)['state'], 'failed')

    def test_retry_still_blocks_on_a_different_failed_shared_cpu_owner(self):
        store = self.store(); payload = self.cpu_payload()
        old = store.submit('owner', payload)['id']; store.state(old, 'failed')
        answer = self.retry(store, old)
        other = store.submit('other', payload, repeat='distinct previous CPU check')['id']
        store.state(other, 'failed', {'checks': {'passed': False}})
        claim = tempfile.TemporaryFile(mode='w+')
        import experiment_sharing
        with patch.object(experiment_sharing, 'cpu_claim', return_value=(claim, other)):
            self.assertEqual(ex.worker(store, answer['id']), 0)
        self.assertEqual(store.get(answer['id'])['state'], 'blocked')
        self.assertEqual(store.get(answer['id'])['result']['cpu_owner'], other)

    def test_missing_failed_baseline_retries_once_only_for_explicit_attempt(self):
        store = self.store()
        payload = self.payload(kind='pair', command=[], knobs={'VLLM_TEST': '1'}, context=self.pair_context())
        old = store.submit('owner', payload)['id']
        failed_base = baseline.reserve(store, old)
        store.state(failed_base, 'failed', {'reason': 'startup failed'})
        with self.assertRaisesRegex(ValueError, 'internal baseline'):
            self.retry(store, failed_base, 'baseline-' + old)
        store.state(old, 'blocked')
        ordinary_payload = copy.deepcopy(payload)
        ordinary_payload['spec']['knobs'] = {'VLLM_TEST': '2'}
        ordinary = store.submit('ordinary', ordinary_payload)['id']
        self.assertEqual(baseline.reserve(store, ordinary), failed_base)
        retry = self.retry(store, old)
        fresh = baseline.reserve(store, retry['id'])
        self.assertNotEqual(fresh, failed_base)
        self.assertEqual(baseline.reserve(store, retry['id']), fresh)
        self.assertIsNone(store.get(fresh)['repeat_reason'])
        self.assertEqual(store.get(fresh)['state'], 'queued')
        shared = store.submit('shared', ordinary_payload, repeat='create distinct consumer')['id']
        with store.db:
            store.db.execute('UPDATE jobs SET repeat_reason=NULL WHERE id=?', (shared,))
        self.assertEqual(baseline.reserve(store, shared), fresh)
        store.state(fresh, 'failed', {'reason': 'still unavailable'})
        store.state(retry['id'], 'blocked')
        ordinary_pinned = copy.deepcopy(store.get(retry['id'])['payload'])
        ordinary_pinned['spec']['knobs'] = {'VLLM_TEST': '3'}
        ordinary_again = store.submit('ordinary-again', ordinary_pinned)['id']
        self.assertFalse(store.is_retry(ordinary_again))
        self.assertEqual(baseline.reserve(store, ordinary_again), fresh)
        next_attempt = self.retry(store, retry['id'])
        next_baseline = baseline.reserve(store, next_attempt['id'])
        self.assertNotEqual(next_baseline, fresh)
        self.assertEqual(baseline.reserve(store, next_attempt['id']), next_baseline)

    def test_retry_keeps_successful_dependencies_and_compatible_baseline_samples(self):
        store = self.store()
        gate = store.submit('gate', self.payload(command=['true', 'gate']))['id']; store.state(gate, 'succeeded')
        evaluations = [{'workload': {'ctx': [2000]}}]
        self.seed_baselines(evaluations)
        payload = self.payload(kind='pair', command=[], knobs={'VLLM_TEST': '1'}, context=self.pair_context(),
                               depends_on=[gate], evaluations=evaluations)
        old = store.submit('owner', payload)['id']; store.state(old, 'failed')
        before = (self.logs / 'onepass.jsonl').read_bytes()
        answer = self.retry(store, old)
        current = store.get(answer['id'])['payload']
        self.assertTrue(baseline.ready(current))
        self.assertEqual(current['spec']['depends_on'], [gate])
        self.assertEqual(store.get(gate)['state'], 'succeeded')
        self.assertEqual((self.logs / 'onepass.jsonl').read_bytes(), before)
        self.assertEqual(store.db.execute("SELECT count(*) FROM dependencies WHERE kind='baseline'").fetchone()[0], 0)
        processes = []; popen = subprocess.Popen
        def tracked(*args, **kwargs):
            process = popen(*args, **kwargs); processes.append(process)
            return process
        with patch.dict(os.environ, self.env, clear=True), patch.object(ex.subprocess, 'Popen', side_effect=tracked) as spawned:
            ex.ensure_worker(store, answer['id'])
        workers = [call.args[0] for call in spawned.call_args_list if isinstance(call.args[0], list) and 'worker' in call.args[0]]
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0][1], str(retrying.controller_path(current) / 'bench/experiments.py'))
        result = self.wait(answer['id'])
        self.assertEqual(result['state'], 'succeeded', result)
        for process in processes:
            process.wait(timeout=5)
        current = store.get(answer['id'])['payload']
        env = ex.child_env(current, store, answer['id'])
        self.assertEqual(env['FLEET'], str(Path(current['repo']) / 'bench/fleet.sh'))
        self.assertEqual(env['LEVER'], str(Path(current['repo']) / 'bench/ab-lever.sh'))
        self.assertNotIn('-BASE-', (self.logs / 'arms').read_text())
        self.assertEqual(store.db.execute("SELECT count(*) FROM dependencies WHERE kind='baseline'").fetchone()[0], 0)

    def test_legacy_pair_retry_keeps_its_three_sample_confirmation_policy(self):
        store = self.store()
        payload = self.payload(kind='pair', command=[], knobs={'VLLM_TEST': '1'}, context=self.pair_context())
        payload['spec'].pop('baseline_policy')
        old = store.submit('owner', payload)['id']; store.state(old, 'failed')
        answer = self.retry(store, old)
        current = store.get(answer['id'])['payload']
        self.assertEqual(current['spec']['baseline_policy'], 'confirm')
        self.assertEqual(baseline.required_samples(current['spec']), 3)


if __name__ == '__main__':
    unittest.main()
