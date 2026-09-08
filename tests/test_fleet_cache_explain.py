"""CPU cache explanations use recorded fingerprints without rerunning work."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import experiments
import experiment_explain as explain


def identity(source='source-a', environment='env-a'):
    parts = dict(dependencies={'overlay/kernel.py':source}, components={'environment':environment})
    return dict(parts, key=hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(),
                scope='audited-logic', files=1)


class CacheExplainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = experiments.Store(self.tmp.name)
        self.addCleanup(self.store.db.close)
        self.sequence = 0

    def add(self, *, cpu_identity=None, state='succeeded', revision=None, result=None,
            command=None, kind='cpu', repeat=None, finished=True):
        self.sequence += 1
        job = 'fixture-' + str(self.sequence)
        spec = dict(kind=kind, command=command or ['python3', 'bench/cpu_checks.py', '--suite', 'logic'],
                    revision=revision or 'revision-' + str(self.sequence), depends_on=[])
        payload = dict(spec=spec, cpu_identity=cpu_identity, repo='/fixture/source')
        if result is None:
            result = dict(checks=dict(passed=True, coverage_complete=True)) if state == 'succeeded' else {}
        self.store.db.execute('INSERT INTO jobs(id,fingerprint,payload,state,created,finished,result,repeat_reason) '
                              'VALUES(?,?,?,?,?,?,?,?)',
                              (job, job, json.dumps(payload), state, self.sequence * 10,
                               self.sequence * 10 + 1 if finished else None, json.dumps(result), repeat))
        self.store.db.commit()
        return self.store.get(job)

    def test_same_identity_after_documentation_revision_is_only_a_match(self):
        old = self.add(cpu_identity=identity(), revision='before-docs')
        current = self.add(cpu_identity=identity(), revision='after-docs', state='queued')
        answer = explain.explain(self.store, current)['cache_reuse']
        self.assertEqual(answer['status'], 'identity_match')
        self.assertEqual(answer['source_id'], old['id'])
        self.assertEqual(answer['changed_paths'], [])
        self.assertIn('no reuse was recorded', answer['reason'])

    def test_source_and_environment_change_names_are_shown_without_values(self):
        old = self.add(cpu_identity=identity())
        current = self.add(cpu_identity=identity(source='source-b', environment='secret-value'), state='queued')
        current['payload']['environment'] = {'TOKEN':'raw-secret-not-for-output'}
        answer = explain.explain(self.store, current)['cache_reuse']
        self.assertEqual(answer['status'], 'changed')
        self.assertEqual(answer['source_id'], old['id'])
        self.assertEqual(answer['changed_paths'], ['overlay/kernel.py'])
        self.assertEqual(answer['changed_components'], ['environment'])
        self.assertNotIn('secret', json.dumps(answer))

    def test_legacy_or_missing_identities_remain_unknown(self):
        old = self.add(cpu_identity={'key':'old-opaque-key', 'scope':'full-tree'})
        current = self.add(cpu_identity=identity(), state='queued')
        answer = explain.cpu_cache_reuse(self.store, current)
        self.assertEqual(answer['status'], 'unknown')
        self.assertEqual(answer['source_id'], old['id'])
        current['payload']['cpu_identity'] = None
        self.assertEqual(explain.cpu_cache_reuse(self.store, current)['status'], 'unknown')
        current['payload']['cpu_identity'] = {'key':'opaque-new-key'}
        self.assertEqual(explain.cpu_cache_reuse(self.store, current)['status'], 'unknown')

    def test_actual_cache_hit_is_reported_from_recorded_provenance(self):
        old = self.add(cpu_identity=None)
        current = self.add(cpu_identity=None, result={'cache_source':old['id']})
        answer = explain.cpu_cache_reuse(self.store, current)
        self.assertEqual(answer['status'], 'cached')
        self.assertEqual(answer['source_id'], old['id'])
        self.assertEqual(answer['changed_components'], [])

    def test_independent_sample_is_not_claimed_as_cache_failure(self):
        self.add(cpu_identity=identity())
        current = self.add(cpu_identity=identity(), state='queued', repeat='user requested extra sample')
        answer = explain.cpu_cache_reuse(self.store, current)
        self.assertEqual(answer['status'], 'identity_match')
        self.assertEqual(answer['reason'], 'An independent sample was requested')

    def test_history_is_bounded_and_unrelated_families_are_not_compared(self):
        self.add(cpu_identity=identity())
        for _ in range(explain.CACHE_HISTORY_LIMIT):
            self.add(cpu_identity=identity(), command=['python3', 'bench/cpu_checks.py', '--suite', 'startup'])
        current = self.add(cpu_identity=identity(), state='queued')
        answer = explain.cpu_cache_reuse(self.store, current)
        self.assertEqual(answer['status'], 'unknown')
        self.assertEqual(answer['search_limit'], 50)

    def test_failed_incomplete_and_later_evidence_cannot_explain_prior_reuse(self):
        self.add(cpu_identity=identity(), state='failed')
        self.add(cpu_identity=identity(), result={'checks':{'passed':True, 'coverage_complete':False}})
        self.add(cpu_identity=identity(), finished=False)
        current = self.add(cpu_identity=identity(), state='queued')
        self.add(cpu_identity=identity())
        self.assertEqual(explain.cpu_cache_reuse(self.store, current)['status'], 'unknown')

    def test_explanation_is_read_only_and_gpu_output_is_unchanged(self):
        self.add(cpu_identity=identity())
        current = self.add(cpu_identity=identity(source='changed'), state='queued')
        gpu = copy.deepcopy(current)
        gpu['payload']['spec']['kind'] = 'pair'
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
        self.store.db.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        try:
            with mock.patch('subprocess.run', side_effect=AssertionError('must not execute commands')):
                self.assertEqual(explain.explain(self.store, current)['cache_reuse']['status'], 'changed')
                self.assertNotIn('cache_reuse', explain.explain(self.store, gpu))
        finally:
            self.store.db.set_authorizer(None)


if __name__ == '__main__':
    unittest.main()
