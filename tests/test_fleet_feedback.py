"""Batch, early evidence, CPU DAG, baseline demand and actionable-result contracts."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import test_fleet_experiments as fixtures
import test_fleet_coalescing as coalescing
ROOT=fixtures.ROOT
import experiments as ex
import experiment_baselines as baseline
import experiment_plan as plan
import experiment_submission as submission
import cpu_evidence
import cpu_unittest


class FeedbackTests(unittest.TestCase):
    setUp=fixtures.SubmissionTests.setUp
    commit=fixtures.SubmissionTests.commit
    cli=fixtures.SubmissionTests.cli
    submit=fixtures.SubmissionTests.submit
    wait=fixtures.SubmissionTests.wait
    refresh_deployed_fixture=fixtures.SubmissionTests.refresh_deployed_fixture
    pair_context=fixtures.SubmissionTests.pair_context
    empty_commit=coalescing.CoalescingTests.empty_commit
    wait_state=coalescing.CoalescingTests.wait_state
    manual_job=coalescing.CoalescingTests.manual_job
    contract_sources=coalescing.CoalescingTests.contract_sources

    def store(self):
        store=ex.Store(self.jobs)
        self.addCleanup(store.db.close)
        return store

    def manifest(self,**changes):
        spec=dict(kind='cpu',revision=self.sha,hypothesis='feedback test',command=['true'])
        spec.update(changes)
        return spec

    def test_cache_hit_needs_no_new_checkout_and_keeps_own_gates(self):
        (self.repo/'tests').mkdir()
        (self.repo/'tests/test_fast.py').write_text('import unittest\nclass C(unittest.TestCase):\n def test_ok(self): self.assertEqual(2+2,4)\n')
        self.commit()
        command=[sys.executable,'tests/test_fast.py']
        first=self.submit('first',command=command)
        self.assertEqual(self.wait(first['id'])['state'],'succeeded')
        self.empty_commit()
        second=self.submit('second',command=command)
        self.assertTrue(second['cache_hit'])
        self.assertEqual(second['state'],'succeeded')
        self.assertFalse((self.jobs/second['id']/'checkout').exists())
        result=self.cli('result',second['id'])
        self.assertEqual(result['result']['cache_source'],first['id'])
        prerequisite=self.submit('bad',command=['false'])
        self.assertEqual(self.wait(prerequisite['id'])['state'],'failed')
        blocked=self.submit('blocked',command=command,depends_on=[prerequisite['id']])
        self.assertEqual(self.wait(blocked['id'])['state'],'blocked')

    def test_batch_fingerprints_runtime_twice_total_and_preserves_distinct_contracts(self):
        store=self.store();calls=[];original=subprocess.run
        def record(argv,*args,**kwargs):
            if len(argv)>2 and argv[1]=='-c' and 'importlib.metadata' in argv[2]:calls.append(argv)
            return original(argv,*args,**kwargs)
        requests=[dict(name=name,manifest=self.manifest(command=['python3','bench/cpu_checks.py','--contract',name])) for name in ('math','layout','dispatch')]
        with patch.dict(os.environ,self.env,clear=True),patch.object(subprocess,'run',side_effect=record):
            answers=submission.submit_many(store,'batch',requests,self.repo,launch=False)['requests']
        self.assertEqual(len(calls),2)
        self.assertEqual(len({a['id'] for a in answers}),3)
        self.assertEqual(len({store.get(a['id'])['payload']['cpu_identity']['key'] for a in answers}),3)
        self.assertTrue(all(not (self.jobs/a['id']/'checkout').exists() for a in answers))

    def test_invalid_batch_or_changed_attestation_creates_no_jobs(self):
        store=self.store()
        requests=[dict(name='first',manifest=self.manifest()),dict(name='bad',manifest=self.manifest(),requires=['future'])]
        with self.assertRaisesRegex(ValueError,'earlier'):
            submission.submit_many(store,'batch',requests,self.repo)
        self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0],0)
        rollback=[dict(name='first',manifest=self.manifest()),
                  dict(name='bad',manifest=self.manifest(depends_on=['missing-id']))]
        with patch.dict(os.environ,self.env,clear=True),self.assertRaisesRegex(ValueError,'unknown experiment'):
            submission.submit_many(store,'batch',rollback,self.repo)
        self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0],0)
        original=submission.Context.payload;calls=[]
        def changed(context,spec):
            value=original(context,spec);calls.append(1)
            if len(calls)==2:value['snapshot']=dict(value['snapshot'],host='changed-host')
            return value
        with patch.dict(os.environ,self.env,clear=True),patch.object(submission.Context,'payload',new=changed),self.assertRaisesRegex(ValueError,'changed'):
            submission.submit_many(store,'batch',requests[:1],self.repo)
        self.assertEqual(store.db.execute('SELECT count(*) FROM jobs').fetchone()[0],0)

    def test_batch_dependency_failure_returns_actionable_explanation(self):
        marker=self.root/'must-not-run'
        requests=[dict(name='gate',manifest=self.manifest(command=['false'])),
            dict(name='consumer',manifest=self.manifest(command=[sys.executable,'-c',f'open({str(marker)!r},"w").close()']),requires=['gate'])]
        path=self.root/'batch.json';path.write_text(json.dumps(requests))
        answers=self.cli('batch','batch',str(path))['requests']
        result=self.wait(answers[1]['id'])
        self.assertEqual(result['state'],'blocked')
        self.assertFalse(marker.exists())
        self.assertEqual(result['explanation']['blocking_dependencies'][0]['id'],answers[0]['id'])
        self.assertEqual(result['explanation']['next_actions'][0]['action'],'inspect_dependency')
        events=self.cli('inbox','batch')['events']
        self.assertTrue(any(e.get('explanation',{}).get('state')=='blocked' for e in events))
        self.assertFalse(result['explanation']['automatic_retry'])

    def test_split_plan_keeps_all_deployment_components_and_cpu_budget(self):
        value=plan.build(dict(hypothesis='all gates',knobs={'VLLM_TEST':'1'},context=self.pair_context(),cpu_suites=['logic','startup'],cpu_jobs=2),self.repo)
        stages={s['name']:s for s in value['stages']}
        self.assertEqual(set(stages),{'checks-core','checks-fleet','checks-startup','gpu'})
        self.assertEqual(stages['checks-fleet']['manifest']['resources']['cpu_slots'],2)
        self.assertEqual(set(stages['gpu']['requires']),set(stages)-{'gpu'})
        self.assertTrue(all(not stages[n]['requires'] for n in stages if n!='gpu'))

    def test_cpu_results_and_ids_are_available_before_gpu_attestation(self):
        (self.repo/'tests').mkdir()
        (self.repo/'tests/test_early.py').write_text('import unittest\nclass C(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n')
        self.refresh_deployed_fixture()
        raw=dict(hypothesis='early CPU feedback',knobs={'VLLM_TEST':'1'},context=self.pair_context(),
                 objective={'metric':'quality'},cpu_suites=[],cpu_tests=['tests/test_early.py'],
                 prepare=[dict(command=[sys.executable,'-c','print("prepared")'],requires=['checks'])])
        path=self.root/'early-plan.json';path.write_text(json.dumps(raw))
        args=SimpleNamespace(manifest=path,base=None,submit=True,prepare_only=False,session='early',supersedes=[])
        store=self.store();original=ex.snapshot;observed=[]
        def attestation(repo,spec,stamp):
            if spec['kind']=='pair':
                saved=json.loads(next((store.root/'plans').glob('*/plan.json')).read_text())
                stages={s['name']:s for s in saved['stages']}
                ids=[stages[name]['submission']['id'] for name in ('checks','prepare-1')]
                self.assertNotIn('submission',stages['gpu'])
                self.assertEqual(stages['gpu']['manifest']['depends_on'],sorted(ids))
                self.assertTrue(stages['gpu']['dependencies_resolved'])
                for name in ('checks','prepare-1'):
                    stage=stages[name]
                    self.assertEqual(json.loads(Path(stage['path']).read_text()),stage['manifest'])
                    self.assertEqual(self.wait(stage['submission']['id'])['state'],'succeeded')
                self.assertFalse((self.logs/'arms').exists())
                observed.append(ids)
            return original(repo,spec,stamp)
        with patch.dict(os.environ,self.env,clear=True),patch.object(ex,'snapshot',side_effect=attestation):
            value=plan.run(args,store,self.repo)
        self.assertNotIn('error',value)
        self.assertEqual(len(observed),2)  # Both fresh deployment attestations still run.
        gpu=value['stages'][-1]
        self.assertEqual(gpu['manifest']['depends_on'],sorted(observed[0]))
        self.assertFalse(any(d.startswith('pending-stage:') for d in gpu['manifest']['depends_on']))
        self.assertEqual(self.wait(gpu['submission']['id'])['state'],'succeeded')
        self.assertEqual((self.logs/'arms').read_text().count('onepass'),2)

    def test_preparation_runs_while_checks_wait_but_gpu_remains_blocked(self):
        release=self.root/'release'
        (self.repo/'tests').mkdir()
        (self.repo/'tests/test_gate.py').write_text('import unittest,time\nfrom pathlib import Path\nclass C(unittest.TestCase):\n def test_gate(self):\n  deadline=time.monotonic()+10\n'
            f'  while not Path({str(release)!r}).exists():\n   assert time.monotonic()<deadline\n   time.sleep(.01)\n'
            '  self.fail("deliberate contract failure")\n')
        self.refresh_deployed_fixture()
        raw=dict(hypothesis='parallel CPU preparation',knobs={'VLLM_TEST':'1'},context=self.pair_context(),cpu_suites=[],cpu_tests=['tests/test_gate.py'],
            prepare=[dict(command=[sys.executable,'-c','from pathlib import Path;Path("build").mkdir(exist_ok=True);Path("build/a").write_text("artifact")'],outputs=['build/a'])])
        path=self.root/'plan.json';path.write_text(json.dumps(raw))
        value=self.cli('plan','parallel',str(path),'--submit');stages={s['name']:s for s in value['stages']}
        self.assertEqual(self.wait(stages['prepare-1']['submission']['id'])['state'],'succeeded')
        self.assertFalse((self.logs/'arms').exists())
        self.assertEqual(self.cli('result',stages['gpu']['submission']['id'])['state'],'waiting_dependencies')
        release.touch()
        failed=self.wait(stages['checks']['submission']['id'])
        self.assertEqual(failed['state'],'failed')
        self.assertTrue(failed['explanation']['failed_checks'][0]['failures'])
        self.assertEqual(failed['explanation']['next_actions'][0]['action'],'reproduce_cpu_failure')
        self.assertEqual(self.wait(stages['gpu']['submission']['id'])['state'],'blocked')
        self.assertFalse((self.logs/'arms').exists())

    def test_explicit_preparation_edge_transfers_verified_outputs(self):
        (self.repo/'tests').mkdir()
        (self.repo/'tests/test_gate.py').write_text('import unittest\nclass C(unittest.TestCase):\n def test_gate(self): self.assertTrue(True)\n')
        self.commit()
        raw=dict(hypothesis='CPU build DAG',knobs={'VLLM_TEST':'1'},context=self.pair_context(),cpu_suites=[],cpu_tests=['tests/test_gate.py'],prepare=[
            dict(command=[sys.executable,'-c','from pathlib import Path;Path("build").mkdir(exist_ok=True);Path("build/a").write_text("artifact")'],outputs=['build/a']),
            dict(command=[sys.executable,'-c','from pathlib import Path;Path("build/b").write_text(Path("build/a").read_text()+" checked")'],outputs=['build/b'],requires=['prepare-1'])])
        path=self.root/'plan.json';path.write_text(json.dumps(raw))
        value=self.cli('plan','dag',str(path),'--prepare-only')
        for stage in value['stages'][:-1]:self.assertEqual(self.wait(stage['submission']['id'])['state'],'succeeded')
        last=self.cli('result',value['stages'][-2]['submission']['id'])
        self.assertEqual(Path(last['result']['artifacts'][0]['path']).read_text(),'artifact checked')

    def test_baseline_objectives_share_required_samples_without_extra_boots(self):
        jobs=[self.submit(str(i),kind='pair',command=[],knobs={'VLLM_TEST':'1'},context=self.pair_context(),
            evaluations=[dict(objective={'metric':metric})]) for i,metric in enumerate(('quality','decode_steps'))]
        for job in jobs:self.assertEqual(self.wait(job['id'])['state'],'succeeded')
        store=self.store()
        owners=[store.db.execute("SELECT dependency FROM dependencies WHERE job=? AND kind='baseline'",(j['id'],)).fetchone()[0] for j in jobs]
        self.assertEqual(owners[0],owners[1])
        base=store.get(owners[0]);self.assertEqual(len(baseline.samples(base['payload'],1)),3)
        self.assertEqual((self.logs/'arms').read_text().count('onepass'),4)

    def test_baseline_cannot_expand_after_start_but_accepts_covered_quality(self):
        store=self.store()
        first=self.manual_job(store,'first',kind='pair',evaluations=[dict(objective={'metric':'decode_steps'})])
        owner=baseline.reserve(store,first);store.state(owner,'running')
        covered=self.manual_job(store,'quality',kind='pair',evaluations=[dict(objective={'metric':'quality'})])
        self.assertEqual(baseline.reserve(store,covered),owner)
        late=self.manual_job(store,'late',kind='pair',evaluations=[dict(objective={'metric':'decode_steps'},workload={'ctx':[4000]})])
        self.assertNotEqual(baseline.reserve(store,late),owner)
        self.assertEqual(len(baseline.planned(store,owner,store.get(owner)['payload'])['spec']['evaluations']),1)

    def test_fleet_cache_ignores_unrelated_kernel_and_fails_closed_on_test_edits(self):
        self.contract_sources()
        for source in (ROOT/'tests').glob('test_fleet*.py'):shutil.copyfile(source,self.repo/'tests'/source.name)
        unrelated=self.repo/'overlay/modules/other/unrelated.py';unrelated.parent.mkdir(parents=True);unrelated.write_text('value=1\n');self.commit()
        spec=ex.normalize(self.manifest(command=['python3','bench/cpu_checks.py','--suite','fleet']),self.repo)
        one=cpu_evidence.identity(self.repo,spec,{})
        self.assertEqual(one['scope'],'audited-fleet')
        unrelated.write_text('value=2\n');self.commit()
        self.assertEqual(cpu_evidence.identity(self.repo,spec,{}),one)
        edited=self.repo/'tests/test_fleet_feedback.py';edited.write_text(edited.read_text()+'\n# changed audit\n');self.commit()
        self.assertEqual(cpu_evidence.identity(self.repo,spec,{})['scope'],'full-tree')


class ParallelRunnerTests(unittest.TestCase):
    def test_shards_count_all_tests_and_keep_failures_and_skips(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,FLEET_CPU_SLOTS='2'):
            root=Path(directory);(root/'tests').mkdir()
            (root/'tests/test_fleet_shard_fixture.py').write_text('import unittest\nclass C(unittest.TestCase):\n def test_a(self): self.assertTrue(True)\n def test_b(self): self.fail("detected")\n @unittest.skip("missing dependency")\n def test_c(self): pass\n def test_d(self): self.assertTrue(True)\n')
            try:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    report=cpu_unittest.execute('tests/test_fleet*.py',root,jobs=2)
            finally:sys.modules.pop('test_fleet_shard_fixture',None)
            self.assertEqual(report['tests_run'],4)
            self.assertEqual(len(set(report['test_ids'])),4)
            self.assertEqual(report['failures'],1)
            self.assertEqual(report['skipped'],['missing dependency'])
            self.assertFalse(report['passed']);self.assertFalse(report['coverage_complete'])
            with self.assertRaisesRegex(ValueError,'slots'):
                cpu_unittest.execute('tests/test_fleet*.py',root,jobs=3)
            sys.modules.pop('test_fleet_shard_fixture',None)
