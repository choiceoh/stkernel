"""Coalescing, supersession, timing and CPU fault-detection acceptance without GPUs."""
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import test_fleet_experiments as fixtures
ROOT = fixtures.ROOT
sys.path.insert(0,str(ROOT/'bench'))
import experiments as ex
import experiment_baselines as baselines
import experiment_groups as groups
import experiment_metrics as metrics
import experiment_retirement as retirement
import cpu_contracts
import cpu_evidence
from measurement_contract import workload


class CoalescingTests(unittest.TestCase):
    setUp = fixtures.SubmissionTests.setUp
    commit = fixtures.SubmissionTests.commit
    cli = fixtures.SubmissionTests.cli
    submit = fixtures.SubmissionTests.submit
    wait = fixtures.SubmissionTests.wait
    refresh_deployed_fixture = fixtures.SubmissionTests.refresh_deployed_fixture
    pair_context = fixtures.SubmissionTests.pair_context

    def slow_test(self, fail=False):
        (self.repo/'tests').mkdir()
        output = self.root/'executions'
        self.release = self.root/'release'
        (self.repo/'tests/test_shared.py').write_text(
            'import time,unittest\nfrom pathlib import Path\nclass Contract(unittest.TestCase):\n'
            f' def test_result(self):\n  with open({str(output)!r},"a") as stream: stream.write("run\\n")\n'
            f'  deadline=time.monotonic()+10\n  while not Path({str(self.release)!r}).exists():\n   assert time.monotonic()<deadline, \"owner release was never signaled\"\n   time.sleep(.01)\n'+('  self.fail("contract failure")\n' if fail else '  self.assertEqual(6*7,42)\n'))
        self.commit()
        return output,[sys.executable,'tests/test_shared.py']

    def empty_commit(self):
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=T','-c','user.email=t@invalid',
                        'commit','--allow-empty','-qm','same tested content'],check=True)
        self.sha = ex.git(self.repo,'rev-parse','HEAD')

    def wait_state(self, job, state):
        store = ex.Store(self.jobs)
        deadline = time.monotonic()+6
        while time.monotonic()<deadline:
            row = store.get(job)
            if row['state'] == state:
                return row
            time.sleep(.01)
        self.fail(str(row))

    def test_same_content_different_commits_share_in_flight_cpu(self):
        output,command = self.slow_test()
        first = self.submit('first',command=command)
        self.wait_state(first['id'],'running')
        original = self.sha
        self.empty_commit()
        second = self.submit('second',command=command)
        self.wait_state(second['id'],'waiting_cpu_evidence')
        self.release.touch()
        result = self.wait(second['id'])
        self.assertNotEqual(first['id'],second['id'])
        self.assertEqual(result['state'],'succeeded',result)
        self.assertEqual(result['result']['cache_source'],first['id'])
        self.assertEqual(result['result']['tested_revision'],original)
        self.assertEqual(result['result']['revision'],self.sha)
        self.assertEqual(output.read_text(),'run\n')
        self.assertFalse((self.fleet/'holder').exists())
        self.assertTrue(any(e['event']=='waiting_cpu_evidence' for e in self.cli('inbox','second')['events']))

    def test_failed_shared_cpu_blocks_followers_without_repeating_the_failure(self):
        output,command = self.slow_test(True)
        first = self.submit('first',command=command)
        self.wait_state(first['id'],'running')
        self.empty_commit()
        second = self.submit('second',command=command)
        self.wait_state(second['id'],'waiting_cpu_evidence')
        self.release.touch()
        result = self.wait(second['id'])
        self.assertEqual(result['state'],'blocked',result)
        self.assertEqual(result['result']['source_state'],'failed')
        self.assertEqual(output.read_text(),'run\n')
        self.assertEqual(self.wait(first['id'])['state'],'failed')

    def test_cpu_claim_survives_supervisor_exit(self):
        output,command = self.slow_test()
        first = self.submit('first',command=command)
        row = self.wait_state(first['id'],'running')
        os.kill(row['worker_pid'],signal.SIGKILL)
        self.empty_commit()
        second = self.submit('second',command=command)
        self.wait_state(second['id'],'waiting_cpu_evidence')
        self.release.touch()
        result = self.wait(second['id'])
        self.assertEqual(result['state'],'succeeded',result)
        self.assertEqual(result['result']['cache_source'],first['id'])
        self.assertEqual(output.read_text(),'run\n')

    def manual_job(self, store, name, kind='cpu', **changes):
        spec = dict(kind=kind,revision=self.sha,hypothesis=name,
                    command=['true',name] if kind=='cpu' else [],
                    knobs={'VLLM_TEST':'1'} if kind=='pair' else {},
                    context={} if kind=='cpu' else self.pair_context())
        spec.update(changes)
        spec = ex.normalize(spec,self.repo)
        payload = dict(spec=spec,repo=str(self.repo),environment={},snapshot={'build':'a'*64,'host':'fixture'},
                       bash=fixtures.BASH,paths={'FLEET_DIR':str(self.fleet),'ONEPASS_JSONL':str(self.logs/'onepass.jsonl')})
        return store.submit(name,payload)['id']

    def seed_baselines(self, evaluations):
        records=[]
        for evaluation in evaluations:
            work=workload(evaluation.get('workload'))
            for index in range(3):
                records.append(fixtures.record(f'BASE-{work["ctx"]}-{index}',git=self.sha,
                    overlay=self.stamp.read_text()[:12],runtime=self.pair_context(),workload=work,
                    prefill=[dict(ctx=c,cold_s=1) for c in work['ctx']]))
        (self.logs/'onepass.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_ready_agents_share_a_boot_and_keep_original_record_binding(self):
        evals=[dict(objective={'metric':'prefill_ttft','ctx':2000},workload={'ctx':[2000]}),
               dict(objective={'metric':'decode_steps'},workload={'ctx':[32000]})]
        self.seed_baselines(evals)
        first=self.submit('first',kind='pair',command=[],knobs={'VLLM_TEST':'1'},context=self.pair_context(),evaluations=[evals[0]])
        second=self.submit('second',kind='pair',command=[],knobs={'VLLM_TEST':'1'},context=self.pair_context(),evaluations=[evals[1]])
        a,b=self.wait(first['id']),self.wait(second['id'])
        self.assertEqual(a['state'],'succeeded',a)
        self.assertEqual(b['state'],'succeeded',b)
        self.assertEqual(a['result']['execution_job'],b['result']['execution_job'])
        self.assertEqual(a['result']['candidate']['boot_id'],b['result']['candidate']['boot_id'])
        self.assertEqual(b['result']['candidate']['experiment_id'],a['result']['execution_job'])
        self.assertEqual((self.logs/'arms').read_text().count('onepass'),1)
        self.assertEqual(self.cli('stats')['shared_boot_members'],1)

    def test_group_seal_separates_late_or_incompatible_requests(self):
        store=ex.Store(self.jobs)
        first=self.manual_job(store,'first',kind='pair',evaluations=[{'workload':{'ctx':[2000]}}])
        second=self.manual_job(store,'second',kind='pair',evaluations=[{'workload':{'ctx':[4000]}}])
        self.assertEqual(groups.register(store,first),first)
        self.assertEqual(groups.register(store,second),first)
        sealed=groups.seal(store,first,store.get(first)['payload'])
        self.assertEqual(len(sealed['spec']['evaluations']),2)
        later=self.manual_job(store,'later',kind='pair',evaluations=[{'workload':{'ctx':[8000]}}])
        self.assertEqual(groups.register(store,later),later)
        changed=self.manual_job(store,'changed',kind='pair',knobs={'VLLM_TEST':'2'})
        self.assertEqual(groups.register(store,changed),changed)

    def test_group_capacity_and_explicit_repeat_preserve_independent_execution(self):
        store=ex.Store(self.jobs)
        leader=None
        for index in range(7):
            job=self.manual_job(store,str(index),kind='pair',evaluations=[{'workload':{'ctx':[2000+index]}}])
            owner=groups.register(store,job)
            if leader is None:leader=owner
            self.assertEqual(owner==leader,index<6)
        repeat=store.submit('repeat',store.get(leader)['payload'],repeat='independent sample')['id']
        self.assertEqual(groups.register(store,repeat),repeat)

    def test_shared_boot_failure_reaches_every_consumer(self):
        evals=[{'objective':{'metric':'quality'},'workload':{'ctx':[c]}} for c in (2000,4000)]
        onepass=self.repo/'bench/onepass.py'
        onepass.write_text(fixtures.FAKE_ONEPASS.replace("knobs = served['knobs']",
            "knobs = served['knobs']\n if knobs and name.endswith('-E2'): raise SystemExit(7)"))
        self.refresh_deployed_fixture();self.seed_baselines(evals)
        jobs=[self.submit(str(i),kind='pair',command=[],knobs={'VLLM_TEST':'1'},context=self.pair_context(),evaluations=[e]) for i,e in enumerate(evals)]
        self.assertEqual([self.wait(j['id'])['state'] for j in jobs],['failed','failed'])
        self.assertEqual((self.logs/'arms').read_text().count('onepass'),1)

    def test_only_missing_baseline_workloads_are_measured(self):
        store=ex.Store(self.jobs)
        job=self.manual_job(store,'baseline-owner',kind='pair',evaluations=[{'workload':{'ctx':[2000]}},{'workload':{'ctx':[32000]}}])
        payload=store.get(job)['payload'];payload['baseline_samples']=3
        rows=[[{'boot_id':f'a{i}'} for i in range(3)],[{'boot_id':f'b{i}'} for i in range(2)]]
        calls=[]
        def measure(store,job,payload,name,knobs,work_indices):
            calls.append(work_indices)
            for i in work_indices:rows[i].append({'boot_id':name})
            return [{'boot_id':name} for _ in work_indices]
        with patch.object(baselines,'samples',side_effect=lambda p,i=0:rows[i]),patch('serving_group.measure',side_effect=measure):
            self.assertEqual(baselines.run(store,job,payload)[0],'succeeded')
        self.assertEqual(calls,[[1]])
        self.assertEqual([len(r) for r in rows],[3,3])

    def test_withdrawal_keeps_other_subscribers_and_preserves_live_holder(self):
        store=ex.Store(self.jobs)
        old=self.manual_job(store,'first')
        store.submit('second',store.get(old)['payload'])
        new=self.manual_job(store,'first',command=['true','replacement'])
        store.submit('second',store.get(new)['payload'])
        (self.fleet/'holder').write_text('other-live-holder|123|fixture\n')
        (self.fleet/'queue').write_text(f'1|exp-{old}|0|1|old|boot|123\n2|other|0|1|keep|boot|456\n')
        self.assertFalse(retirement.retire(store,'first',old,new,'new revision')['retired'])
        self.assertTrue(retirement.retire(store,'second',old,new,'new revision')['retired'])
        self.assertEqual(store.get(old)['state'],'retired')
        self.assertNotIn('exp-'+old,(self.fleet/'queue').read_text())
        self.assertIn('other',(self.fleet/'queue').read_text())
        self.assertEqual((self.fleet/'holder').read_text(),'other-live-holder|123|fixture\n')
        self.assertEqual(ex.execute(store,old),0)

    def test_dependencies_and_started_work_cannot_be_retired(self):
        store=ex.Store(self.jobs)
        old=self.manual_job(store,'owner')
        new=self.manual_job(store,'owner',command=['true','dependent'],depends_on=[old])
        answer=retirement.retire(store,'owner',old,new,'still required')
        self.assertFalse(answer['retired']);self.assertEqual(answer['dependents'],[new])
        running=self.manual_job(store,'active',command=['true','running'])
        replacement=self.manual_job(store,'active',command=['true','next'])
        store.state(running,'running')
        self.assertFalse(retirement.retire(store,'active',running,replacement,'newer request')['retired'])
        self.assertEqual(store.get(running)['state'],'running')
        with self.assertRaises(ValueError):
            retirement.retire(store,'stranger',running,replacement,'not a subscriber')

    def test_superseding_submission_does_not_launch_retired_job(self):
        store=ex.Store(self.jobs)
        old=self.manual_job(store,'owner')
        path=self.root/'replacement.json'
        path.write_text(json.dumps(dict(kind='cpu',revision=self.sha,hypothesis='replacement',command=['true'])))
        new=self.cli('submit','owner',str(path),'--supersedes',old)
        self.assertTrue(new['superseded'][0]['retired'])
        self.assertEqual(self.wait(new['id'])['state'],'succeeded')
        ex.ensure_worker(store,old)
        self.assertFalse((self.jobs/old/'run.log').exists())
        self.assertEqual(store.get(old)['state'],'retired')

    def test_retired_request_resubmission_creates_new_work(self):
        store=ex.Store(self.jobs)
        old=self.manual_job(store,'owner')
        new=self.manual_job(store,'owner',command=['true','replacement'])
        retirement.retire(store,'owner',old,new,'obsolete')
        again=store.submit('owner',store.get(old)['payload'])
        self.assertNotEqual(again['id'],old)
        self.assertEqual(again['state'],'queued')

    def test_state_transition_does_not_commit_partial_retirement(self):
        store=ex.Store(self.jobs)
        old=self.manual_job(store,'owner')
        new=self.manual_job(store,'owner',command=['true','replacement'])
        event=store.event
        def fail_event(job,kind,data):
            if kind=='supersession':raise ValueError('transaction interrupted')
            event(job,kind,data)
        with patch.object(store,'event',side_effect=fail_event),self.assertRaises(ValueError):
            retirement.retire(store,'owner',old,new,'obsolete')
        self.assertEqual(store.get(old)['state'],'queued')
        self.assertEqual(store.db.execute('SELECT count(*) FROM withdrawals').fetchone()[0],0)

    def test_retired_cpu_leaves_resource_wait_without_starting(self):
        from experiment_resources import run_cpu
        store=ex.Store(self.jobs)
        old=self.manual_job(store,'owner')
        new=self.manual_job(store,'owner',command=['true','replacement'])
        def exhausted(*args):
            retirement.retire(store,'owner',old,new,'obsolete')
            return False
        with patch('experiment_resources.acquire',side_effect=exhausted),patch('subprocess.Popen') as launch:
            with self.assertRaises(ex.RetiredJob):
                run_cpu(store,old,['true'],store.get(old)['payload'])
            launch.assert_not_called()

    def test_shared_member_cannot_publish_changed_inputs(self):
        store=ex.Store(self.jobs)
        first=self.manual_job(store,'owner',kind='pair')
        second=self.manual_job(store,'follower',kind='pair',evaluations=[{'objective':{'metric':'quality'}}])
        groups.register(store,first);self.assertEqual(groups.register(store,second),first)
        with patch.object(ex,'verify',side_effect=ValueError('prepared artifacts changed')),patch.object(ex,'pair_result') as judge:
            groups.publish(store,first,'succeeded',{})
        judge.assert_not_called()
        self.assertEqual(store.get(second)['state'],'failed')

    def test_managed_wait_exits_after_retirement_without_requeue(self):
        store=ex.Store(self.jobs)
        old=self.manual_job(store,'owner')
        new=self.manual_job(store,'owner',command=['true','next'])
        retirement.retire(store,'owner',old,new,'obsolete')
        (self.fleet/'queue').write_text(f'1|exp-{old}|0|1|old|boot|{os.getpid()}\n')
        env=dict(self.env,FLEET_EXPERIMENT_ID=old,FLEET_PID=str(os.getpid()))
        p=subprocess.run([fixtures.BASH,str(ROOT/'bench/fleet.sh'),'wait','exp-'+old,'1'],env=env,text=True,capture_output=True,timeout=5)
        self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertNotIn(old,(self.fleet/'queue').read_text())
        self.assertFalse((self.fleet/'holder').exists())

    def contract_sources(self):
        paths=['tests/test_logic.py',*[v[1] for v in cpu_contracts.CONTRACTS.values()]]
        for path in set(paths):
            target=self.repo/path;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/path,target)
        self.commit()

    def test_contract_cache_reuses_unrelated_changes_and_invalidates_inputs(self):
        self.contract_sources()
        spec=ex.normalize(dict(kind='cpu',revision=self.sha,hypothesis='math',
                               command=[sys.executable,'bench/cpu_checks.py','--contract','math']),self.repo)
        env={k:v for k,v in self.env.items() if k in ex.BASE_ENV}
        first=cpu_evidence.identity(self.repo,spec,env)
        self.assertEqual(first['scope'],'audited-contracts')
        (self.repo/'unrelated.py').write_text('value = 1\n');self.commit()
        self.assertEqual(cpu_evidence.identity(self.repo,spec,env),first)
        source=self.repo/cpu_contracts.CONTRACTS['math'][1]
        source.write_text(source.read_text()+'\n# changed dependency\n');self.commit()
        self.assertNotEqual(cpu_evidence.identity(self.repo,spec,env)['key'],first['key'])
        logic=self.repo/'tests/test_logic.py'
        logic.write_text(logic.read_text()+'\n# changed dependency audit\n');self.commit()
        self.assertEqual(cpu_evidence.identity(self.repo,spec,env)['scope'],'full-tree')

    def test_auto_contract_plan_falls_back_for_unknown_changes_and_blocks_faults(self):
        self.contract_sources();base=self.sha
        source=self.repo/cpu_contracts.CONTRACTS['math'][1]
        source.write_text(source.read_text().replace('max_logits_elems = max_logits_bytes // 4','max_logits_elems = max_logits_bytes'))
        self.commit()
        self.assertEqual(cpu_contracts.changed_contracts(self.repo,base),['math'])
        path=self.root/'plan.json';path.write_text(json.dumps(dict(hypothesis='changed helper',knobs={'VLLM_TEST':'1'},context=self.pair_context())))
        plan=self.cli('plan','planner',str(path),'--base',base,'--prepare-only')
        self.assertEqual([s['name'] for s in plan['stages']],['checks-math','sensitivity','gpu'])
        for stage in plan['stages'][:-1]:
            self.assertEqual(self.wait(stage['submission']['id'])['state'],'failed')
        self.assertFalse((self.logs/'arms').exists())
        (self.repo/'unrelated.py').write_text('value=1\n');self.commit()
        self.assertIsNone(cpu_contracts.changed_contracts(self.repo,base))

    def test_new_helper_dependencies_disable_narrow_selection_and_cache(self):
        self.contract_sources();base=self.sha
        source=self.repo/cpu_contracts.CONTRACTS['math'][1]
        source.write_text(source.read_text().replace('max_logits_elems = max_logits_bytes // 4',
            'max_logits_elems = max_logits_bytes // __import__("unreviewed_dependency").width'))
        self.commit()
        self.assertIsNone(cpu_contracts.changed_contracts(self.repo,base))
        spec=ex.normalize(dict(kind='cpu',revision=self.sha,hypothesis='changed dependency',
            command=[sys.executable,'bench/cpu_checks.py','--contract','math']),self.repo)
        self.assertEqual(cpu_evidence.identity(self.repo,spec,{})['scope'],'full-tree')


class ContractAndTimingTests(unittest.TestCase):
    def test_faults_are_detected_by_existing_checks_and_survivors_fail(self):
        report=cpu_contracts.sensitivity(ROOT)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['detected'],4)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in {'tests/test_logic.py',*[v[1] for v in cpu_contracts.CONTRACTS.values()]}:
                target=root/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
            path=root/'tests/test_logic.py';path.write_text(path.read_text().replace('if not cond:\n','if False:\n'))
            weak=cpu_contracts.sensitivity(root)
            self.assertFalse(weak['passed'])
            self.assertTrue(any(m['status']=='survived' for m in weak['mutations']))

    def test_empirical_estimates_require_matching_successful_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ex.Store(directory)
            payload=dict(spec=dict(kind='cpu',revision='a'*40,command=['true'],context={'runtime':'one'},
                                   depends_on=[],hypothesis='estimate',estimate_min=30),environment={})
            for index,seconds in enumerate((61,121,181)):
                job=store.submit('worker',payload,repeat='sample')['id'];store.state(job,'running');store.state(job,'succeeded')
                with store.db:store.db.execute('INSERT INTO timings VALUES(?,?,?,?,?)',(job,'cpu_run',seconds,time.time()+index,1))
                if index==0:self.assertEqual(metrics.predict(store.db,payload)['source'],'declared')
            predicted=metrics.predict(store.db,payload)
            self.assertEqual(predicted['minutes'],4,predicted)
            self.assertEqual(predicted['source'],'observed-p90')
            changed=copy.deepcopy(payload);changed['spec']['context']['runtime']='different'
            self.assertEqual(metrics.predict(store.db,changed)['source'],'declared')
            lines=['1|long|100|30|long|boot|','2|short|101|30|short|boot|']
            rank=fixtures.fleet_priority.rank(lines,{},102,estimates={'short':{'minutes':1,'source':'observed-p90'}})
            self.assertEqual(rank[0]['session'],'short')
            self.assertEqual(fixtures.fleet_priority.rank(lines,{},2000,estimates={'short':{'minutes':1}})[0]['session'],'long')

    def test_phase_failures_do_not_become_successful_duration_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ex.Store(directory)
            job=store.submit('a',dict(spec=dict(kind='pair',revision='a'*40,hypothesis='phases',depends_on=[]),environment={}))['id']
            with self.assertRaises(ValueError):
                with metrics.timed(store,job,'boot'):
                    raise ValueError('boot failed')
            summary=metrics.summary(store)
            self.assertEqual(summary['boot']['n'],0)
            self.assertEqual(summary['boot']['failed_samples'],1)


if __name__=='__main__':
    unittest.main()
