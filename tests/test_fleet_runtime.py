"""Fair CPU admission, owned-process monitoring and unused request cleanup."""
import contextlib
import io
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
import experiment_resources as resources
import experiment_retirement as retirement
import cpu_unittest


class RuntimeTests(unittest.TestCase):
    setUp=fixtures.SubmissionTests.setUp
    commit=fixtures.SubmissionTests.commit
    cli=fixtures.SubmissionTests.cli
    submit=fixtures.SubmissionTests.submit
    wait=fixtures.SubmissionTests.wait
    wait_state=coalescing.CoalescingTests.wait_state
    pair_context=fixtures.SubmissionTests.pair_context
    manual_job=coalescing.CoalescingTests.manual_job

    def store(self, suffix=''):
        store=ex.Store(self.jobs/suffix)
        self.addCleanup(store.db.close)
        return store

    def policy(self, slots=2):
        (self.fleet/'cpu-policy.json').write_text(json.dumps(dict(slots=slots,memory_mb=256,reserve_mb=0)))

    def test_large_waiter_cannot_be_overtaken_by_repeated_small_arrivals(self):
        self.policy();store=self.store()
        busy,wide,small=[self.manual_job(store,n) for n in ('busy','wide','small')]
        one=resources.normalize(dict(cpu_slots=1,cpu_memory_mb=64))
        two=resources.normalize(dict(cpu_slots=2,cpu_memory_mb=64))
        with patch.object(resources,'memory',return_value=(1024,1024)) as probes:
            self.assertTrue(resources.acquire(store,busy,one))
            self.assertFalse(resources.acquire(store,wide,two))
            ticket=store.db.execute('SELECT ticket FROM cpu_waiters WHERE job=?',(wide,)).fetchone()[0]
            # Recovery must refresh declared budgets without losing FIFO age.
            with store.db:store.db.execute('UPDATE cpu_waiters SET slots=1,memory_mb=32 WHERE job=?',(wide,))
            for _ in range(3):
                self.assertFalse(resources.acquire(store,small,one))
                self.assertFalse(resources.acquire(store,wide,two))
            self.assertEqual(probes.call_count,1)
            self.assertEqual(tuple(store.db.execute('SELECT slots,memory_mb FROM cpu_waiters WHERE job=?',(wide,)).fetchone()),(2,64))
            self.assertEqual(store.db.execute('SELECT ticket FROM cpu_waiters WHERE job=?',(wide,)).fetchone()[0],ticket)
            with store.db:store.db.execute('DELETE FROM cpu_leases WHERE job=?',(busy,))
            self.assertFalse(resources.acquire(store,small,one))
            self.assertTrue(resources.acquire(store,wide,two))
            with store.db:store.db.execute('DELETE FROM cpu_leases WHERE job=?',(wide,))
            self.assertTrue(resources.acquire(store,small,one))
        self.assertEqual(store.db.execute('SELECT count(*) FROM cpu_waiters').fetchone()[0],0)

    def test_live_workers_admit_wide_request_before_later_small_request(self):
        self.policy();store=self.store()
        releases=[self.root/name for name in ('release-busy','release-wide')]
        # Always unblock fixture children, including after an assertion failure.
        for release in releases:self.addCleanup(release.touch)
        def command(name, release=None):
            code='from pathlib import Path; import time; '+f'Path({str(self.root/name)!r}).touch();'
            if release:
                code+='deadline=time.monotonic()+10\n'+f'while not Path({str(release)!r}).exists():\n assert time.monotonic()<deadline\n time.sleep(.01)\n'
            return [sys.executable,'-c',code]
        busy=self.submit('busy',command=command('busy-ran',releases[0]),resources=dict(cpu_slots=1,cpu_memory_mb=64))
        self.wait_state(busy['id'],'running')
        wide=self.submit('wide',command=command('wide-ran',releases[1]),resources=dict(cpu_slots=2,cpu_memory_mb=64))
        self.wait_state(wide['id'],'waiting_cpu')
        small=self.submit('small',command=command('small-ran'),resources=dict(cpu_slots=1,cpu_memory_mb=64))
        self.wait_state(small['id'],'waiting_cpu')
        self.assertFalse((self.root/'small-ran').exists())
        releases[0].touch()
        wide_row=self.wait_state(wide['id'],'running')
        self.assertEqual(store.get(small['id'])['state'],'waiting_cpu')
        releases[1].touch()
        small_row=self.wait(small['id'])
        self.assertEqual(small_row['state'],'succeeded',small_row)
        self.assertGreater(small_row['started'],wide_row['started'])
        self.assertEqual(self.wait(wide['id'])['state'],'succeeded')
        self.assertEqual(self.wait(busy['id'])['state'],'succeeded')
        self.assertFalse((self.fleet/'holder').exists())

    def test_dead_retired_or_oversized_head_does_not_strand_followers(self):
        for cause in ('dead','retired','policy'):
            with self.subTest(cause=cause):
                self.policy();store=self.store(cause)
                busy,wide,small=[self.manual_job(store,n) for n in ('busy','wide','small')]
                one=resources.normalize(dict(cpu_slots=1,cpu_memory_mb=64))
                two=resources.normalize(dict(cpu_slots=2,cpu_memory_mb=64))
                with patch.object(resources,'memory',return_value=(1024,1024)):
                    self.assertTrue(resources.acquire(store,busy,one))
                    self.assertFalse(resources.acquire(store,wide,two))
                    self.assertFalse(resources.acquire(store,small,one))
                    if cause=='dead':
                        with store.db:store.db.execute('UPDATE cpu_waiters SET pid=? WHERE job=?',(1073741824,wide))
                    elif cause=='retired':store.state(wide,'retired')
                    else:
                        self.policy(slots=1)
                        with store.db:store.db.execute('DELETE FROM cpu_leases WHERE job=?',(busy,))
                    self.assertTrue(resources.acquire(store,small,one))
                    self.assertIsNone(store.db.execute('SELECT 1 FROM cpu_waiters WHERE job=?',(wide,)).fetchone())

    def test_resource_timeout_cleans_only_its_waiter(self):
        self.policy(slots=1);store=self.store()
        busy=self.manual_job(store,'busy')
        waiting=self.manual_job(store,'waiting',timeout_s=1,resources=dict(cpu_memory_mb=64))
        clock=[0.0]
        def sleep(seconds):clock[0]+=seconds
        with patch.object(resources,'memory',return_value=(1024,1024)):
            self.assertTrue(resources.acquire(store,busy,resources.normalize(dict(cpu_memory_mb=64))))
            with patch.object(resources.time,'monotonic',side_effect=lambda:clock[0]), \
                 patch.object(resources.time,'sleep',side_effect=sleep),patch.object(resources.subprocess,'Popen') as launch:
                rc,reason=resources.run_cpu(store,waiting,['true'],store.get(waiting)['payload'])
                launch.assert_not_called()
        self.assertEqual(rc,124);self.assertIn('queue',reason)
        self.assertEqual(store.db.execute('SELECT count(*) FROM cpu_waiters').fetchone()[0],0)
        self.assertEqual([row[0] for row in store.db.execute('SELECT job FROM cpu_leases')],[busy])

    def test_retirement_after_admission_releases_capacity_without_spawning(self):
        self.policy();store=self.store();job=self.manual_job(store,'retire',resources=dict(cpu_memory_mb=64))
        original=resources.acquire
        def acquire(*args):
            answer=original(*args)
            if answer:store.state(job,'retired')
            return answer
        with patch.object(resources,'memory',return_value=(1024,1024)),patch.object(resources,'acquire',side_effect=acquire), \
             patch.object(resources.subprocess,'Popen') as launch,self.assertRaises(ex.RetiredJob):
            resources.run_cpu(store,job,['true'],store.get(job)['payload'])
        launch.assert_not_called()
        self.assertEqual(store.db.execute('SELECT count(*) FROM cpu_leases').fetchone()[0],0)


    def test_finished_child_uses_wait_instead_of_a_fixed_sleep(self):
        self.policy();store=self.store();job=self.manual_job(store,'quick',resources=dict(cpu_memory_mb=64))
        from unittest.mock import Mock
        proc=Mock(pid=os.getpid(),returncode=0)
        proc.poll.side_effect=[None,0]
        with patch.object(resources,'total_memory',return_value=1024), \
             patch.object(resources,'memory',return_value=(1024,1024)),patch.object(resources,'group_snapshot',return_value=[('S',1)]), \
             patch.object(resources.subprocess,'Popen',return_value=proc),patch.object(resources,'stop'), \
             patch.object(resources.time,'sleep',side_effect=AssertionError('fixed sleep delayed completion')):
            self.assertEqual(resources.run_cpu(store,job,['true'],store.get(job)['payload']),(0,None))
        proc.wait.assert_called_once()
        self.assertEqual(store.db.execute('SELECT count(*) FROM cpu_leases').fetchone()[0],0)

    def pair(self, store, name, value):
        return self.manual_job(store,name,kind='pair',knobs={'VLLM_TEST':str(value)})





    def test_duration_assignment_balances_cost_and_keeps_exact_coverage(self):
        ids=['a','b','c','d'];costs=dict(a=8,b=1,c=7,d=1)
        cold,mode=cpu_unittest.shard_assignment(ids,2,{})
        warm,mode=cpu_unittest.shard_assignment(ids,2,costs)
        self.assertEqual(mode,'duration-balanced')
        self.assertEqual(sorted(name for group in warm for name in group),ids)
        self.assertLess(max(sum(costs[n] for n in g) for g in warm),max(sum(costs[n] for n in g) for g in cold))
        bad={name:float('nan') for name in ids}
        self.assertEqual(cpu_unittest.shard_assignment(ids,2,bad),(cold,'round-robin'))

    def test_recorded_timings_are_reused_but_changed_test_files_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tests').mkdir()
            name='test_fleet_timing_fixture';source=root/'tests'/f'{name}.py'
            source.write_text('import unittest\nclass C(unittest.TestCase):\n'+''.join(
                f' def test_{i}(self): self.assertTrue(True)\n' for i in range(4)))
            try:
                with patch.dict(os.environ,FLEET_CPU_SLOTS='2',FLEET_EXPERIMENT_ROOT=str(root/'times')), \
                     contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    cold=cpu_unittest.execute('tests/test_fleet*.py',root,jobs=2)
                    warm=cpu_unittest.execute('tests/test_fleet*.py',root,jobs=2)
                    source.write_text(source.read_text()+'\n# new reviewed source\n')
                    changed=cpu_unittest.execute('tests/test_fleet*.py',root,jobs=2)
                self.assertEqual(cold['scheduling'],'round-robin')
                self.assertEqual(warm['scheduling'],'duration-balanced');self.assertEqual(warm['timing_hints'],4)
                self.assertEqual(changed['scheduling'],'round-robin')
                for report in (cold,warm,changed):
                    self.assertTrue(report['passed']);self.assertEqual(report['tests_run'],4)
                    self.assertEqual(len(report['test_durations_s']),4)
            finally:sys.modules.pop(name,None)

    def test_malformed_assignment_cannot_skip_or_duplicate_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tests').mkdir();marker=root/'executed'
            name='test_fleet_assignment_fixture'
            (root/'tests'/f'{name}.py').write_text('import unittest\nfrom pathlib import Path\nclass C(unittest.TestCase):\n'
                f' def test_a(self): Path({str(marker)!r}).touch()\n def test_b(self): pass\n')
            plan=root/'plan.json';plan.write_text(json.dumps([[name+'.C.test_a'],[name+'.C.test_a']]))
            try:
                with self.assertRaisesRegex(ValueError,'exactly once'):
                    cpu_unittest.execute('tests/test_fleet*.py',root,shard='0/2',shard_plan=plan)
                self.assertFalse(marker.exists())
            finally:sys.modules.pop(name,None)
