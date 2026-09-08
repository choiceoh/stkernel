"""CPU-only failure-path checks. All serving/GPU operations are fakes."""
import copy
import base64
import datetime
import gzip
import hashlib
import io
import os
import re
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
sys.path.insert(0,str(ROOT/'tests'))
import prefill_serving as m
from test_prefill_compare import fixture


class ServingTests(unittest.TestCase):
    def test_redirected_file_logs_prove_launch_and_reject_stale_empty_or_changed_file(self):
        section = m.SNAPSHOT[m.SNAPSHOT.index('# This launcher'):]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'glm53.log'
            raw = b'[megakernel] mla prefill32 LAUNCHED T=6912 W=2048\n'
            path.write_bytes(raw)
            started = datetime.datetime.fromtimestamp(path.stat().st_mtime-1, datetime.timezone.utc).isoformat()
            container = dict(Id='container-A', Mounts=[dict(Destination='/glmlogs',Source=directory)],
                             State=dict(StartedAt=started, Running=True))
            def execute(c=container, command='vllm serve model > /glmlogs/glm53.log 2>&1', check=None):
                ns=dict(c=c, cmd=command, archive=True, state={}, marker='mla prefill32 LAUNCHED',
                        pathlib=__import__('pathlib'),datetime=datetime,re=re,json=json,gzip=gzip,base64=base64,
                        sha=lambda b:hashlib.sha256(b).hexdigest(),
                        subprocess=SimpleNamespace(check_output=check or (lambda *a,**k:json.dumps([c]))))
                with redirect_stdout(io.StringIO()):exec(section,ns)
                return ns['state']
            state=execute()
            self.assertTrue(state['launch_proof'])
            self.assertEqual(gzip.decompress(base64.b64decode(state['log_gzip_base64'])),raw)
            self.assertEqual(state['log_source']['container_id'],'container-A')
            with self.assertRaisesRegex(RuntimeError,'redirection'):
                execute(command='vllm serve model >> /glmlogs/glm53.log 2>&1')
            path.write_bytes(b'')
            with self.assertRaisesRegex(RuntimeError,'empty'):execute()
            path.write_bytes(raw);os.utime(path,(1,1))
            with self.assertRaisesRegex(RuntimeError,'predates'):execute()
            path.write_bytes(raw)
            stopped=copy.deepcopy(container);stopped['State']['Running']=False
            with self.assertRaisesRegex(RuntimeError,'changed during capture'):
                execute(check=lambda *a,**k:json.dumps([stopped]))

    def test_fresh_gpu_failure_stops_before_gate_admission_and_deploy(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(source=ROOT,out=Path(directory)/'serving',revision='c'*40,
                                 candidate='mla',gate_dir=Path(directory)/'gpu',refresh_gate=True)
            with patch.object(m,'check_holder'),patch.object(m,'pinned'),patch.object(m.subprocess,'run'),\
                 patch.object(m,'run_owned',side_effect=RuntimeError('fresh GPU failed')) as run,\
                 patch.object(m,'verify_gate') as verify:
                with self.assertRaisesRegex(RuntimeError,'fresh GPU failed'):m.run_bracket(args)
                self.assertEqual(run.call_count,1)
                self.assertIn(str(ROOT/'probes/glm53_offline_checks.py'),run.call_args.args[0])
                self.assertEqual(run.call_args.kwargs['env']['OFFLINE_SOURCE_REV'],'c'*40)
                verify.assert_not_called()
                self.assertFalse(args.out.exists())

    def test_refresh_refuses_wrong_candidate_or_sanitizer_only_plan(self):
        with patch.object(m,'run_owned') as run:
            with self.assertRaisesRegex(RuntimeError,'exact single-candidate'):
                m.refresh_gpu_gate('moe',ROOT,'c'*40,ROOT/'unused')
            bad=list(m.PINS)
            bad[0]=(*bad[0][:3],[*bad[0][3],'--sanitize-only'])
            with patch.object(m,'PINS',tuple(bad)),self.assertRaisesRegex(RuntimeError,'exact single-candidate'):
                m.refresh_gpu_gate('mla',ROOT,'c'*40,ROOT/'unused')
            run.assert_not_called()

    def test_arm_primes_then_measures_and_records_both_phases(self):
        self.collect(False)

    def test_bad_priming_quality_stops_before_measured_traffic(self):
        self.collect(True)

    def collect(self, bad_quality):
        data=fixture()[0]
        nodes=copy.deepcopy(data['before'])
        after=copy.deepcopy(nodes)
        for state in after.values():state['launch_proof']=False
        contract={k:nodes['10.10.10.2'][k] for k in ('manifest_sha','mounts')}
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            env=dict(REPO=str(ROOT),PREFILL_SERVING_REV='c'*40,
                     PREFILL_SERVING_OUT=str(out),PREFILL_SERVING_CANDIDATE='moe',PREFILL_SERVING_FIRST_ARM='TESTB1')
            def run(command,**kw):
                name=command[command.index('--name')+1]
                calls.append(name)
                phase=copy.deepcopy(data['priming' if name.endswith('PRIME') else 'measured'])
                phase['record']['name']=phase['fresh']['name']=name
                if bad_quality:phase['record']['quality']['ok']=8
                with (out/'onepass.jsonl').open('a') as f:f.write(json.dumps(phase['record'])+'\n')
                m.save(out/(name+'.fresh.json'),phase['fresh'])
            with patch.dict(m.os.environ,env),patch.object(m,'check_holder'),patch.object(m,'pinned'),\
                 patch.object(m,'source_contract',return_value=contract),patch.object(m,'IMAGE','sha256:'+'a'*64),\
                 patch.object(m,'capture',side_effect=[nodes,after]) as capture,patch.object(m,'run_owned',side_effect=run):
                args=SimpleNamespace(name='TESTB1',enabled=False)
                if bad_quality:
                    with self.assertRaisesRegex(RuntimeError,'quality'):m.collect_arm(args)
                    self.assertEqual(calls,['TESTB1PRIME'])
                    self.assertFalse((out/'TESTB1.json').exists())
                    self.assertEqual(capture.call_count,1)
                else:
                    m.collect_arm(args)
                    self.assertEqual(calls,['TESTB1PRIME','TESTB1'])
                    arm=json.loads((out/'TESTB1.json').read_text())
                    self.assertIn('priming',arm);self.assertIn('measured',arm)
                    self.assertEqual(arm['before'],nodes)

    def test_attestation_rejects_wrong_worker_source_and_capacity(self):
        nodes=fixture()[0]['before']
        contract={k:nodes['10.10.10.2'][k] for k in ('manifest_sha','mounts')}
        with patch.object(m,'IMAGE','sha256:'+'a'*64):
            m.attest(nodes,contract,m.CANDIDATES['moe'][0],False)
            for field in ('source','capacity'):
                changed=copy.deepcopy(nodes)
                if field=='source':changed['10.10.10.4']['manifest_sha']='d'*64
                else:changed['10.10.10.4']['args']['num-gpu-blocks-override']='1056'
                with self.subTest(field=field),self.assertRaises(RuntimeError):
                    m.attest(changed,contract,m.CANDIDATES['moe'][0],False)

    def test_failed_gate_never_reaches_deploy_or_creates_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'not-created'
            args=SimpleNamespace(source=ROOT,out=out,revision='c'*40,candidate='moe',gate_dir=Path(directory))
            with patch.object(m,'check_holder'),patch.object(m,'pinned'),patch.object(m.subprocess,'run'),\
                 patch.object(m,'verify_gate',side_effect=RuntimeError('GPU failed')),patch.object(m,'run_owned') as run:
                with self.assertRaisesRegex(RuntimeError,'GPU failed'):m.run_bracket(args)
                run.assert_not_called()
                self.assertFalse(out.exists())

    def test_deploy_failure_reports_failure_without_public_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            args=SimpleNamespace(source=ROOT,out=out,revision='c'*40,candidate='moe',
                                 gate_dir=Path(directory),name='TEST')
            calls=[]
            def run(command,**kw):
                calls.append(command)
                if 'deploy' in command:raise RuntimeError('deploy failed')
                self.fail('unexpected recovery boot')
            with patch.dict(m.os.environ,FLEET_SESSION='test'),patch.object(m,'check_holder'),patch.object(m,'pinned'),\
                 patch.object(m.subprocess,'run'),patch.object(m,'verify_gate',return_value={}),\
                 patch.object(m,'remote',return_value={'disk_free_gib':200}),patch.object(m,'run_owned',side_effect=run):
                self.assertEqual(m.run_bracket(args),1)
            self.assertEqual(len(calls),1)
            self.assertIn('deploy',calls[0])
            result=json.loads((out/'completion.json').read_text())
            self.assertIn('deploy failed',result['error'])
            self.assertNotIn('restore_error',result)
            self.assertEqual(result['public_recovery'],'central idle controller')
            self.assertFalse(result.get('restored'))

    def test_later_boot_uses_first_baseline_budget_without_second_deduction(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            env=dict(REPO=str(ROOT),PREFILL_SERVING_OUT=str(out),PREFILL_SERVING_CANDIDATE='moe',
                     PREFILL_SERVING_FIRST_ARM='TESTB1')
            controls=m.boot_controls(fixture()[0]['before']['10.10.10.2'])
            self.assertEqual(controls['GMU'],'0.6229')
            self.assertEqual(controls['CG_UTIL_DELTA'],'0')
            with patch.dict(m.os.environ,env),patch.object(m,'check_holder'),patch.object(m,'run_owned') as run:
                args=SimpleNamespace(name='TESTA',knobs=m.CANDIDATES['moe'][0]+'=1')
                with self.assertRaisesRegex(RuntimeError,'baseline boot controls missing'):m.boot_arm(args)
                run.assert_not_called()
                m.save(out/'boot-controls.json',controls)
                m.boot_arm(args)
                self.assertEqual(run.call_args.kwargs['env']['GMU'],'0.6229')
                self.assertEqual(run.call_args.kwargs['env']['CG_UTIL_DELTA'],'0')


if __name__=='__main__':unittest.main()
