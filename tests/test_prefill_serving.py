"""CPU-only failure-path checks. All serving/GPU operations are fakes."""
import copy
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
                     PREFILL_SERVING_OUT=str(out),PREFILL_SERVING_CANDIDATE='moe')
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

    def test_deploy_failure_still_attempts_public_restore_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            args=SimpleNamespace(source=ROOT,out=out,revision='c'*40,candidate='moe',
                                 gate_dir=Path(directory),name='TEST')
            calls=[]
            def run(command,**kw):
                calls.append(command)
                if 'deploy' in command:raise RuntimeError('deploy failed')
                raise RuntimeError('restore failed too')
            with patch.dict(m.os.environ,FLEET_SESSION='test'),patch.object(m,'check_holder'),patch.object(m,'pinned'),\
                 patch.object(m.subprocess,'run'),patch.object(m,'verify_gate',return_value={}),\
                 patch.object(m,'remote',return_value={'disk_free_gib':200}),patch.object(m,'run_owned',side_effect=run):
                self.assertEqual(m.run_bracket(args),1)
            self.assertEqual(len(calls),2)
            self.assertIn('deploy',calls[0])
            self.assertIn('TESTRESTORE',calls[1])
            result=json.loads((out/'completion.json').read_text())
            self.assertIn('deploy failed',result['error'])
            self.assertIn('restore failed too',result['restore_error'])
            self.assertFalse(result.get('restored'))


if __name__=='__main__':unittest.main()
