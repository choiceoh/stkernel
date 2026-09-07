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

    def test_gpu_gate_requires_both_transports_all_cases_source_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frozen=root/'frozen';repo=root/'serving';gate=root/'gate';gate.mkdir()
            relative=('build/glm53/manifest.tsv','build/glm53/module.py','profiles/glm53.env',
                      'launchers/start-glm53-nvfp4-tp4.sh','probes/glm53_moe_overlap_check.py',
                      'probes/run_glm53_moe_overlap_tp4_check.sh')
            for tree in (frozen,repo):
                for path in relative:
                    p=tree/path;p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_text('module.py\t/target/module.py\tfixture\n' if path.endswith('manifest.tsv') else 'same bytes\n')
            complete=dict(ended=2,exit_code=0,restored_original=True,probes={'moe-overlap':dict(
                revision='a'*40,exit_code=0,ended=1,command=['bash','probes/run_glm53_moe_overlap_tp4_check.sh'])})
            provenance={'module.py':hashlib.sha256((repo/'build/glm53/module.py').read_bytes()).hexdigest()}
            reports=[dict(verdict='MOE_OVERLAP_GPU_PASS',transport=transport,provenance=provenance,
                results=[dict(rows=rows,skew=skew,overlap_admitted=rows>=6144,
                              eager={'bad_rows':0},changed={'bad_rows':0})
                         for rows in (4096,4143,6912,8192) for skew in (False,True)])
                     for transport in ('bf16','fp8-v3')]
            def write(c=complete,r=reports,marker=True):
                m.save(gate/'completion.json',c)
                (gate/'moe-overlap.log').write_text('{compiler diagnostic}\n'+'\n'.join(json.dumps(v) for v in r)+
                    ('\nMOE_OVERLAP_ALL_GATES_PASS\n' if marker else '\n'))
            candidate=('VLLM_GLM53_PREFILL_MOE_OVERLAP','marker','a'*40,str(frozen))
            with patch.dict(m.CANDIDATES,{'moe-overlap':candidate}),patch.object(m,'pinned'):
                write();result=m.verify_gate('moe-overlap',gate,repo)
                self.assertEqual(result['revision'],'a'*40)
                for case in ('transport','coverage','numerics','source','routing','marker','recovery','outer_failure'):
                    c,r=copy.deepcopy(complete),copy.deepcopy(reports)
                    if case=='transport':r=r[:1]
                    elif case=='coverage':r[1]['results'].pop()
                    elif case=='numerics':r[1]['results'][4]['changed']['bad_rows']=1
                    elif case=='source':r[0]['provenance']['module.py']='x'*64
                    elif case=='routing':r[0]['results'][4]['overlap_admitted']=False
                    elif case=='recovery':c['restored_original']=False
                    elif case=='outer_failure':c['exit_code']=1
                    write(c,r,case!='marker')
                    with self.subTest(case=case),self.assertRaises(RuntimeError):m.verify_gate('moe-overlap',gate,repo)
                write();(repo/'build/glm53/module.py').write_text('untested edit')
                with self.assertRaisesRegex(RuntimeError,'source changed'):m.verify_gate('moe-overlap',gate,repo)

    def test_fresh_gpu_failure_prevents_gate_admission_or_serving_deploy(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(source=ROOT,out=Path(directory)/'serving',revision='c'*40,
                candidate='moe-overlap',gate_dir=Path(directory)/'gpu',refresh_gate=True)
            with patch.object(m,'check_holder'),patch.object(m,'pinned'),patch.object(m.subprocess,'run'),\
                 patch.object(m,'run_owned',side_effect=RuntimeError('GPU failed')) as run,\
                 patch.object(m,'verify_gate') as verify:
                with self.assertRaisesRegex(RuntimeError,'GPU failed'):m.run_bracket(args)
                self.assertEqual(run.call_count,1)
                command=run.call_args.args[0]
                self.assertEqual(command[command.index('--probe-revision')+1],m.CANDIDATES['moe-overlap'][2])
                self.assertEqual(run.call_args.kwargs['env']['OFFLINE_SOURCE_REV'],'c'*40)
                verify.assert_not_called();self.assertFalse(args.out.exists())

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
                     PREFILL_SERVING_OUT=str(out),PREFILL_SERVING_CANDIDATE='moe-overlap',PREFILL_SERVING_FIRST_ARM='TESTB1')
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
            m.attest(nodes,contract,m.CANDIDATES['moe-overlap'][0],False)
            for field in ('source','capacity'):
                changed=copy.deepcopy(nodes)
                if field=='source':changed['10.10.10.4']['manifest_sha']='d'*64
                else:changed['10.10.10.4']['args']['num-gpu-blocks-override']='1056'
                with self.subTest(field=field),self.assertRaises(RuntimeError):
                    m.attest(changed,contract,m.CANDIDATES['moe-overlap'][0],False)

    def test_failed_gate_never_reaches_deploy_or_creates_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'not-created'
            args=SimpleNamespace(source=ROOT,out=out,revision='c'*40,candidate='moe-overlap',gate_dir=Path(directory))
            with patch.object(m,'check_holder'),patch.object(m,'pinned'),patch.object(m.subprocess,'run'),\
                 patch.object(m,'verify_gate',side_effect=RuntimeError('GPU failed')),patch.object(m,'run_owned') as run:
                with self.assertRaisesRegex(RuntimeError,'GPU failed'):m.run_bracket(args)
                run.assert_not_called()
                self.assertFalse(out.exists())

    def test_deploy_failure_still_attempts_public_restore_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            args=SimpleNamespace(source=ROOT,out=out,revision='c'*40,candidate='moe-overlap',
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

    def test_later_boot_uses_first_baseline_budget_without_second_deduction(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            env=dict(REPO=str(ROOT),PREFILL_SERVING_OUT=str(out),PREFILL_SERVING_CANDIDATE='moe-overlap',
                     PREFILL_SERVING_FIRST_ARM='TESTB1')
            controls=m.boot_controls(fixture()[0]['before']['10.10.10.2'])
            self.assertEqual(controls['GMU'],'0.6229')
            self.assertEqual(controls['CG_UTIL_DELTA'],'0')
            with patch.dict(m.os.environ,env),patch.object(m,'check_holder'),patch.object(m,'run_owned') as run:
                args=SimpleNamespace(name='TESTA',knobs=m.CANDIDATES['moe-overlap'][0]+'=1')
                with self.assertRaisesRegex(RuntimeError,'baseline boot controls missing'):m.boot_arm(args)
                run.assert_not_called()
                m.save(out/'boot-controls.json',controls)
                m.boot_arm(args)
                self.assertEqual(run.call_args.kwargs['env']['GMU'],'0.6229')
                self.assertEqual(run.call_args.kwargs['env']['CG_UTIL_DELTA'],'0')


if __name__=='__main__':unittest.main()
