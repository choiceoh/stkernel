"""Finite CPU fixtures for actual checkpoint emission and unchanged proof rules."""
import ast
from dataclasses import dataclass,field
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'overlay/modules/glm53_runtime/glm53_prep_fused.py'
PROOF=ROOT/'bench/glm53_prep_proof.py'
OLD='if first_plan_check or st.checks_ok % 64 == 0 or st.mode == "shadow" and st.checks_ok % 16 == 0:'
NEW='if first_plan_check or st.mode == "on" or st.checks_ok % 64 == 0 or st.mode == "shadow" and st.checks_ok % 16 == 0:'
spec=importlib.util.spec_from_file_location('checkpoint_proof_fixture',PROOF)
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


def runtime(mode,path,*,old=False,cadence=64):
    source=RUNTIME.read_text()
    assert source.count(NEW)==1
    if old:source=source.replace(NEW,OLD)
    tree=ast.parse(source)
    nodes=[n for n in tree.body if getattr(n,'name',None) in (
        '_State','_report_decode_opt','_patched_prepare_inputs')]
    def warn(fmt,*args):
        with path.open('a') as stream:stream.write((fmt % args)+'\n')
    ns=dict(__name__=__name__,dataclass=dataclass,field=field,
            logger=SimpleNamespace(warning=warn,exception=warn))
    module=ast.Module(body=[ast.parse('from __future__ import annotations').body[0],*nodes],type_ignores=[])
    exec(compile(module,str(RUNTIME),'exec'),ns)
    st=ns['_State'](mode=mode,shadow_every=1,selfcheck_every=cadence)
    st.plan=SimpleNamespace(decode_opt=False)
    fused,stock=object(),object();checks=[];bad=[]
    def verify(*args):checks.append(st.steps_fused);return stock,list(bad)
    ns.update(_state_of=lambda _:st,_eligible=lambda *_:not st.plan_failed,
              _fused_prepare_inputs=lambda *_:fused,_verify=verify,
              _ORIG={'prepare_inputs':lambda *_:stock})
    return ns['_patched_prepare_inputs'],st,fused,stock,checks,bad


def context(path):
    path.write_text('[prep-fused] plan built: mode=on kernel=cuda groups=7 gdn=[2,3,4,5] '
                    'attn_g=0 factor=4 ratio=4 sbs=64 q=6 shadow_every=1 selfcheck_every=64\n')
    boot='a'*64+'|2026-09-10T00:00:00Z'
    launch=dict(boot_id=boot,image='sha256:'+'b'*64,command_sha256='c'*64,
        config_sha256='d'*64,method='dflash',node_rank=0,num_speculative_tokens=5,
        environment_spec_k='5',preparation_mode='1',preparation_kernel='cuda',
        shadow_every='1',selfcheck_every='64')
    return dict(expected_mode='1',exclusive=True,boot_id=boot,
                launch_before=launch,launch_after=dict(launch))


def emitted(path):
    return [(int(m[2]),int(m[4])) for m in prep.STATS.finditer(path.read_text())]


class PreparationCheckpointLoggingTests(unittest.TestCase):
    def test_on_emits_each_success_without_adding_verification_or_changing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            for old in (True,False):
                path=Path(tmp)/str(old);path.write_text('')
                call,st,fused,stock,checks,bad=runtime('on',path,old=old)
                for _ in range(192):self.assertIs(call(None,None,None,None),fused)
                self.assertEqual(checks,[1,64,128,192])
                self.assertEqual(emitted(path),[(1,1)] if old else [(1,1),(64,2),(128,3),(192,4)])
                self.assertEqual((st.steps_fused,st.checks_ok,st.checks_drift),(192,4,0))

    def test_shadow_every16_and_periodic_disabled_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            traces=[]
            for old in (True,False):
                path=Path(tmp)/str(old);path.write_text('')
                call,st,fused,stock,checks,bad=runtime('shadow',path,old=old)
                for _ in range(128):self.assertIs(call(None,None,None,None),fused)
                self.assertEqual(checks,list(range(1,129)))
                traces.append(emitted(path))
            self.assertEqual(traces[0],traces[1])
            self.assertEqual(traces[1],[(1,1)]+[(n,n) for n in range(16,129,16)])
            path=Path(tmp)/'disabled';path.write_text('')
            call,st,fused,stock,checks,bad=runtime('on',path,cadence=0)
            for _ in range(128):call(None,None,None,None)
            self.assertEqual(checks,[1]);self.assertEqual(emitted(path),[(1,1)])

    def test_startup_checkpoint_cannot_prove_workload_and_fresh_actual_check_can(self):
        with tempfile.TemporaryDirectory() as tmp:
            for old in (True,False):
                path=Path(tmp)/str(old);ctx=context(path)
                call,st,fused,stock,checks,bad=runtime('on',path,old=old)
                call(None,None,None,None)  # Startup's first successful check.
                ctx['log_prefix']=prep.log_prefix(path)
                with path.open('a') as stream:stream.write('workload started\n')
                for _ in range(63):call(None,None,None,None)
                self.assertEqual(checks,[1,64])
                result=prep.evidence(ctx,path)
                if old:
                    self.assertEqual(result['verdict'],'REJECTED')
                    self.assertEqual(result['reason_code'],'no_fresh_checkpoint')
                    self.assertEqual(result['launch'],ctx['launch_before'])
                    self.assertEqual(result['log_prefix'],ctx['log_prefix'])
                    self.assertGreater(result['log_after_bytes'],ctx['log_prefix']['bytes'])
                else:
                    self.assertEqual(result['verdict'],'PASS')
                    self.assertEqual(result['checkpoint_count'],1)
                    self.assertEqual(result['last_checkpoint']['fused_steps'],64)
                    self.assertEqual(result['last_checkpoint']['checks_ok'],2)

    def test_drift_is_not_logged_as_success_and_rejection_fields_are_allowlisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'drift';ctx=context(path)
            call,st,fused,stock,checks,bad=runtime('on',path)
            call(None,None,None,None);ctx['log_prefix']=prep.log_prefix(path)
            bad.append('input_ids')
            for _ in range(62):call(None,None,None,None)
            self.assertIs(call(None,None,None,None),stock)
            self.assertIs(call(None,None,None,None),stock)
            self.assertEqual(emitted(path),[(1,1)])
            self.assertEqual(checks,[1,64]);self.assertTrue(st.plan_failed)
            ctx['launch_before']['raw_env']='DO_NOT_PUBLISH'
            ctx['launch_after']['raw_env']='DO_NOT_PUBLISH'
            ctx['log_prefix']['raw_argv']='DO_NOT_PUBLISH'
            result=prep.evidence(ctx,path)
            self.assertEqual(result['verdict'],'REJECTED')
            self.assertEqual(result['reason_code'],'preparation_failure')
            self.assertNotIn('DO_NOT_PUBLISH',str(result))
            self.assertNotIn('raw_env',result['launch'])
            self.assertNotIn('raw_argv',result['log_prefix'])
            self.assertNotIn('error_text',result)


if __name__=='__main__':unittest.main(verbosity=2)
