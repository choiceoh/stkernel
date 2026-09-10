"""Exercise preparation evidence with captured log files and mocked Docker only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
sys.path.insert(0, str(Path(__file__).parent))
import glm53_prep_proof as prep
import glm53_launch_metadata as launch
import onepass
import proof
import baseline
import judge
from test_onepass_speculation_proof import BOOT, command

KNOB = 'VLLM_GLM53_PREP_FUSED'


class PreparationProofTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'serving.log'

    def fixture(self, mode='1'):
        self.path.write_text('starting identified boot\n')
        actual = dict(launch.launch_speculation(command(5)), boot_id=BOOT,
            image='sha256:'+'b'*64, environment_spec_k='5', preparation_mode=mode,
            preparation_kernel='cuda', shadow_every='1', selfcheck_every='64')
        context = dict(expected_mode=mode, boot_id=BOOT, launch_before=actual,
            launch_after=copy.deepcopy(actual), exclusive=True, log_prefix=prep.log_prefix(self.path))
        return context

    def append(self, text):
        with self.path.open('a') as stream:
            stream.write(text)

    def log(self, mode='on', *, fused=64, checks=2):
        return (f'[prep-fused] plan built: mode={mode} kernel=cuda groups=3 gdn=(0,) attn_g=1 '
                f'factor=1 ratio=4 sbs=64 q=6 shadow_every=1 selfcheck_every=64\n'
                f'[prep-fused] {mode}: fused_steps={fused} stock_steps=8 checks ok={checks} '
                'drift=0 first_plan_check=False\n')

    def test_same_boot_on_and_full_shadow_prove_execution_after_workload_start(self):
        for value, mode, checks in (('1','on',2), ('shadow','shadow',64)):
            context = self.fixture(value)
            self.append(self.log(mode, checks=checks))
            result = proof.check([KNOB], str(self.path), preparation=context)
            self.assertEqual(result['proof_ok'], '1/1')
            self.assertEqual(result['preparation']['last_checkpoint']['checks_ok'], checks)
            self.assertIn('not fixed-request', result['preparation']['scope'])
            self.assertNotIn('/private', json.dumps(result))

    def test_plan_echo_and_previous_checkpoints_cannot_prove_current_execution(self):
        context = self.fixture()
        self.append(self.log().splitlines()[0]+'\n')
        self.assertEqual(prep.evidence(context, self.path)['verdict'], 'REJECTED')
        self.append(self.log())
        context['log_prefix'] = prep.log_prefix(self.path)
        self.append('request finished\n')
        self.assertEqual(prep.evidence(context, self.path)['verdict'], 'REJECTED')

    def test_drift_fallback_wrong_plan_unverified_and_reset_counters_are_rejected(self):
        for transform in (
            lambda text:text+'[prep-fused] DISARM -> stock\n',
            lambda text:text+'[prep-fused] on DRIFT at fused step 70\n',
            lambda text:text.replace('kernel=cuda','kernel=triton'),
            lambda text:text.replace('q=6','q=4'),
            lambda text:text.replace('q=6','q=6 q=6'),
            lambda text:text.replace('checks ok=2','checks ok=0'),
            lambda text:text.replace('checks ok=2','checks ok=65'),
            lambda text:text.replace('drift=0','drift=1'),
            lambda text:text+self.log(fused=1, checks=1).splitlines()[1]+'\n',
        ):
            context = self.fixture(); self.append(transform(self.log()))
            self.assertEqual(prep.evidence(context,self.path)['verdict'],'REJECTED')
        context = self.fixture('shadow'); self.append(self.log('shadow',checks=63))
        self.assertEqual(prep.evidence(context,self.path)['verdict'],'REJECTED')

    def test_stale_boot_and_actual_configuration_changes_are_rejected(self):
        for field,value in (('preparation_mode','0'), ('preparation_kernel','triton'),
                            ('selfcheck_every','0'), ('shadow_every','2'), ('node_rank',False),
                            ('image',None), ('num_speculative_tokens',3)):
            context = self.fixture(); self.append(self.log())
            context['launch_before'][field] = value; context['launch_after'][field] = value
            self.assertEqual(prep.evidence(context,self.path)['verdict'],'REJECTED')
        for change in (lambda c:c.update(exclusive=False),
                       lambda c:c.update(boot_id='c'*64+'|new'),
                       lambda c:c['launch_after'].update(command_sha256='d'*64)):
            context=self.fixture();self.append(self.log());change(context)
            self.assertEqual(prep.evidence(context,self.path)['verdict'],'REJECTED')

    def test_log_replacement_and_in_place_prefix_changes_are_rejected(self):
        context = self.fixture()
        self.path.rename(self.path.with_suffix('.old'))
        self.path.write_text('starting identified boot\n'+self.log())
        self.assertEqual(prep.evidence(context,self.path)['verdict'],'REJECTED')
        context = self.fixture(); self.append(self.log())
        with self.path.open('r+b') as stream: stream.write(b'X')
        self.assertEqual(prep.evidence(context,self.path)['verdict'],'REJECTED')
        self.assertIsNone(prep.log_prefix(self.path.with_suffix('.missing')))

    def test_actual_container_environment_is_read_without_changing_speculation_schema(self):
        container = dict(Id='a'*64, Image='sha256:'+'b'*64,
            State=dict(StartedAt=BOOT.split('|')[1], Running=True, Paused=False, Restarting=False),
            Config=dict(Cmd=command(5), Env=['VLLM_GLM53_SPEC_K=5',KNOB+'=shadow']))
        with patch('subprocess.check_output',return_value=json.dumps([container])):
            actual=onepass._served_speculation(BOOT,preparation=True)
            self.assertEqual(actual['preparation_mode'],'shadow')
            self.assertEqual(actual['selfcheck_every'],'64')
            self.assertNotIn('preparation_mode',onepass._served_speculation(BOOT))
        container['Config']['Env'].append(KNOB+'=1')
        with patch('subprocess.check_output',return_value=json.dumps([container])):
            self.assertIsNone(onepass._served_speculation(BOOT,preparation=True))

    def test_standalone_marker_without_actual_context_never_passes(self):
        self.fixture();self.append(self.log())
        result=proof.check([KNOB],str(self.path))
        self.assertEqual(result['proof_ok'],'0/1')
        self.assertEqual(result['preparation']['verdict'],'REJECTED')

    def record(self, name='B', *, required=True):
        row = dict(name=name, overlay='same', git='a'*40, harness=40, doc_lang='en',
            thinking=True, workload={}, runtime=None, knobs={}, proof={},
            boot_id=name, quality=dict(ok=18,total=18), korean=dict(dirty=0,n=8),
            decode=dict(windows_med=20))
        if required:
            onepass._require_preparation(row)
        return row

    def test_default_preparation_failure_and_missing_proof_make_either_arm_invalid(self):
        for side in ('A','B'):
            for missing in (None, False, 1, 'true'):
                base, candidate = self.record('B'), self.record('A')
                candidate['knobs'] = {'VLLM_GLM53_EP_TILED':'1'}
                candidate['proof']['VLLM_GLM53_EP_TILED'] = True
                base['proof'][KNOB] = candidate['proof'][KNOB] = True
                (candidate if side == 'A' else base)['proof'][KNOB] = missing
                self.assertEqual(judge.judge(candidate,base,[base,candidate])['status'], 'invalid')
                self.assertTrue(judge.unproved(candidate if side == 'A' else base))
        base = self.record()
        self.assertEqual(base['knobs'], {})
        self.assertEqual(base['preparation']['verdict'], 'REJECTED')
        self.assertTrue(judge.record_errors(base))
        base.pop('proof')
        self.assertTrue(judge.unproved(base))

    def test_baseline_reuse_and_noise_floor_require_matching_complete_declared_proofs(self):
        base, old, bad = self.record('B'), self.record('OLD',required=False), self.record('BAD')
        base['proof'][KNOB] = True
        self.assertTrue(baseline.comparison_baseline(base,base))
        self.assertFalse(baseline.comparison_baseline(bad,base))
        self.assertFalse(baseline.comparison_baseline(old,base))
        self.assertFalse(judge.compatible(old,base))
        rows = [base,old,bad]
        self.assertEqual(judge.floor_of(rows,base)[1],1)
        # Same explicit declaration is order-independent and cannot be supplied
        # by a legacy same-build row merely carrying an extra proof boolean.
        old['proof'][KNOB] = True
        self.assertFalse(judge.compatible(old,base))

    def test_malformed_required_proof_declaration_fails_closed(self):
        for required in (None, 'VLLM_GLM53_PREP_FUSED', [KNOB,KNOB], [None], ['OTHER'], {}):
            row = self.record(); row['required_proofs'] = required; row['proof'][KNOB] = True
            self.assertTrue(judge.unproved(row))
            self.assertFalse(judge.compatible(row,self.record()))
            self.assertFalse(baseline.comparison_baseline(row,self.record()))


if __name__ == '__main__':
    unittest.main()
