"""Shared startup controls retain same-build, quality and drift requirements."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bench'))
import startup_campaign as campaign

PROFILE = 'VLLM_GLM53_RANK_CACHE=1\nVLLM_GLM53_FP8_CACHE=1\nVLLM_A=0\nVLLM_B=0\n'
SPEC = dict(schema=1, baseline_policy='confirm', baseline={'VLLM_A':'0','VLLM_B':'0'}, prime={'VLLM_A':'1','VLLM_B':'1'},
            candidates=[dict(name='A',knobs={'VLLM_A':'1','VLLM_B':'0'}),
                        dict(name='B',knobs={'VLLM_A':'0','VLLM_B':'1'})])


class CampaignTests(unittest.TestCase):
    def test_default_plan_has_one_shared_control_and_no_drift_verdict(self):
        spec = copy.deepcopy(SPEC); del spec['baseline_policy']
        arms = campaign.plan(spec, PROFILE)
        self.assertEqual([r['stage'] for r in arms], ['PRIME','BASE1','AR1','BR1','BR2','AR2'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'campaign.json').write_text(json.dumps(dict(spec=spec,arms=arms)))
            (root/'campaign-receipts.jsonl').write_text(''.join(json.dumps(dict(stage=r['stage'],name=r['stage']))+'\n' for r in arms))
            (root/'health-wall-seconds.tsv').write_text(''.join(r['stage']+'\t100\n' for r in arms))
            result = campaign.summarize(root)
            self.assertEqual(result['status'], 'exploration-unconfirmed')
            self.assertIsNone(result['baseline_drift_fraction'])
            self.assertFalse(result['promotion_ready'])
            self.assertEqual((result['boots'],result['independent_boots']), (6,8))

    def test_two_candidates_share_one_prime_and_two_controls(self):
        arms = campaign.plan(SPEC, PROFILE)
        self.assertEqual([r['stage'] for r in arms], ['PRIME','BASE1','AR1','BR1','BR2','AR2','BASE2'])
        self.assertEqual(sum(r['role']=='baseline' for r in arms), 2)
        self.assertEqual(sum(r['role']=='prime' for r in arms), 1)
        self.assertEqual(campaign.plan(SPEC, PROFILE.replace('CACHE=1', 'CACHE=/cache/artifacts')), arms)

    def test_missing_knob_duplicate_and_cache_off_are_rejected(self):
        for change in ('missing','duplicate','cache-off','unknown','drift'):
            spec = copy.deepcopy(SPEC); profile = PROFILE
            if change=='missing': del spec['candidates'][0]['knobs']['VLLM_B']
            if change=='duplicate': spec['candidates'].append(spec['candidates'][0])
            if change=='cache-off': profile=profile.replace('CACHE=1','CACHE=0')
            if change=='unknown': spec['baseline']['VLLM_X']='1'
            if change=='drift': spec['max_drift_fraction']=float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                campaign.plan(spec, profile)

    def test_changed_build_reused_boot_failed_quality_and_drift_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); arms = campaign.plan(SPEC, PROFILE)
            identity = dict(revision='a'*40, overlay='b'*64, ctx='2000', profile='c'*64)
            (root/'campaign.json').write_text(json.dumps(dict(spec=SPEC,arms=arms,identity=identity)))
            for arm in arms:
                stage = arm['stage']
                record = dict(name=stage, git=identity['revision'][:7], overlay=identity['overlay'][:12], boot_id=stage,
                              quality=dict(ok=2,total=2), korean=dict(dirty=0,n=2))
                (root/(stage+'-cache-env.json')).write_text(json.dumps([k+'='+v for k,v in arm['knobs'].items()]))
                modules = ''.join('e'*64+'  /usr/module'+str(n)+'\n' for n in range(3))
                for node in (1,2,3,4):
                    (root/f'{stage}-srv{node}.state').write_text('running 0 false sha256:'+'d'*64+'\n'+modules)
                (root/f'{stage}-srv2.sha256').write_text(modules)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    campaign.check(root,stage,record,dict(identity,revision='d'*40))
                with self.assertRaisesRegex(ValueError, 'quality'):
                    campaign.check(root,stage,dict(record,quality=dict(ok=1,total=2)),identity)
                if stage != 'PRIME':
                    state = root/f'{stage}-srv3.state'
                    good = state.read_text(); state.write_text(good.replace('d'*64, 'f'*64))
                    with self.assertRaisesRegex(ValueError, 'image or node module changed'):
                        campaign.check(root,stage,record,identity)
                    state.write_text(good)
                campaign.check(root,stage,record,identity)
                with self.assertRaisesRegex(ValueError, 'distinct boot'):
                    campaign.check(root,stage,record,identity)
            (root/'health-wall-seconds.tsv').write_text(''.join(r['stage']+'\t'+('140' if r['stage']=='BASE2' else '100')+'\n' for r in arms))
            result = campaign.summarize(root)
            self.assertEqual(result['status'], 'incomplete-drift')
            self.assertFalse(result['promotion_ready'])
            self.assertEqual((result['boots'], result['independent_boots']), (7,10))


if __name__ == '__main__':
    unittest.main()
