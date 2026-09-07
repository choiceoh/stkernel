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
SPEC = dict(schema=1, baseline={'VLLM_A':'0','VLLM_B':'0'}, prime={'VLLM_A':'1','VLLM_B':'1'},
            candidates=[dict(name='A',knobs={'VLLM_A':'1','VLLM_B':'0'}),
                        dict(name='B',knobs={'VLLM_A':'0','VLLM_B':'1'})])


class CampaignTests(unittest.TestCase):
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
                with self.assertRaisesRegex(ValueError, 'changed'):
                    campaign.check(root,stage,record,dict(identity,revision='d'*40))
                with self.assertRaisesRegex(ValueError, 'quality'):
                    campaign.check(root,stage,dict(record,quality=dict(ok=1,total=2)),identity)
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
