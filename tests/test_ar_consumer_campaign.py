"""CPU-only dispatch fixtures for the AR consumer's canonical onepass path."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('probes', 'bench', 'launchers', 'profiles', 'bin', 'logs/fleet', 'build/glm53'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.calls = self.root / 'calls.jsonl'
        self.calls.touch()
        self.env = dict(os.environ, PATH=str(self.root / 'bin') + os.pathsep + os.environ['PATH'],
                        LOGD=str(self.root / 'logs'), FLEET_DIR=str(self.root / 'logs/fleet'),
                        FLEET_SESSION='fixture', FLEET_RESTORE_MANAGED='1',
                        REPO=str(self.root), AR_CONSUMER_OUT=str(self.root / 'evidence'),
                        CALLS=str(self.calls))
        for name in ('ONEPASS_JSONL', 'ONEPASS_VERDICTS', 'FLEET_RUNNER_REPO',
                     'ONEPASS_FIXED_DECODE_TOKENS', 'ONEPASS_FIXED_DECODE_REPS', 'LEGS'):
            self.env.pop(name, None)
        (self.root / 'logs/fleet/holder').write_text('fixture|123|host|1|10|fixture|boot\n')
        (self.root / 'profiles/glm53.env').write_text('VLLM_GLM53_AR_CONSUMER_PDL=1\n')
        for name in ('run_ar_consumer_campaign.sh', 'ar_consumer_lever.sh'):
            shutil.copyfile(ROOT / 'probes' / name, self.root / 'probes' / name)
        self.write('bin/git', '''#!/usr/bin/env python3
import sys
if sys.argv[1:3] == ['rev-parse', 'HEAD']: print('fixture-source')
''')
        # The real stop_serving coordinator still runs, but every container and
        # network command has an inert fixture executable before the host PATH.
        self.write('bin/docker', '#!/usr/bin/env python3\n')
        self.write('bin/ssh', '#!/usr/bin/env python3\n')
        self.write('bin/curl', '''#!/usr/bin/env python3
import sys
if '-w' in sys.argv: print('000')
''')
        self.write('bench/fleet_entry.py', 'raise SystemExit(0)\n')
        self.recording_shell('launchers/deploy-overlays.sh', 'deploy')
        self.recording_shell('bench/pair.sh', 'pair')
        self.recording_shell('bench/ab-lever.sh', 'onepass')

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        path.chmod(0o755)

    def recording_shell(self, name, event):
        self.write(name, '''#!/usr/bin/env bash
python3 - "$@" <<'CODE'
import json, os, sys
with open(os.environ['CALLS'], 'a') as output:
    output.write(json.dumps(dict(event=EVENT, args=sys.argv[1:],
        lever=os.environ.get('LEVER'), ledger=os.environ.get('ONEPASS_JSONL'),
        fixed_tokens=os.environ.get('ONEPASS_FIXED_DECODE_TOKENS'))) + '\\n')
CODE
exit "${FIXTURE_ARM_RC:-0}"
'''.replace('EVENT', repr(event)))

    def run_script(self, script, *args):
        result = subprocess.run(['bash', str(self.root / 'probes' / script), *args],
                                env=self.env, text=True, capture_output=True, timeout=10)
        calls = [json.loads(row) for row in self.calls.read_text().splitlines()]
        return result, calls

    def test_promoted_profile_uses_one_opposite_candidate_and_shared_baseline_ledger(self):
        result, calls = self.run_script('run_ar_consumer_campaign.sh', '--gpu-evidence', '/missing/old-evidence')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call['event'] for call in calls], ['deploy', 'pair'])
        arm = calls[-1]
        self.assertEqual(arm['args'], ['fixtureAR0', 'VLLM_GLM53_AR_CONSUMER_PDL=0'])
        self.assertEqual(arm['lever'], str(self.root / 'bench/ab-lever.sh'))
        self.assertEqual(arm['ledger'], str(self.root / 'logs/bracket-onepass.jsonl'))
        self.assertIsNone(arm['fixed_tokens'])
        self.assertIn('--gpu-evidence is obsolete', result.stdout)
        self.assertFalse((self.root / 'evidence/gpu').exists())

    def test_default_off_profile_measures_enabled_candidate(self):
        (self.root / 'profiles/glm53.env').write_text('VLLM_GLM53_AR_CONSUMER_PDL=0\n')
        result, calls = self.run_script('run_ar_consumer_campaign.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls[-1]['args'], ['fixtureAR1', 'VLLM_GLM53_AR_CONSUMER_PDL=1'])

    def test_baseline_only_dispatches_exactly_one_standard_default_arm(self):
        result, calls = self.run_script('run_ar_consumer_campaign.sh', '--baseline-only')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call['event'] for call in calls], ['deploy', 'onepass'])
        self.assertEqual(calls[-1]['args'], ['fixtureBASE', ''])

    def test_unknown_profile_fails_before_any_serving_mutation(self):
        (self.root / 'profiles/glm53.env').write_text('VLLM_GLM53_AR_CONSUMER_PDL=unknown\n')
        result, calls = self.run_script('run_ar_consumer_campaign.sh')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])
        self.assertFalse((self.root / 'evidence').exists())

    def prepare_lever(self):
        (self.root / 'evidence').mkdir()
        (self.root / 'logs/glm53.log').write_text('onepass runtime fixture\n')
        names = ['glm53_megakernel.cu', 'glm53_megakernel.py',
                 'dsv4_oneshot_ar.cu', 'dsv4_oneshot_shim.py']
        for name in names:
            (self.root / 'build/glm53' / name).write_text('fixture\n')
        (self.root / 'build/glm53/manifest.tsv').write_text(''.join(
            name + '\t/runtime/' + name + '\n' for name in names))
        self.write('probes/ar_consumer_runtime_proof.py', '''import json, os
with open(os.environ['CALLS'], 'a') as output:
    output.write(json.dumps(dict(event='passive-proof')) + '\\n')
''')

    def test_compatibility_lever_runs_one_canonical_arm_before_passive_proof(self):
        self.prepare_lever()
        result, calls = self.run_script('ar_consumer_lever.sh', 'arm', '')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call['event'] for call in calls], ['onepass', 'passive-proof'])
        self.assertEqual(calls[0]['args'], ['arm', ''])
        self.assertTrue((self.root / 'evidence/boot-arm.log').is_file())

    def test_failed_onepass_does_not_run_followup_proof_or_gpu_work(self):
        self.prepare_lever()
        self.env['FIXTURE_ARM_RC'] = '7'
        result, calls = self.run_script('ar_consumer_lever.sh', 'arm', 'VLLM_GLM53_AR_CONSUMER_PDL=0')
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertEqual([call['event'] for call in calls], ['onepass'])


if __name__ == '__main__':
    unittest.main()
