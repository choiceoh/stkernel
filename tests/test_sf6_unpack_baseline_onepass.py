"""Baseline-only continuation keeps the measured candidate workload intact."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from fleet_onepass import validate
from fleet_prepare import command_environment


class BaselineOnepass(unittest.TestCase):
    def test_scalar_baseline_matches_prior_workload_and_video_capacity(self):
        prior = json.loads((ROOT / 'probes/sf6_unpack_onepass_v1.json').read_text())
        argv = json.loads((ROOT / 'probes/sf6_unpack_baseline_onepass_v2.json').read_text())
        command, env = command_environment(argv, {})
        _, old_env = command_environment(prior, {})
        self.assertEqual(command, ['bash', 'bench/chain.sh', 'sf6-unpack-base-0909v2B='])
        self.assertEqual(env['MM_LIMIT'], '{"image":4,"video":1}')
        self.assertEqual(env['VLLM_GLM53_SF6_UNPACK_U8X4'], '0')
        path_keys = {'ONEPASS_JSONL', 'ONEPASS_MEMORY_DIR', 'ONEPASS_VERDICTS'}
        self.assertEqual({k: v for k, v in env.items() if k not in path_keys | {'MM_LIMIT'}},
                         {k: v for k, v in old_env.items() if k not in path_keys})
        for key in path_keys:
            self.assertIn('SF6-UNPACK-BASE-sf6-unpack-base-0909v2', env[key])
        validate(argv, ROOT, ROOT)


if __name__ == '__main__':
    unittest.main()
