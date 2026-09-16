"""The HPC-Ops top-k comparison probe: vendored bytes match their record, the lane is wired, CPU import is safe."""
import hashlib
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'probes' / 'vendor' / 'hpcops_topk'


class VendoredSources(unittest.TestCase):
    def test_every_vendored_file_matches_its_recorded_sha256(self):
        record = (VENDOR / 'SOURCE.txt').read_text()
        pairs = re.findall(r'^([0-9a-f]{64})  (\S+)$', record, flags=re.M)
        self.assertEqual(len(pairs), 7)
        for digest, name in pairs:
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((VENDOR / name).read_bytes()).hexdigest(), digest)

    def test_the_mit_notice_is_kept(self):
        self.assertIn('licensed under MIT', (VENDOR / 'LICENSE.txt').read_text())

    def test_nothing_served_imports_the_vendored_operator(self):
        for path in (ROOT / 'engine').rglob('*.py'):
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotIn('hpcops', path.read_text(errors='replace'))


class Lane(unittest.TestCase):
    def test_kernel_check_routes_the_lane_to_the_probe(self):
        text = (ROOT / 'probes' / 'engine_kernel_check.py').read_text()
        self.assertIn("args.lanes == 'topk_hpcops'", text)
        self.assertIn('from probes.engine_topk_hpcops import run', text)

    def test_the_probe_imports_without_a_gpu_and_names_its_arms(self):
        import importlib
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch is not installed')
        probe = importlib.import_module('probes.engine_topk_hpcops')
        self.assertEqual(probe.K, 512)
        self.assertTrue(all(n % 4 == 0 for _, n in probe.DECODE + probe.PREFILL))
        self.assertEqual({rows for rows, _ in probe.PREFILL}, {1024})


if __name__ == '__main__':
    unittest.main()
