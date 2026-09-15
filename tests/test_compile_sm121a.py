"""bench/compile_sm121a.sh: the local sm_121a compile lane's contract.

The lane runs on a box that is not a Spark (bench/OST_97X_LANE.md). Nothing here needs a
device or docker -- what it guards is the three things that silently break it: a default
image tag that drifts from the builder's output, a mount over the wrong site directory, and
a script that lets a CUDA context open and so lets someone read a number off the wrong card.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / 'bench/compile_sm121a.sh'
BUILDER = ROOT / 'engine/runtime/build-x86_64.sh'
PROBE = ROOT / 'probes/engine_moe_m64_compile.py'
DOC = ROOT / 'bench/OST_97X_LANE.md'


def default_of(text, name):
    match = re.search(rf'\$\{{{name}:-([^}}]+)\}}', text)
    return match.group(1) if match else None


class LaneContractTests(unittest.TestCase):
    def setUp(self):
        self.lane = LANE.read_text(encoding='utf-8')

    def test_the_image_default_is_the_one_the_builder_writes(self):
        built = default_of(BUILDER.read_text(encoding='utf-8'), 'ST_X86_IMAGE')
        self.assertEqual(default_of(self.lane, 'ST_X86_IMAGE'), built,
                         'the lane would silently run a stale or absent image')

    def test_it_mounts_over_site_packages_because_that_shadows_dist_packages(self):
        self.assertIn('site=/usr/local/lib/python3.12/site-packages', self.lane)
        self.assertNotIn('/dist-packages', self.lane.split('# Why ST_VENDORED')[-1]
                         .split('set -euo')[-1],
                         'mounting over dist-packages is the silent no-op this lane exists past')
        self.assertIn('SHADOWS', self.lane, 'and the header must say why')

    def test_no_device_and_no_network(self):
        for flag in ('CUDA_VISIBLE_DEVICES=', 'NVIDIA_VISIBLE_DEVICES=void', '--network=none'):
            self.assertIn(flag, self.lane, flag)
        self.assertIn('CUTE_DSL_ARCH=sm_121a', self.lane,
                      'sm_121a is the compile target, not this box')

    def test_it_refuses_without_the_vendored_flashinfer_and_says_how_to_get_it(self):
        self.assertIn('no vendored flashinfer at', self.lane)
        self.assertIn('_collapse_to_vmk', self.lane, 'the header must name what is missing')
        self.assertIn('tar czf -', self.lane, 'and the one-time export')


class ProbeContractTests(unittest.TestCase):
    def test_the_compile_probe_asserts_no_cuda_context_was_created(self):
        text = PROBE.read_text(encoding='utf-8')
        self.assertIn('torch.cuda.is_initialized()', text)
        self.assertIn("'PASS' if not report['cuda_initialized']", text,
                      'a compile that woke the driver is not a device-free compile')

    def test_it_compiles_both_arms_so_either_side_breaking_is_caught(self):
        text = PROBE.read_text(encoding='utf-8')
        self.assertIn('m128_control', text)
        self.assertIn('m64_candidate', text)
        self.assertIn('_prefill_tile64', text)


class DocumentedTests(unittest.TestCase):
    def test_the_lane_document_names_the_script_and_the_flashinfer_gap(self):
        doc = DOC.read_text(encoding='utf-8')
        self.assertIn('bench/compile_sm121a.sh', doc)
        self.assertIn('_collapse_to_vmk', doc)
        self.assertIn('site-packages', doc)


if __name__ == '__main__':
    unittest.main()
