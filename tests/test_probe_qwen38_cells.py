"""The Qwen3.8 cells probe (engine/QWEN38_CARRY.md C1): the lane is wired, its config is the checkpoint's, the facts it
builds are the ones the served profile asserts, and every glue case it runs exists."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class Lane(unittest.TestCase):
    def test_kernel_check_routes_the_lane_to_the_probe(self):
        text = (ROOT / 'probes' / 'engine_kernel_check.py').read_text()
        self.assertIn("args.lanes == 'qwen38_cells'", text)
        self.assertIn('from probes.engine_qwen38_cells import run', text)

    def test_the_config_is_the_bytes_it_records(self):
        source = (ROOT / 'probes' / 'engine_qwen38_cells.py').read_text()
        digest = hashlib.sha256((ROOT / 'probes' / 'qwen38_config.json').read_bytes()).hexdigest()
        self.assertIn(f"CONFIG_SHA256 = '{digest}'", source)

    def test_the_qualify_report_takes_a_lane_of_pairs_and_a_lane_of_numbers(self):
        # lanes.qualify: the gated residual, GDN and QSA give (max, rms) a key, the skinny GEMV one number a shape. The
        # report line took every value for a pair and died on main before any GPU case ran, once that lane was added.
        import json
        from probes.engine_qwen38_cells import held
        qualified = {'qsa_norm_rope': {'norm_rope_4x128': (0.0069, 0.0023)}, 'skinny_gemv': {'513x2560': 0.002814}}
        self.assertEqual(held(qualified), {'qsa_norm_rope': {'norm_rope_4x128': [0.0069, 0.0023]},
                                           'skinny_gemv': {'513x2560': 0.002814}})
        json.dumps(held(qualified))
        source = (ROOT / 'probes' / 'engine_qwen38_cells.py').read_text()
        self.assertIn("**held(lanes.qualify(torch.device('cuda'), F))", source)
        self.assertNotIn('list(v) for k, v in worst.items()', source)


@unittest.skipUnless(torch is not None, "requires torch")
class Facts(unittest.TestCase):
    def test_the_config_builds_the_served_facts(self):
        from probes.engine_qwen38_cells import facts
        F = facts()
        self.assertEqual((F.layers, len(F.qsa_layers), len(F.gdn_layers)), (48, 12, 36))
        self.assertEqual((F.hidden, F.hc, F.hc_rank), (2560, 4, 320))
        self.assertEqual((F.k_heads_local, F.v_heads_local, F.heads_local, F.kv_heads_local), (4, 12, 6, 1))
        self.assertEqual((F.experts_local, F.topk_experts, F.moe_inter), (128, 10, 640))

    def test_the_config_derives_the_shape_the_wizard_judges(self):
        from engine.profiles.qwen38 import shapes
        from probes.engine_qwen38_cells import facts
        from tests.test_engine_kernel_shape import QWEN38_TEXT_CONFIG
        self.assertEqual(facts().kernel_shape(), shapes.kernel_shape(QWEN38_TEXT_CONFIG))

    def test_every_glue_case_it_runs_exists_and_needs_a_gpu(self):
        import importlib
        import unittest as ut
        from probes.engine_qwen38_cells import GLUE_CASES
        for name in GLUE_CASES:
            module, cls = name.rsplit('.', 1)
            with self.subTest(case=name):
                case = getattr(importlib.import_module(module), cls)
                self.assertTrue(issubclass(case, ut.TestCase))
                self.assertTrue(ut.defaultTestLoader.getTestCaseNames(case))

    def test_the_cases_left_out_exist_in_the_classes_it_runs(self):
        import importlib
        import unittest as ut
        from probes.engine_qwen38_cells import GLUE_CASES, GLUE_LEFT_OUT, _cases
        for name in GLUE_LEFT_OUT:
            cls, method = name.rsplit('.', 1)
            with self.subTest(case=name):
                self.assertIn(cls, GLUE_CASES)
                module, klass = cls.rsplit('.', 1)
                self.assertIn(method, ut.defaultTestLoader.getTestCaseNames(getattr(importlib.import_module(module), klass)))
        suite = ut.defaultTestLoader.loadTestsFromNames([n.rsplit('.', 1)[0] for n in GLUE_LEFT_OUT])
        ids = {case.id() for case in _cases(suite)}
        self.assertTrue(set(GLUE_LEFT_OUT) <= ids and len(ids) > len(GLUE_LEFT_OUT))   # the rest of their classes run


if __name__ == '__main__':
    unittest.main()
