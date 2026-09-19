"""engine/profiles/dsv41/composition: DeepSeek-V4.1's layer plan from its config, and the checks that make it a plan
rather than a guess.

The config is the checkpoint's own, kept with the 09-10 record (measurements/dsv41_mhc_20260910/reference-config.json);
tests/test_engine_kernel_shape.py's fixture, read off srv4 three days later, agrees with it on every key it carries.
The derivation is the retired overlay's (dsv41_layers.py, removed with the vLLM stack in #1152), whose probe held it
to the checkpoint's 96,085 tensors in both directions. No accelerator, no checkpoint.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "measurements/dsv41_mhc_20260910/reference-config.json"


def config() -> dict:
    return json.loads(CONFIG.read_text())["text_config"]


class LayerPlanTests(unittest.TestCase):
    def setUp(self):
        from engine.profiles.dsv41.composition import layer_plans
        self.cfg = config()
        self.plans = layer_plans(self.cfg)

    def test_the_stack_is_twenty_encoder_layers_then_twenty_decoder_layers(self):
        """The boundary the config names (candidate_source_layer_id 20) is the one compress_ratios steps at."""
        from engine.profiles.dsv41.composition import DECODER, ENCODER
        self.assertEqual(len(self.plans), 40)
        self.assertEqual([p.index for p in self.plans], list(range(40)))
        self.assertEqual([p.role for p in self.plans[:20]], [ENCODER] * 20)
        self.assertEqual([p.role for p in self.plans[20:]], [DECODER] * 20)
        self.assertEqual([p.compress_ratio for p in self.plans[:2]], [0, 0])
        self.assertEqual({p.compress_ratio for p in self.plans[2:20]}, {2})
        self.assertEqual({p.compress_ratio for p in self.plans[20:]}, {1})

    def test_kv_sources_stop_at_the_boundary_and_indexers_cross_it(self):
        self.assertEqual([p.index for p in self.plans if p.kv_source], [2, 8, 14, 20])
        self.assertEqual([p.index for p in self.plans if p.indexer], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertTrue(all(p.kv_source is False for p in self.plans[21:]))

    def test_the_two_layers_that_do_not_compress_take_the_other_rotary_table(self):
        """A ratio of 0 is a sliding window AND the base rope_theta with YaRN off; every other layer takes
        compress_rope_theta with YaRN on."""
        swa = [p for p in self.plans if p.swa_only]
        self.assertEqual([p.index for p in swa], [0, 1])
        self.assertEqual(swa[0].rope(self.cfg), (0, 10000.0))
        self.assertEqual(self.plans[2].rope(self.cfg), (65536, 160000.0))
        self.assertEqual(self.plans[39].rope(self.cfg), (65536, 160000.0))

    def test_the_engram_layers_are_the_configs_and_they_index_their_own_table(self):
        engram = [p for p in self.plans if p.engram]
        self.assertEqual([p.index for p in engram], [1, 14])
        self.assertEqual([p.engram_table for p in engram], [0, 1])     # the table each reads, in config order
        self.assertTrue(all(p.engram_table is None for p in self.plans if p.index not in (1, 14)))

    def test_every_layer_routes_and_the_plan_names_the_features(self):
        """`intermediate_size` is null: the dense MLP is the shared expert, so there is no dense prefix to declare."""
        from engine.profiles.dsv41.composition import (ENGRAM, MOE, SPARSE_ATTENTION, WINDOW_ATTENTION, plan)
        self.assertIsNone(self.cfg.get("intermediate_size"))
        p = plan(self.cfg)
        self.assertEqual(len(p.layers), 40)
        self.assertEqual({layer.mlp for layer in p.layers}, {MOE})
        self.assertEqual([layer.mixer for layer in p.layers[:2]], [WINDOW_ATTENTION] * 2)
        self.assertEqual({layer.mixer for layer in p.layers[2:]}, {SPARSE_ATTENTION})
        self.assertEqual(p.layers_of(ENGRAM), [1, 14])
        self.assertEqual(p.layers_of(MOE), list(range(40)))
        self.assertEqual(p.layers[1].inject, (ENGRAM,))
        self.assertEqual(p.layers[0].inject, ())

    def test_describe_says_the_split_in_one_line(self):
        from engine.profiles.dsv41.composition import describe
        line = describe(self.plans)
        self.assertIn("encoder 20", line)
        self.assertIn("decoder 20", line)
        self.assertIn("kv sources [2, 8, 14, 20]", line)
        self.assertIn("engram [1, 14]", line)


class RefusalTests(unittest.TestCase):
    """Each check is a way the config could be read wrong; a plan that survived one of these would be a wrong model
    that still runs."""

    def setUp(self):
        self.cfg = config()

    def refuses(self, pattern, **changes):
        from engine.profiles.dsv41.composition import layer_plans
        with self.assertRaisesRegex(ValueError, pattern):
            layer_plans(dict(self.cfg, **changes))

    def test_a_ratio_list_that_is_not_the_layers_plus_the_mtp_heads(self):
        self.refuses("compress_ratios", compress_ratios=self.cfg["compress_ratios"][:-1])

    def test_a_kv_source_past_the_boundary(self):
        """Also the two-decoder-source case: a second source at or past the boundary is one past it."""
        self.refuses("reach past", kv_source_layer_ids=[2, 8, 14, 20, 24])
        self.refuses("reach past", candidate_source_layer_id=14)

    def test_a_boundary_the_ratios_do_not_step_at(self):
        # 25 is past every kv source, so this reaches the step check rather than the one above
        self.refuses("does not step", candidate_source_layer_id=25)

    def test_an_engram_layer_outside_the_stack(self):
        self.refuses("outside", engram_layer_ids=[1, 40])


if __name__ == "__main__":
    unittest.main()
