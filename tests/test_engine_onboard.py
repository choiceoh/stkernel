"""The front door (engine/base/onboard, tools/onboard.py): a checkpoint no profile was written for is READ, and what
its config does not settle comes back as a blank rather than a guess.

CHARTER D5 names no model, so attaching one has to be a path and not an edit: a profile package declares itself and
the wizard discovers it (`kernel_shape.profiles`/`claims`), and a config nobody claims still gets a shape when it
settles one -- with the lane table and the work list that follow. These cases pin both halves, and the third pins the
honest outcome: the three models this repo serves today all come back with blanks naming exactly what their profiles
supply. CPU only, no checkpoint.
"""
import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSV41_CONFIG = ROOT / "measurements/dsv41_mhc_20260910/reference-config.json"

# a routed transformer the ordinary way: no hyper-connection, no sparse indexer, no linear attention
PLAIN_MOE = {"model_type": "some_moe_text", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
             "num_key_value_heads": 8, "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8,
             "moe_intermediate_size": 768, "intermediate_size": 12288, "hidden_act": "silu"}


def pinned(name: str) -> dict:
    """A *_TEXT_CONFIG fixture out of tests/test_engine_kernel_shape.py without importing torch."""
    tree = ast.parse((ROOT / "tests/test_engine_kernel_shape.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "fixture", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} is not in tests/test_engine_kernel_shape.py")


def cli():
    spec = importlib.util.spec_from_file_location("onboard_cli", ROOT / "tools/onboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadingTests(unittest.TestCase):
    def read(self, cfg, **kw):
        from engine.base.onboard import read_config
        return read_config(cfg, **kw)

    def test_a_config_that_settles_everything_gets_a_shape(self):
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertEqual((r.blanks, r.complete), ((), True))
        s = r.shape
        self.assertEqual((s.hidden, s.hc, s.tp, s.hc_variant), (4096, 1, 4, None))
        self.assertEqual((s.attention.kind, s.attention.heads, s.attention.head_dim, s.attention.kv_heads), ("gqa", 8, 128, 2))
        self.assertEqual((s.linear, s.indexer, s.spec_k), (None, None, 1))
        self.assertEqual((s.moe.experts, s.moe.experts_local, s.moe.inter, s.moe.inter_local), (128, 32, 768, 768))
        self.assertEqual((s.moe.topk, s.moe.quant, s.moe.activation), (8, "bf16", "silu"))
        self.assertEqual(s.moe.dense_inter_local, 12288 // 4)

    def test_every_value_carries_the_key_it_was_read_from(self):
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertEqual(r.sources["hidden"], "hidden_size")
        self.assertEqual(r.sources["attention.kind"], "num_key_value_heads")
        self.assertEqual(r.sources["moe.experts"], "num_experts")
        self.assertEqual(r.sources["tp"], "hardware (CHARTER D5)")            # not the model's
        self.assertIn("no MTP key", r.sources["spec_k"])
        self.assertIn("no indexer key", r.sources["indexer"])

    def test_the_sink_is_carried_as_not_established_not_as_false(self):
        """False would be a claim about the model's softmax. None is the shape's own 'not established', and the lane
        refuses it by name -- which is the outcome we want from a config that never mentions it."""
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertIsNone(r.shape.attention.sink)
        self.assertEqual([b.field for b in r.unsettled], ["attention.sink"])
        from engine.kernels import cells
        mla = [v for v in cells.admission(r.shape) if v.lane == "mla"][0]
        self.assertEqual(mla.status, "refused")

    def test_placement_is_the_operators_and_a_routed_model_without_it_is_a_blank(self):
        ep, tp = self.read(PLAIN_MOE, placement="ep").shape.moe, self.read(PLAIN_MOE, placement="tp").shape.moe
        self.assertEqual((ep.experts_local, ep.inter_local), (32, 768))        # whole experts a rank
        self.assertEqual((tp.experts_local, tp.inter_local), (128, 192))       # every expert's intermediate sliced
        blanks = self.read(PLAIN_MOE).blanks
        self.assertEqual([b.field for b in blanks], ["moe.experts_local"])
        self.assertIn("operator", blanks[0].why)
        with self.assertRaises(ValueError):
            self.read(PLAIN_MOE, placement="everywhere")

    def test_a_config_without_the_width_says_so_and_stops(self):
        r = self.read({"model_type": "x"})
        self.assertEqual(([b.field for b in r.blanks], r.shape), (["hidden"], None))

    def test_the_streams_and_their_mixer_are_two_different_facts(self):
        r = self.read(dict(PLAIN_MOE, hc_mult=4), placement="ep")
        self.assertEqual((r.shape.hc, r.shape.hc_variant), (4, None))          # four streams, mixer not established
        self.assertIn("hc_variant", [b.field for b in r.unsettled])
        from engine.kernels import cells
        lanes = {v.lane: v.status for v in cells.admission(r.shape)}
        self.assertEqual(lanes.get("mhc_decode"), "refused")

    def test_one_stream_has_no_mhc_lane_at_all(self):
        """hc 1 is a plain residual: no streams to mix, so the mHC lanes do not apply -- the rule the indexer and KDA
        lanes already follow. The measured cell keeps all fourteen of its lanes."""
        from engine.base.kernel_shape import MEASURED
        from engine.kernels import cells
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertEqual([v.lane for v in cells.admission(r.shape) if v.lane.startswith("mhc")], [])
        self.assertEqual(len(cells.admission(MEASURED)), 14)

    def test_judging_an_incomplete_reading_gives_no_table(self):
        """A lane verdict on a guessed shape is worth less than no verdict."""
        from engine.base.onboard import judge
        out = judge(self.read(PLAIN_MOE))                                      # no placement
        self.assertEqual((out["admission"], out["plan"]), ([], []))


class ServedModelsTests(unittest.TestCase):
    """The three checkpoints this repo serves today all have profiles -- so the generic door must NOT pretend to
    settle them. Each blank names exactly what its profile supplies."""

    def blanks(self, cfg):
        from engine.base.onboard import read_config
        return {b.field: b.why for b in read_config(cfg, placement="ep").blanks}

    def test_glm53_and_qwen38_need_their_indexer_and_decay(self):
        for name in ("GLM53_TEXT_CONFIG", "QWEN38_TEXT_CONFIG"):
            with self.subTest(config=name):
                blanks = self.blanks(pinned(name))
                self.assertEqual(sorted(blanks), ["indexer.compress", "linear.decay"])
                self.assertIn("kpool, ced, qsa", blanks["indexer.compress"])

    def test_dsv41_needs_its_attention_kind_indexer_and_expert_format(self):
        cfg = json.loads(DSV41_CONFIG.read_text())
        text = dict(cfg["text_config"], quantization_config=cfg["quantization_config"])
        blanks = self.blanks(text)
        self.assertEqual(sorted(blanks), ["attention.kind", "indexer.compress", "moe.quant"])
        self.assertIn("kv_lora_rank", blanks["attention.kind"])                # one KV head and no latent key
        self.assertIn("fp8", blanks["moe.quant"])                              # what the config actually says

    def test_a_profile_claims_its_model_type_and_the_wizard_discovers_it(self):
        from engine.base import kernel_shape as ks
        self.assertEqual(set(ks.profiles()), {"glm53", "qwen38", "dsv41"})
        self.assertTrue(all(m.startswith("engine.profiles.") for m in ks.profiles().values()))
        self.assertEqual(ks.claims("glm5_next_text"), "glm53")
        self.assertEqual(ks.claims("qwen4_exp_text"), "qwen38")
        self.assertEqual(ks.claims("deepseek_v41_text"), "dsv41")
        self.assertIsNone(ks.claims("some_moe_text"))
        self.assertIsNone(ks.claims(None))

    def test_the_claimed_model_type_is_the_one_the_profile_asserts(self):
        """A profile's claim and the assert in its own facts must be the same string, or a checkpoint would be routed
        to a derivation that then refuses it."""
        import importlib
        for name, asserted in (("glm53", "glm5_next_text"), ("qwen38", "qwen4_exp_text")):
            source = (ROOT / f"engine/profiles/{name}/facts.py").read_text()
            self.assertIn(f'== "{asserted}"', source, name)
            self.assertEqual(importlib.import_module(f"engine.profiles.{name}").MODEL_TYPES, (asserted,))


class CliTests(unittest.TestCase):
    def test_the_command_prints_a_shape_a_table_and_the_work(self):
        module = cli()
        result = module.run(PLAIN_MOE, placement="ep", ckpt=None)
        text = module.render(result)
        self.assertEqual(result["door"], "generic")
        self.assertIn("lanes:", text)
        self.assertIn("work, cheapest first:", text)
        self.assertIn("moe", text)
        payload = json.loads(module.as_json(result))
        self.assertEqual(payload["door"], "generic")
        self.assertEqual(payload["plan"], [v.lane for v in result["plan"]])
        self.assertTrue(payload["shape"])

    def test_a_config_that_does_not_settle_prints_its_blanks_and_exits_two(self):
        module = cli()
        result = module.run(PLAIN_MOE, placement=None, ckpt=None)              # no placement: one blank
        text = module.render(result)
        self.assertIn("no shape -- the config does not settle", text)
        self.assertIn("moe.experts_local", text)
        self.assertIn("engine/profiles/", text)                                # what fills a blank

    def test_it_reads_a_nested_text_config_and_the_outer_quantization(self):
        module = cli()
        cfg = module.text_config(DSV41_CONFIG)
        self.assertEqual(cfg["model_type"], "deepseek_v41_text")               # the nested one, not the wrapper
        self.assertEqual(cfg["quantization_config"]["quant_method"], "fp8")    # lifted from the outer config

    def test_the_door_names_the_profile_when_one_claims_the_checkpoint(self):
        module = cli()
        result = module.run({"model_type": "glm5_next_text", "hidden_size": 4096}, placement="ep", ckpt=None)
        self.assertIn("profile glm53", result["door"])                         # without --ckpt it still says who owns it


if __name__ == "__main__":
    unittest.main()
