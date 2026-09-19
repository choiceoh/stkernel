"""probes/engine_qwen38_step: the decode step solved from small nets -- four layer sets give the fixed part, a GDN layer,
a QSA layer and the PLE injection exactly, and 48 layers are 36 GDN, 12 QSA and one PLE. The GPU half runs on the
single-GPU lane (probes/engine_kernel_check.py --lanes qwen38_step)."""
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "requires numpy")
class SolveTests(unittest.TestCase):
    def facts(self):
        cfg = json.loads((ROOT / "probes/qwen38_config.json").read_text())
        cfg = cfg.get("text_config", cfg)
        # facts.load's convention: the PLE injects before layer (ple_layer_id - 1)
        return NS(config={"layer_types": cfg["layer_types"]}, ple_layers=[i - 1 for i in cfg.get("ple_layer_ids") or ()])

    def test_the_layer_sets_hold_four_independent_unknowns(self):
        from probes import engine_qwen38_step as step
        F = self.facts()
        rows = [step.counts(F, s) for s in step.LAYER_SETS]
        self.assertEqual(rows, [{"fixed": 1, "gdn": 3, "qsa": 1, "ple": 0}, {"fixed": 1, "gdn": 2, "qsa": 0, "ple": 0},
                                {"fixed": 1, "gdn": 0, "qsa": 1, "ple": 0}, {"fixed": 1, "gdn": 1, "qsa": 0, "ple": 1}])
        truth = {"fixed": 900.0, "gdn": 410.0, "qsa": 620.0, "ple": 75.0}
        measured = [(c, sum(truth[u] * c[u] for u in truth)) for c in rows]
        parts = step.solve(measured)
        for u in truth:
            self.assertAlmostEqual(parts[u], truth[u], places=6)
        self.assertAlmostEqual(step.extrapolate(parts), 900 + 36 * 410 + 12 * 620 + 75, places=4)

    def test_the_full_model_counts_are_the_configs(self):
        from probes import engine_qwen38_step as step
        F = self.facts()
        self.assertEqual(step.counts(F, range(len(F.config["layer_types"]))),
                         {"fixed": 1, "gdn": step.FULL["gdn"], "qsa": step.FULL["qsa"], "ple": step.FULL["ple"]})

    def test_kernel_names_land_in_families(self):
        from probes import engine_qwen38_step as step
        self.assertEqual(step.family("_qsa_mqa_paged_kernel"), "qsa")
        self.assertEqual(step.family("fused_recurrent_gated_delta_rule_fwd_kernel"), "gdn / kda")
        self.assertEqual(step.family("nvjet_tst_128x64_64x7_1x1_v_bz_coopA_TNT"), "gemm (cublas/cutlass)")
        self.assertEqual(step.family("_skinny_gemv_kernel"), "gemm (cublas/cutlass)")
        self.assertEqual(step.family("something_new"), "other")


class LaneTests(unittest.TestCase):
    def test_the_kernel_check_runs_it_as_a_lane(self):
        source = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("if args.lanes == 'qwen38_step':", source)
        self.assertIn("qwen38_step(args.output, args.ranks)", source)
        self.assertIn("if args.lanes == 'qwen38_step_ab':", source)
        self.assertIn("qwen38_step(args.output, args.ranks, arms=ARMS)", source)

    def test_the_mtp_gemv_lane_takes_out_the_shapes_the_table_names(self):
        """qwen38_step_mtp_gemv's second arm pops the MTP head's shapes from skinny_gemv.CONFIGS: each must be there
        (a pop that raises is a table that moved), the one-row shapes among them, and the gemv lane's MTP sweep the same."""
        from engine.kernels.common import skinny_gemv
        from probes.engine_qwen38_gemv import SHAPES
        from probes.engine_qwen38_step import GEMV_ARMS, MTP_GEMV
        self.assertEqual(GEMV_ARMS[0], "served")
        self.assertTrue(set(MTP_GEMV) <= set(skinny_gemv.CONFIGS))
        self.assertTrue(skinny_gemv.ONE_ROW <= set(MTP_GEMV))
        self.assertEqual({shape for label, (shape, _) in SHAPES.items() if label.startswith("mtp ")}, set(MTP_GEMV))
        source = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("if args.lanes == 'qwen38_step_mtp_gemv':", source)
        self.assertIn("for arms in (GEMV_ARMS, GEMV_ARMS[::-1], GEMV_ARMS)]", source)   # both orders, alternating

    def test_the_ab_lane_is_the_first_arm_less_the_second(self):
        from probes.engine_qwen38_step import ARMS, difference

        def step(wall, device, gemm, qsa):
            return {"target rows 1 blocks 6": {"wall_us": {"step_48": wall}, "device_us": {"step_48": device},
                                               "launches": {"step_48": 900.0},
                                               "families": {"gemm": {"step_48_us": gemm}, "qsa": {"step_48_us": qsa}}}}
        self.assertEqual(ARMS[0], "served")
        got = difference(step(20000.0, 18000.0, 9000.0, 500.0), step(21000.0, 18900.0, 9900.0, 500.4))
        self.assertEqual(got, {"target rows 1 blocks 6": {"wall_us": -1000.0, "device_us": -900.0, "launches": 0.0,
                                                          "families_us": {"gemm": -900.0}}})

    def test_the_kernel_shape_is_bound_before_any_lane_is_built(self):
        """As fleet.main does first: the lanes admit their cells against the bound shape."""
        source = (ROOT / "probes/engine_qwen38_step.py").read_text(encoding="utf-8")
        one = source[source.index("def measure("):source.index("def run(")]
        self.assertLess(one.index("kernel_shape.bind_recorded("), one.index("build(ranks, ranks, rank"))

    def test_each_layer_set_is_a_process_of_its_own(self):
        """A built net's weights stay referenced by the lanes' prepared views: two in one process ran out of 4 GiB."""
        source = (ROOT / "probes/engine_qwen38_step.py").read_text(encoding="utf-8")
        run = source[source.index("def run("):]
        self.assertIn('"--one", name', run)
        self.assertNotIn("build(", run.split("def ")[1] if False else run[:run.index("return report")])

    def test_the_ahead_lane_alternates_its_arms_on_one_build(self):
        """qwen38_step_ahead (fleet --draft-ahead): each arm first as often as last, one layer set built once (the arms
        differ in the host's order of work, not in what is built), the kernel shape bound first, the tokens compared."""
        from probes.engine_qwen38_step import AHEAD_ORDER
        self.assertEqual(AHEAD_ORDER.count(True), AHEAD_ORDER.count(False))
        self.assertEqual(AHEAD_ORDER[:3], tuple(not on for on in AHEAD_ORDER[3:]))
        source = (ROOT / "probes/engine_qwen38_step.py").read_text(encoding="utf-8")
        body = source[source.index("def ahead("):source.index("def assemble(")]
        self.assertEqual(body.count("build(ranks, ranks, rank"), 1)
        self.assertLess(body.index("kernel_shape.bind_recorded("), body.index("build(ranks, ranks, rank"))
        self.assertIn("draft_ahead=on, tokens=True)", body)
        check = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("if args.lanes == 'qwen38_step_ahead':", check)
        self.assertIn("qwen38_step_ahead(args.output, args.ranks)", check)

    def test_the_budget_fits_beside_production(self):
        from probes import engine_qwen38_step as step
        self.assertLessEqual(step.MAX_GIB, 4.0)
        self.assertTrue(all(len(s) <= 4 for s in step.LAYER_SETS))


if __name__ == "__main__":
    unittest.main()
