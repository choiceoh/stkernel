"""The prefill-chunk / decode-rows profile the queue admits (45차, C=4/C=1).

What can be checked without a GPU: the probe is admitted by name and present, needs no argument the queue
would not pass, selects native execution like the graph profile it extends, picks the rank whose file the
node holds (srv4 holds rank3of4 only), and its fixed-cost fit recovers a planted line.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_engine_graph_profile import admitted  # noqa: E402

PROBE = "probes/engine_prefill_chunk_profile.py"


class AdmissionTests(unittest.TestCase):
    def test_the_probe_is_admitted_and_present(self):
        self.assertIn(PROBE, admitted())
        self.assertTrue((ROOT / PROBE).is_file())

    def test_it_takes_only_flags_the_queue_admits(self):
        """`fleet.sh run --gpu` validates argv against ST_FLAGS: a flag outside it is a check the queue starts
        and never runs. Every flag this probe declares must be one the policy admits."""
        import ast
        policy = ast.parse((ROOT / "bench" / "fleet_onepass.py").read_text())
        flags = next(ast.literal_eval(n.value) for n in policy.body
                     if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ST_FLAGS" for t in n.targets))
        tree = ast.parse((ROOT / PROBE).read_text())
        declared = {next(a.value for a in n.args if isinstance(a, ast.Constant))
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"}
        self.assertTrue(declared)
        self.assertEqual(declared - set(flags), set())
        self.assertIn("--seqs", flags)          # the graph profile's row count is admitted too (C=4 needs --seqs 4)

    def test_it_selects_native_execution(self):
        import ast
        tree = ast.parse((ROOT / PROBE).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build"]
        self.assertEqual(len(calls), 1)
        args = {k.arg: k.value for k in calls[0].keywords}
        self.assertEqual(ast.literal_eval(args["execution"]), "native")


class HelperTests(unittest.TestCase):
    def test_the_isolated_rank_is_the_one_whose_file_the_node_holds(self):
        from probes.engine_prefill_chunk_profile import rank_on_this_node
        from probes.engine_graph_profile import rank_on_this_node as graph_rank
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "rank3of4.safetensors").write_bytes(b"")
            self.assertEqual(rank_on_this_node(d), 3)
            self.assertEqual(graph_rank(d), 3)
            (Path(d) / "rank0of4.safetensors").write_bytes(b"")
            self.assertEqual(rank_on_this_node(d), 0)
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            rank_on_this_node(d)

    def test_the_fixed_cost_fit_recovers_a_planted_line(self):
        from probes.engine_prefill_chunk_profile import fit_fixed_cost
        tokens, F, v = 27648, 380.0, 0.29                      # ms per chunk, ms per token
        points = [(tokens, tokens // c, tokens * v + (tokens // c) * F) for c in (2304, 4608, 6912, 9216)]
        fixed, per_token = fit_fixed_cost(points)
        self.assertAlmostEqual(fixed, F, places=6)
        self.assertAlmostEqual(per_token, v, places=9)


if __name__ == "__main__":
    unittest.main()


class ExposureTests(unittest.TestCase):
    """The timeline arm's arithmetic: a kernel's exposed time is the part no other stream's kernel covers."""

    def test_exposed_time_subtracts_only_other_streams(self):
        import importlib
        probe = importlib.import_module("probes.engine_prefill_chunk_profile")
        events = [
            ("moe_static_kernel", 7, 0.0, 100.0),        # stream 7: 0..100
            ("mk_gemm2_kernel", 9, 10.0, 40.0),          # stream 9: inside the MoE kernel -> hidden entirely
            ("mk_gemm2_kernel", 9, 90.0, 130.0),         # stream 9: 90..130 -> 10 hidden, 30 exposed
            ("mk_mhc_ar_kernel", 7, 130.0, 150.0),       # stream 7 alone -> exposed
            ("mk_gemm2_kernel", 7, 20.0, 30.0),          # same stream as MoE: never subtracted by its own stream
        ]
        ex = probe.trace_exposure(events, steps=1)
        lanes = ex["lanes"]
        self.assertAlmostEqual(lanes["MoE"]["raw_ms"], 0.100)
        self.assertAlmostEqual(lanes["MoE"]["exposed_ms"], 0.100 - 0.030 - 0.010)   # covered by stream 9's 10..40 and 90..100
        self.assertAlmostEqual(lanes["dense GEMM"]["raw_ms"], 0.080)
        self.assertAlmostEqual(lanes["dense GEMM"]["exposed_ms"], 0.030)             # 10..40 hidden, 20..30 (stream 7) hidden by 9's 10..40, 100..130 exposed
        self.assertAlmostEqual(lanes["mHC"]["exposed_ms"], 0.020)
        self.assertAlmostEqual(ex["wall_ms"], 0.150)
        self.assertAlmostEqual(ex["busy_ms"], 0.150)
        self.assertAlmostEqual(ex["idle_ms"], 0.0)
        self.assertEqual(ex["streams"], [7, 9])

    def test_gaps_and_steps_divide(self):
        import importlib
        probe = importlib.import_module("probes.engine_prefill_chunk_profile")
        events = [("a_kernel", 1, 0.0, 10.0), ("a_kernel", 1, 20.0, 30.0)]
        ex = probe.trace_exposure(events, steps=2)
        self.assertAlmostEqual(ex["wall_ms"], 0.015)
        self.assertAlmostEqual(ex["busy_ms"], 0.010)
        self.assertAlmostEqual(ex["idle_ms"], 0.005)
        self.assertEqual(ex["lanes"]["other"]["launches"], 1.0)
        self.assertEqual(probe.trace_exposure([], 1)["lanes"], {})
