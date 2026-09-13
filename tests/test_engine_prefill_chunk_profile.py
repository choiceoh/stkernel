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
