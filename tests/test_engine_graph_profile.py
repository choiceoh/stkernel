"""The probe the fleet queue already admits but nobody had written (45차 §90).

`bench/fleet_onepass.ST_PROBES` names `probes/engine_graph_profile.py` -- the queue would accept it, refuse
any other path, and there was no file. These pin the two things that can be checked without four Sparks: the
name is admitted and present, and a kernel is filed under the lane it came from.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def admitted():
    """`bench/fleet_onepass.ST_PROBES`, read rather than imported: that module imports its siblings as
    top-level names, which only works with bench/ on the path, and this asks about a literal."""
    import ast
    tree = ast.parse((ROOT / "bench" / "fleet_onepass.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "ST_PROBES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError("fleet_onepass no longer declares ST_PROBES")


class AdmittedProbeTests(unittest.TestCase):
    def test_every_admitted_st_probe_exists(self):
        """A name in the list with no file behind it is a check the queue would take and then fail to run."""
        self.assertEqual([p for p in admitted() if not (ROOT / p).is_file()], [])

    def test_the_graph_profile_is_one_of_them(self):
        self.assertIn("probes/engine_graph_profile.py", admitted())

    def test_forward_profile_explicitly_selects_native_execution(self):
        # build's default is the stock composition. A captured graph alone
        # never proves that the probe used production's W4/FP8 linears.
        import ast
        tree = ast.parse((ROOT / "probes/engine_graph_profile.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'build']
        self.assertEqual(len(calls), 1)
        args = {k.arg: k.value for k in calls[0].keywords}
        self.assertEqual(ast.literal_eval(args['execution']), 'native')

    def test_none_of_them_requires_an_argument(self):
        """The queue invokes an admitted check with NO arguments, and `ST_FLAGS` admits only a handful --
        `--drafter-dir` and `--tier-dir` are not among them, so a check that requires either can be started
        and never run. Three of the six did (45차 §95). Every default has to work on a node."""
        import ast
        needed = {}
        for rel in admitted():
            tree = ast.parse((ROOT / rel).read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                        and any(k.arg == "required" and getattr(k.value, "value", False) is True
                                for k in node.keywords)):
                    flag = next((a.value for a in node.args if isinstance(a, ast.Constant)), "?")
                    needed.setdefault(rel, []).append(flag)
        self.assertEqual(needed, {})


class LaneTests(unittest.TestCase):
    def test_a_kernel_is_filed_under_the_lane_it_came_from(self):
        from probes.engine_graph_profile import lane_of
        for name, lane in (("mk_mhc_kernel", "mHC"), ("mk_mla_pair_kernel", "MLA / DSA"),
                           ("fused_recurrent_kda_fwd", "KDA"), ("causal_conv1d_ring", "KDA"),
                           ("b12x_moe_dynamic_gated", "MoE"), ("nvjet_tst_128x_64", "dense GEMM"),
                           ("ncclDevKernel_AllReduce_Sum", "collective")):
            with self.subTest(kernel=name):
                self.assertEqual(lane_of(name), lane)

    def test_an_unclaimed_kernel_keeps_its_own_name(self):
        """A surprise must not hide inside a bucket: that is the whole point of profiling."""
        from probes.engine_graph_profile import lane_of
        self.assertEqual(lane_of("some_kernel_nobody_declared"), "other")


if __name__ == "__main__":
    unittest.main()
