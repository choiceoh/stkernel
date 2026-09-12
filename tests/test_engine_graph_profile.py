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
