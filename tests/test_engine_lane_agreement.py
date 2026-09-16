"""The ranks agree on the MoE lanes they armed, before the capture that depends on them.

2026-09-16, seven boots: rank 1 armed FOUR `static v2` lanes where its peers armed seven, ran ahead to
one-shot sequence 1183 while the peers sat at 1179, and spun in the peer wait until the transport's own
`__trap()` fired at 30 s. It reached the logs as `unspecified launch failure` with an Xid 43 beside it,
was read as a broken GPU by three different readers, and the divergence itself was one line per lane in
four separate logs. #1036 removed that cause by arming every expert variant before the capture; this is
the check that says so out loud, in a second, whatever the next cause turns out to be.
"""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    """`armed_static_lanes` is the log line as a value."""

    def module(self):
        source = (ROOT / "engine/kernels/b12x/moe_dispatch.py").read_text(encoding="utf-8")
        self.assertIn("_STATIC_V2_ARMED.append(name)", source)
        return source

    def test_the_name_is_recorded_where_the_line_is_printed(self):
        source = self.module()
        appended = source.index("_STATIC_V2_ARMED.append(name)")
        printed = source.index('"[b12x static v2] lane serving: %s')
        self.assertLess(appended, printed)
        # and both sit after the cache is filled, so a lane counted is a lane built
        self.assertLess(source.index("_STATIC_V2_KERNEL_CACHE[cache_key] = result"), appended)

    def test_the_accessor_is_sorted_and_a_tuple(self):
        """Two ranks compare values, not arming order: the order is the process's, the set is not."""
        source = self.module()
        head = source.index("def armed_static_lanes()")
        self.assertIn("tuple(sorted(_STATIC_V2_ARMED))", source[head:head + 400])

@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class AgreementTests(unittest.TestCase):
    def engine(self, comm):
        from engine.profiles.glm53.adapter import Glm53Engine
        engine = Glm53Engine.__new__(Glm53Engine)
        engine.net = NS(comm=comm)
        return engine

    def comm(self):
        return NS(world_size=4, rank=0)

    def test_the_armed_lanes_are_agreed_as_a_payload(self):
        from engine.base.tripwire import Tripwire
        engine = self.engine(self.comm())
        seen = {}
        with patch.dict("sys.modules", {"engine.kernels.b12x.moe_dispatch":
                                        NS(armed_static_lanes=lambda: ("a", "b"))}), \
             patch.object(Tripwire, "agree_payload",
                          lambda self, site, payload: seen.update(site=site, payload=payload)):
            engine._agree_on_armed_lanes()
        self.assertEqual(seen, {"site": "boot:b12x-lanes", "payload": ["a", "b"]})

    def test_a_table_that_binds_no_b12x_kernel_is_not_an_error(self):
        """The reference lane table exists to talk, not to serve; it arms no static lane."""
        from engine.base.tripwire import Tripwire
        engine = self.engine(self.comm())
        # a None entry in sys.modules is what an absent module looks like to `from ... import`
        with patch.dict("sys.modules", {"engine.kernels.b12x.moe_dispatch": None}),              patch.object(Tripwire, "agree_payload", side_effect=AssertionError("must not be reached")):
            engine._agree_on_armed_lanes()          # returns quietly

    def test_ranks_that_armed_different_lanes_die_naming_the_site(self):
        from engine.base.tripwire import CollectiveDivergence, Tripwire
        engine = self.engine(self.comm())
        with patch.dict("sys.modules", {"engine.kernels.b12x.moe_dispatch":
                                        NS(armed_static_lanes=lambda: ("a",))}), \
             patch.object(Tripwire, "agree_payload",
                          side_effect=CollectiveDivergence("the ranks disagree at 'boot:b12x-lanes'")):
            with self.assertRaises(CollectiveDivergence) as caught:
                engine._agree_on_armed_lanes()
        self.assertIn("boot:b12x-lanes", str(caught.exception))


class PlacementTests(unittest.TestCase):
    """After the warmups that arm the lanes, before the capture that depends on them."""

    def setUp(self):
        source = (ROOT / "engine/profiles/glm53/adapter.py").read_text(encoding="utf-8")
        self.capture = source[source.index("    def capture_decode(self, max_seqs: int)"):
                              source.index("    def _agree_on_armed_lanes(self)")]
        self.warmup = source[source.index("    def _warmup_decode_experts(self"):]
        self.warmup = self.warmup[:self.warmup.index("\n    def ", 1)]

    def test_it_runs_after_the_decode_expert_rendezvous(self):
        self.assertLess(self.warmup.index('wait_prepared("decode-experts"'),
                        self.warmup.index("self._agree_on_armed_lanes()"))

    def test_the_capture_comes_after_the_warmups_that_arm(self):
        self.assertLess(self.capture.index("self._warmup_serving_kernels()"),
                        self.capture.index("graphs_began = time.time()"))


if __name__ == "__main__":
    unittest.main()
