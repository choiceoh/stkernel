#!/usr/bin/env python3
"""A gauge does not take the fleet down.

2026-09-12, production: a request reached its generation limit before the drafter's block was
verified, so `note_ceilings` got four target rows against five draft rows. `draft_ceilings` raised,
and the exception travelled decode -> runner.step -> serve.loop -> exit. All four ranks died at the
rendezvous, and the only thing between them was a statistic sampled every 64th verification that
nothing reads back into an answer.

The shape bug is fixed (#721). This pins the class: the sample is dropped, counted, and after a few
of them the gauge disarms itself -- and every one of those states is visible in /metrics.
"""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Ceilings:
    """The two fields `note_ceilings` needs, without a model, a device or a checkpoint."""

    CEILING_FAILURES_KEPT = 3

    def __init__(self):
        from engine.profiles.glm53.adapter import Glm53Engine
        self.note_ceilings = types.MethodType(Glm53Engine.note_ceilings, self)
        self.CEILING_FAILURES_KEPT = Glm53Engine.CEILING_FAILURES_KEPT
        self._ceiling_every = 64
        self.steps_verified = 63                    # the next call is the sampled one
        self.reachable_mass = self.covered_mass = 0.0
        self.ceiling_positions = self.ceiling_failures = 0
        self.ceiling_last_error = ""
        self.ceilings_off = False


class Probs:
    """Enough of a tensor for the arithmetic the gauge does after the call returns."""

    def __init__(self, rows=5):
        self.shape = (rows, 7)

    def dim(self):
        return 2


class CeilingGaugeTests(unittest.TestCase):
    def arm(self, raises):
        import engine.base.sampler as sampler
        original = sampler.draft_ceilings

        def stub(target, draft):
            if raises:
                raise RuntimeError("The size of tensor a (4) must match the size of tensor b (5)")
            return 0.25, 0.5

        sampler.draft_ceilings = stub
        self.addCleanup(setattr, sampler, "draft_ceilings", original)

    def test_a_sample_that_raises_is_dropped_counted_and_never_reaches_the_caller(self):
        self.arm(raises=True)
        engine = Ceilings()
        engine.note_ceilings(Probs(4), Probs(5))          # would have ended the step, and the fleet
        self.assertEqual(engine.ceiling_failures, 1)
        self.assertIn("must match", engine.ceiling_last_error)
        self.assertEqual((engine.reachable_mass, engine.ceiling_positions), (0.0, 0))
        self.assertFalse(engine.ceilings_off, "one failure is not a verdict")

    def test_it_disarms_itself_rather_than_paying_for_a_measurement_it_cannot_take(self):
        self.arm(raises=True)
        engine = Ceilings()
        for _ in range(engine.CEILING_FAILURES_KEPT):
            engine.steps_verified = 63
            engine.note_ceilings(Probs(4), Probs(5))
        self.assertTrue(engine.ceilings_off)
        self.assertEqual(engine.ceiling_failures, engine.CEILING_FAILURES_KEPT)
        engine.steps_verified = 63                        # and stays off: no further sampling cost
        engine.note_ceilings(Probs(4), Probs(5))
        self.assertEqual(engine.ceiling_failures, engine.CEILING_FAILURES_KEPT)

    def test_a_working_gauge_still_measures(self):
        self.arm(raises=False)
        engine = Ceilings()
        engine.note_ceilings(Probs(5), Probs(5))
        self.assertEqual((engine.reachable_mass, engine.covered_mass), (0.25, 0.5))
        self.assertEqual(engine.ceiling_positions, 5)
        self.assertEqual(engine.ceiling_failures, 0)

    def test_the_scrape_says_when_it_stopped_measuring(self):
        import test_engine_serve as T
        s = T.server()
        s.engine.ceiling_failures, s.engine.ceilings_off = 0, False
        text = s.metrics()
        self.assertIn("st:spec_ceiling_samples_failed_total", text)
        self.assertNotIn("st:spec_ceiling_sampling_off", text)
        s.engine.ceiling_failures, s.engine.ceilings_off = 3, True
        text = s.metrics()
        self.assertIn('st:spec_ceiling_samples_failed_total{engine="st"} 3', text)
        self.assertIn('st:spec_ceiling_sampling_off{engine="st"} 1', text)


if __name__ == "__main__":
    unittest.main()
