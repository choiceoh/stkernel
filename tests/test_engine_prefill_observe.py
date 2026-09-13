"""Prefix snapshots must retain the same drafter context after incremental observation."""
from types import SimpleNamespace
import unittest

import torch

from engine.profiles.glm53.adapter import Glm53Engine
from engine.profiles.glm53.drafter import Drafter, DrafterFacts


class Caches:
    def __init__(self, ring):
        self.ring = ring.clone()
        self.snaps = {}

    def draft_ring(self, slot):
        return self.ring

    def mark_draft(self, snap, slot):
        self.snaps[snap] = self.ring.clone()


def original(drafter, ring, start, aux, cuts):
    """The independent, full-prefix computation used before this change."""
    out = Caches(ring)
    for end, snap in cuts:
        state = ring.clone()
        drafter.observe(state, torch.arange(start, start + end), aux[:end])
        out.snaps[snap] = state
    drafter.observe(out.ring, torch.arange(start, start + len(aux)), aux)
    return out


def incremental(drafter, ring, start, aux, cuts):
    engine = Glm53Engine.__new__(Glm53Engine)
    engine.caches = Caches(ring)
    engine.drafter = drafter
    engine._observe_prefill(1, start, aux, cuts)
    return engine.caches


class PrefillObserveTests(unittest.TestCase):
    def test_every_snapshot_and_final_ring_match_independent_prefixes(self):
        torch.manual_seed(7)
        facts = DrafterFacts(layers=2, hidden=16, heads=2, kv_heads=1, head_dim=4,
                            inter=16, rms_eps=1e-6, rope_theta=10000., window=64,
                            block=4, mask_id=20, conv_taps=2, conv_group=4,
                            sel_rank=4, sel_top_k=3, target_layers=(1,), k=3)
        d = Drafter(facts, SimpleNamespace(), 21)
        rand = lambda *shape: (torch.randn(*shape) * .2).bfloat16()
        d.p = {"fc.weight": rand(16, 16), "hidden_norm.weight": torch.ones(16).bfloat16()}
        for layer in range(2):
            prefix = f"layers.{layer}.self_attn."
            d.p.update({prefix + "k_proj.weight": rand(4, 16),
                        prefix + "v_proj.weight": rand(4, 16),
                        prefix + "k_norm.weight": torch.ones(4).bfloat16()})
        # Both the reference weights and the production merged context projection.
        for merged in (False, True):
            d.context_kv = torch.cat([d.p[f"layers.{layer}.self_attn.{kind}_proj.weight"]
                                     for layer in range(2) for kind in ("k", "v")]) if merged else None
            for start, length, ends in ((0, 31, ()), (21, 33, (11,)),
                                        (1023, 257, (33, 96, 160, 256)),
                                        (2048, 192, (64, 128))):
                with self.subTest(merged=merged, start=start, length=length):
                    ring = rand(2, 2, 70, 1, 4)  # includes the decode scratch tail
                    aux = rand(length, 16)
                    cuts = tuple((end, i) for i, end in enumerate(ends))
                    baseline = original(d, ring, start, aux, cuts)
                    actual = incremental(d, ring, start, aux, cuts)
                    for result in (actual,):
                        for got, expected in [(result.ring, baseline.ring)] + [
                                (result.snaps[i], baseline.snaps[i]) for i in baseline.snaps]:
                            torch.testing.assert_close(got, expected, atol=0, rtol=0)
                            self.assertTrue(torch.equal(got[:, :, 64:], ring[:, :, 64:]))

    def test_short_tail_does_not_switch_context_rows_to_decode_precision(self):
        class Projection:
            def observe(self, ring, positions, aux):
                # Deliberately different W4/FP8 results, as the real dispatcher is.
                n = min(2048, len(positions))
                ring[positions[-n:] % 2048] = aux[-n:, 0] + (100000 if n <= 32 else 0)

        d = Projection()
        for ends, length in (((16, 768), 769), ((32, 768), 800), ((33, 768), 801),
                             ((768, 1536), 1540), ((), 31), ((768, 2304), 6912)):
            with self.subTest(ends=ends, length=length):
                ring = torch.full((2048,), -1.)
                aux = torch.arange(length).float()[:, None]
                cuts = tuple((end, i) for i, end in enumerate(ends))
                baseline = original(d, ring, 1311, aux, cuts)
                actual = incremental(d, ring, 1311, aux, cuts)
                self.assertTrue(torch.equal(actual.ring, baseline.ring))
                for snap in baseline.snaps:
                    self.assertTrue(torch.equal(actual.snaps[snap], baseline.snaps[snap]))


if __name__ == "__main__":
    unittest.main()
