"""A feature whose cached rows live on another layer (engine/base/composition.Feature.rows_at): kept once per owner,
addressed by the owner, refused anywhere else.

DeepSeek-V4.1 is why the hook exists -- its compressed KV is produced by the layers `kv_source_layer_ids` names and
attended by the layers after them, so a lane per layer would hold the same rows several times -- but nothing here is
DSv4.1's: the mechanism is the composition's, the mapping is a profile's fact (engine/DSV41_COMPOSITION.md). Every
feature GLM-5.3 and Qwen3.8 have declares no `rows_at` and is kept exactly as before, which the first case pins.

CPU contracts, no accelerator.
"""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

HID = 8


def feature(rows_at=None, key="shared_rows", width=4):
    """A mixer that keeps `width` bytes a token, optionally on another layer's lane."""
    from engine.base.cache_spec import PagedSpec

    class Mixer:
        def __call__(self, layer, x, step, state):
            state.put_rows(self.owner(layer), key, step.segments[0].seq, x[:, :width])
            return x

        def owner(self, layer):
            return layer if rows_at is None else rows_at(layer)

        def cache_specs(self, layers):
            return [PagedSpec("shared rows", len(layers), width * 2, "a test lane",
                              key=key, dtype="bfloat16", shape=(width,))]

    if rows_at is not None:
        Mixer.rows_at = staticmethod(rows_at)
    return Mixer()


def composition(plan_layers, mixer):
    from engine.base.composition import Composition, Layer, Plan
    embed = torch.zeros(16, HID)
    return Composition(Plan(tuple(Layer("mix", "mlp") for _ in range(plan_layers))),
                       embed=lambda ids: embed[ids], residual=_PassThrough(),
                       features={"mix": mixer, "mlp": lambda layer, x, step, state: x},
                       head=lambda h: h)


class _PassThrough:
    """The smallest residual form: the state IS the hidden rows."""
    def open(self, x):
        return x

    def enter(self, layer, site, h, step=None, state=None):
        return h, None

    def leave(self, layer, site, out, carry, step=None, state=None):
        return out

    def close(self, h):
        return h


@unittest.skipUnless(torch is not None, "requires torch")
class SpecTests(unittest.TestCase):
    def test_a_feature_without_rows_at_is_kept_once_per_layer_it_runs_on(self):
        comp = composition(4, feature())
        paged, slots = comp.cache_specs()
        self.assertEqual(([s.key for s in paged], slots), (["shared_rows"], []))
        self.assertEqual(paged[0].layers, 4)
        self.assertEqual(comp.spec_layers(), {"shared_rows": [0, 1, 2, 3]})

    def test_rows_at_keeps_one_lane_per_owner(self):
        """Six layers, two owners: the lane is kept twice, not six times, and the owners are what the store indexes."""
        comp = composition(6, feature(rows_at=lambda layer: 0 if layer < 4 else 4))
        paged, _ = comp.cache_specs()
        self.assertEqual(paged[0].layers, 2)
        self.assertEqual(comp.spec_layers(), {"shared_rows": [0, 4]})

    def test_the_owners_are_sorted_and_counted_once(self):
        comp = composition(5, feature(rows_at=lambda layer: (4, 4, 0, 4, 0)[layer]))
        self.assertEqual(comp.spec_layers(), {"shared_rows": [0, 4]})

    def test_a_drafters_offset_moves_the_owners_with_it(self):
        """A head's plan layer i is model layer offset + i, and `rows_at` sees the model layer."""
        from engine.base.composition import Composition, Layer, Plan
        comp = composition(4, feature(rows_at=lambda layer: layer - layer % 2))
        moved = Composition(comp.plan, embed=comp.embed, residual=comp.residual, features=comp.features,
                            head=comp.head, offset=10)
        self.assertEqual(comp.spec_layers(), {"shared_rows": [0, 2]})
        self.assertEqual(moved.spec_layers(), {"shared_rows": [10, 12]})

    def test_an_owner_that_is_not_a_model_layer_is_refused(self):
        for bad in (None, 1.0, "0"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "rows_at"):
                composition(2, feature(rows_at=lambda layer: bad)).cache_specs()


@unittest.skipUnless(torch is not None, "requires torch")
class StoreTests(unittest.TestCase):
    def layout(self, comp):
        from engine.base.composed import Layout
        paged, slots = comp.cache_specs()
        return Layout(paged, slots, 4, comp.spec_layers())

    def test_the_store_gives_one_region_per_owner_and_refuses_the_others_by_name(self):
        """`Layout.region` is where a consumer that forgot to address the owner fails -- loudly, at its read."""
        layout = self.layout(composition(6, feature(rows_at=lambda layer: 0 if layer < 4 else 4)))
        self.assertEqual((layout.region("shared_rows", 0), layout.region("shared_rows", 4)), (0, 1))
        for consumer in (1, 2, 3, 5):
            with self.subTest(layer=consumer), self.assertRaisesRegex(ValueError, "not kept for layer"):
                layout.region("shared_rows", consumer)

    def test_a_lane_per_layer_still_carves_a_region_per_layer(self):
        layout = self.layout(composition(3, feature()))
        self.assertEqual([layout.region("shared_rows", i) for i in range(3)], [0, 1, 2])


@unittest.skipUnless(torch is not None, "requires torch")
class ForwardTests(unittest.TestCase):
    def test_the_layers_sharing_a_lane_read_what_the_owner_wrote(self):
        """Four layers write through one owner: the state holds one lane of four tokens' rows, not four lanes."""
        from engine.base.composition import State, Step
        comp = composition(4, feature(rows_at=lambda layer: 0))
        state = State()
        ids = torch.zeros(3, dtype=torch.int64)
        comp.forward(Step.of([(0, 0, ids)]), state, logits="all")
        self.assertEqual(sorted(k[:2] for k in state._rows), [(0, "shared_rows")])
        self.assertEqual(state.rows(0, "shared_rows", 0, 12).shape[0], 12)       # 4 layers x 3 tokens, one lane
        with self.assertRaises(ValueError):
            state.rows(1, "shared_rows", 0, 1)

    def test_without_rows_at_every_layer_keeps_its_own(self):
        from engine.base.composition import State, Step
        comp = composition(4, feature())
        state = State()
        comp.forward(Step.of([(0, 0, torch.zeros(3, dtype=torch.int64))]), state, logits="all")
        self.assertEqual(sorted(k[:2] for k in state._rows), [(i, "shared_rows") for i in range(4)])
        self.assertEqual(state.rows(2, "shared_rows", 0, 3).shape[0], 3)


if __name__ == "__main__":
    unittest.main()
