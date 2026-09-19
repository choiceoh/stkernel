"""A Qwen3.8 verify step's host half reads nothing back off the device for the PLE staging.

`decode_graphs.TargetGraphs.run` staged a captured step's PLE rows from `ids.tolist()` -- the step's ids, which the host
had uploaded a moment before -- and `net.stage_ple` read each row's carried context off the slots' id rings: two device
reads and a launch sequence a step, with the GPU idle behind them. `adapter.ServedModel` holds both (the ids ring holds
what was fed at each position, which is the row's own history), so its verify step hands them down:
`forward(..., host=(ids, carried))` -> `TargetGraphs.run(step, known=)` -> `stage_ple(..., carried=)`.

CPU only, with recording stand-ins for the graphs and the net: what is held is what reaches the staging, for full and
padded rows, and that a caller without a host copy (base ComposedModel.decode) still stages from the device.
"""
import importlib.util
import unittest
from types import SimpleNamespace

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None, "requires torch")
class StagingInputsTests(unittest.TestCase):
    def graphs(self, tokens: int):
        """TargetGraphs.run over stand-ins: a net that records what it is asked to stage, graphs that replay nothing."""
        from engine.profiles.qwen38.decode_graphs import TargetGraphs
        staged = []
        net = SimpleNamespace(ple_stage=SimpleNamespace(upload=lambda: None),
                              stage_ple=lambda slots, contexts, ids, t, caches, carried=None: staged.append(
                                  (list(slots), list(contexts), list(ids), t, carried)))
        g = TargetGraphs.__new__(TargetGraphs)
        g.net, g.caches, g.tokens, g.max_seqs, g.narrow_rows = net, SimpleNamespace(), tokens, 4, 0
        g._meta_host = torch.zeros(12, dtype=torch.int64)
        g._meta = g._meta_host.numpy()
        g.metadata = {}
        g.shape = lambda rows, end, t=None: (rows, tokens if t is None else t, 1)
        g.publish = lambda rows: None
        g.graphs = SimpleNamespace(run=lambda shape, fill: (torch.zeros(1), torch.zeros(1)))
        return g, staged

    def step(self, lengths):
        from engine.profiles.qwen38.net import Segment, Step
        segments, at = [], 0
        for i, n in enumerate(lengths):
            segments.append(Segment(i, i + 1, 10 * (i + 1), at, n))
            at += n
        return Step(torch.arange(100, 100 + at, dtype=torch.int64), tuple(segments))

    def test_the_hosts_ids_and_context_reach_the_staging_unread(self):
        g, staged = self.graphs(tokens=2)
        step = self.step([2, 2])
        flat, carried = [100, 101, 102, 103], [[7, 8], [9, 10]]
        ids = step.ids
        try:
            torch.Tensor.tolist, original = (lambda self: (_ for _ in ()).throw(AssertionError("read off the device"))), torch.Tensor.tolist
            g.run(step, known=(flat, carried))
        finally:
            torch.Tensor.tolist = original
        self.assertEqual(staged, [([1, 2], [10, 20], flat, 2, carried)])
        self.assertTrue(torch.equal(ids, step.ids))

    def test_a_padded_row_stages_its_last_token_again_as_the_graph_is_fed(self):
        g, staged = self.graphs(tokens=3)
        step = self.step([1, 3])                                        # the first row carries no drafts: padded to 3
        rows = g.run(step, known=([100, 101, 102, 103], [[1, 2], [3, 4]]))[2]
        self.assertEqual(staged[0][2], [100, 100, 100, 101, 102, 103])
        self.assertEqual(rows, [0, 3, 4, 5])
        del staged[:]
        g.run(step)                                                     # no host copy: the device's ids, no carried
        self.assertEqual(staged[0][2:], ([100, 100, 100, 101, 102, 103], 3, None))

    def test_the_composition_passes_the_host_copy_to_its_graphs_only(self):
        from engine.profiles.qwen38.adapter import ServedComposition
        seen = {}
        comp = ServedComposition.__new__(ServedComposition)
        comp.net = comp.caches = SimpleNamespace(pool=None)
        comp.graphs = SimpleNamespace(admits=lambda served, pool: True, tokens=1,
                                      run=lambda served, known=None: (seen.setdefault("host", known),
                                                                     torch.zeros(1, 4), torch.zeros(1, 4), None, 1)[1:])
        store = SimpleNamespace(check=lambda step: None, commit=lambda step: None, slot_of={0: 1})
        base = SimpleNamespace(ids=torch.zeros(1, dtype=torch.int64),
                               segments=(SimpleNamespace(seq=0, ctx=5, start=0, length=1),))
        comp.forward(base, store, logits="all", hidden=True, host=([9], [[1, 2]]))
        self.assertEqual(seen["host"], ([9], [[1, 2]]))


@unittest.skipUnless(torch is not None, "requires torch")
class CarriedContextTests(unittest.TestCase):
    def test_the_carried_tokens_are_the_rows_history_and_dead_before_it(self):
        from engine.modules.ngram_embedding import DEAD
        from engine.profiles.qwen38.adapter import _served_model_class
        Model = _served_model_class()
        model = Model.__new__(Model)
        model.composition = SimpleNamespace(net=SimpleNamespace(F=SimpleNamespace(ngram_size=3)))
        model.tokens = {4: [11, 12, 13, 14]}
        self.assertEqual(model._carried(4, 3), [12, 13])                # positions 1 and 2: what was fed there
        self.assertEqual(model._carried(4, 1), [DEAD, 11])
        self.assertEqual(model._carried(4, 0), [DEAD, DEAD])

    def test_the_verify_step_hands_them_down_with_its_ids(self):
        import inspect
        from engine.profiles.qwen38 import adapter
        source = inspect.getsource(adapter._served_model_class)
        self.assertIn("carried = [self._carried(seq, segment.ctx) for seq, segment in zip(seqs, segments)]", source)
        self.assertIn("hidden=True, host=(flat, carried))", source)


if __name__ == "__main__":
    unittest.main()
