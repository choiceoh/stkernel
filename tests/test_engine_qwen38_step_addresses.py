"""A captured Qwen3.8 step's addressing in one launch equals the torch composition it replaces (engine/QWEN38_CARRY.md Q2).

`Qwen38Net.step_meta` built a captured step's StepMeta from about forty launches, twice a step. On CUDA it now calls
engine/kernels/step_addresses.captured, one program a row; the composition stays the CPU's form. Every value is
integer arithmetic, so the kernel must reproduce every tensor byte for byte: positions across a block boundary, rows
closing an index-key group and rows that do not, raw-key-ring slots only for a row's last QSA_KEY_RING positions,
page-table entries past a row's reservation (-1) read as page 0, rows in a sequence order that is not the row order.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_step_addresses
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET else "cuda"
FIELDS = ("positions", "positions32", "rows_req", "page_table", "lengths", "starts", "slot_table", "kv_slots",
          "key_slots", "ring_slots")


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class CapturedAddressesTests(unittest.TestCase):
    BLOCK, RATIO = 768, 4

    def table(self, seqs, blocks):
        torch.manual_seed(819)
        table = torch.randint(0, 40000, (seqs, blocks + 2), dtype=torch.int32)
        table[:, blocks:] = -1                                       # past the bucket
        table[0, 1] = -1                                             # an unreserved entry inside the bucket
        return table

    def reference(self, contexts, slots, seqs, table, tokens, blocks):
        """The composition, run on CPU tensors (net.step_meta's CPU path)."""
        from engine.profiles.qwen38.net import Qwen38Net
        step = SimpleNamespace(captured=True, rows=contexts.numel(), tokens=tokens, blocks=blocks, contexts=contexts,
                               slots=slots, seqs=seqs, ids=torch.zeros(contexts.numel() * tokens, dtype=torch.int64))
        net = SimpleNamespace(F=SimpleNamespace(block=self.BLOCK, idx_ratio=self.RATIO))
        meta = Qwen38Net.step_meta(net, step, SimpleNamespace(block_table=table))
        return {name: getattr(meta, name) for name in FIELDS}

    def test_every_tensor_is_the_compositions(self):
        from engine.kernels import step_addresses
        from engine.profiles.qwen38.caches import QSA_KEY_RING
        cases = [((766,), (3,), (0,)), ((766, 5), (1, 3), (1, 0)), ((766, 5, 1530), (3, 1, 7), (2, 0, 1))]
        for contexts, slots, seqs in cases:
            for tokens in (1, 2, 4):
                with self.subTest(rows=len(contexts), tokens=tokens):
                    c, s, q = (torch.tensor(v, dtype=torch.int64) for v in (contexts, slots, seqs))
                    table = self.table(3, 2)
                    expected = self.reference(c, s, q, table, tokens, 2)
                    got = step_addresses.captured(c.to(DEVICE), s.to(DEVICE), q.to(DEVICE), table.to(DEVICE),
                                                  tokens=tokens, blocks=2, block=self.BLOCK, ratio=self.RATIO,
                                                  ring=QSA_KEY_RING)
                    for name, value in zip(FIELDS, got):
                        self.assertEqual(value.dtype, expected[name].dtype, name)
                        self.assertTrue(torch.equal(value.cpu(), expected[name]), name)

    def test_a_picture_s_rows_turn_at_their_delta(self):
        """With mRoPE deltas (a sequence whose prompt held a picture) the launch also writes each token's rotary
        position (cache position + its row's delta) and its group-first member's, as the composition does; the ten
        addressing tensors do not move."""
        from engine.kernels import step_addresses
        from engine.profiles.qwen38.caches import QSA_KEY_RING
        from engine.profiles.qwen38.net import Qwen38Net
        c, s, q = (torch.tensor(v, dtype=torch.int64) for v in ((766, 5, 1530), (3, 1, 7), (2, 0, 1)))
        deltas = torch.tensor([-1180, 0, -3], dtype=torch.int64)
        table = self.table(3, 2)
        for tokens in (1, 4):
            with self.subTest(tokens=tokens):
                step = SimpleNamespace(captured=True, rows=3, tokens=tokens, blocks=2, contexts=c, slots=s, seqs=q,
                                       deltas=deltas, ids=torch.zeros(3 * tokens, dtype=torch.int64))
                net = SimpleNamespace(F=SimpleNamespace(block=self.BLOCK, idx_ratio=self.RATIO))
                want = Qwen38Net.step_meta(net, step, SimpleNamespace(block_table=table))
                got = step_addresses.captured(c.to(DEVICE), s.to(DEVICE), q.to(DEVICE), table.to(DEVICE), tokens=tokens,
                                              blocks=2, block=self.BLOCK, ratio=self.RATIO, ring=QSA_KEY_RING,
                                              deltas=deltas.to(DEVICE))
                self.assertEqual(len(got), 12)
                for name, value in zip(FIELDS, got):
                    self.assertTrue(torch.equal(value.cpu(), getattr(want, name)), name)
                rope, first = got[10].cpu(), got[11].cpu()
                self.assertTrue(torch.equal(rope, want.rope) and torch.equal(first, want.rope_first))
                self.assertTrue(torch.equal(rope, want.positions + deltas.repeat_interleave(tokens)))
                self.assertTrue(torch.equal(first, rope - (self.RATIO - 1)))

    def test_the_captured_step_rows_are_views_of_one_tensor(self):
        """decode_graphs builds contexts, seqs and slots as rows of one [3, n] tensor: strided views."""
        from engine.kernels import step_addresses
        from engine.profiles.qwen38.caches import QSA_KEY_RING
        meta = torch.tensor([[766, 5, 1530], [2, 0, 1], [3, 1, 7]], dtype=torch.int64)
        contexts, seqs, slots = meta.unbind(0)
        table = self.table(3, 2)
        expected = self.reference(contexts.contiguous(), slots.contiguous(), seqs.contiguous(), table, 2, 2)
        got = step_addresses.captured(contexts.to(DEVICE), slots.to(DEVICE), seqs.to(DEVICE), table.to(DEVICE),
                                      tokens=2, blocks=2, block=self.BLOCK, ratio=self.RATIO, ring=QSA_KEY_RING)
        for name, value in zip(FIELDS, got):
            self.assertTrue(torch.equal(value.cpu(), expected[name]), name)


@unittest.skipUnless(torch is not None, "requires torch")
class RoutingTests(unittest.TestCase):
    def test_a_cuda_step_takes_the_launch_and_the_composition_stays_the_cpus(self):
        source = (ROOT / "engine/profiles/qwen38/net.py").read_text()
        self.assertIn('if getattr(step, "captured", False) and step.ids.is_cuda:', source)
        self.assertIn("step_addresses.captured(", source)

    def test_it_refuses_mismatched_inputs_before_a_launch(self):
        from engine.kernels import step_addresses
        c = torch.zeros(2, dtype=torch.int64)
        with self.assertRaises(ValueError):
            step_addresses.captured(c, c, c.int(), torch.zeros(2, 4, dtype=torch.int32), tokens=2, blocks=2, block=768,
                                    ratio=4, ring=8)
        with self.assertRaises(ValueError):
            step_addresses.captured(c, c, c, torch.zeros(2, 4, dtype=torch.int32), tokens=2, blocks=5, block=768,
                                    ratio=4, ring=8)


if __name__ == "__main__":
    unittest.main()
