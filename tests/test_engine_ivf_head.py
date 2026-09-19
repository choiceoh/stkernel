"""engine/kernels/dense/ivf_head: the drafter's argmax from an inverted-file index over an FP8 head's rows -- the index
(balanced, every row once), the argmax (every cluster probed = the head's own argmax), and where the net and the
fleet take it.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= python3 -m unittest tests.test_engine_ivf_head
"""
import importlib.util
import os
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
READY = torch is not None and importlib.util.find_spec("triton") is not None
DEVICE = "cpu" if INTERPRET else "cuda"


def head(n=640, k=256, seed=0, device="cpu"):
    """A head in FP8Linear's recipe: e4m3 rows, a power-of-two scale a 128 x 128 block."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=gen) * 0.05
    blocks = w.view(n // 128, 128, k // 128, 128).abs().amax((1, 3)).clamp_min(1e-4)
    ws = torch.exp2(torch.ceil(torch.log2(blocks / 448.0)))
    wq = (w / ws.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    return wq.to(device), ws.to(device)


def ids(key):
    return (0xffffffff - (key & 0xffffffff)).tolist()


@unittest.skipUnless(READY, "torch and triton required")
class IndexTests(unittest.TestCase):
    def test_every_row_once_and_no_cluster_over_its_cap(self):
        from engine.kernels.dense import ivf_head
        index = ivf_head.build(head(), clusters=16, probes=4, iters=3)
        members = index.members[index.members >= 0]
        self.assertEqual(sorted(members.tolist()), list(range(640)))
        self.assertEqual(index.cap, -(-int(1.15 * 640) // 16))
        self.assertEqual(tuple(index.centroids.shape), (16, 256))
        self.assertEqual(index.centroids.dtype, torch.bfloat16)

    def test_rows_limits_the_index(self):
        from engine.kernels.dense import ivf_head
        index = ivf_head.build(head(), clusters=8, probes=2, rows=500, iters=2)
        self.assertEqual(sorted(index.members[index.members >= 0].tolist()), list(range(500)))

    def test_a_seed_builds_the_same_index(self):
        from engine.kernels.dense import ivf_head
        a = ivf_head.build(head(), clusters=8, probes=2, iters=2)
        b = ivf_head.build(head(), clusters=8, probes=2, iters=2)
        self.assertTrue(torch.equal(a.members, b.members) and torch.equal(a.centroids, b.centroids))

    def test_it_refuses_more_probes_than_clusters(self):
        from engine.kernels.dense import ivf_head
        with self.assertRaises(ValueError):
            ivf_head.build(head(), clusters=8, probes=9)


@unittest.skipUnless(READY and (INTERPRET or (torch is not None and torch.cuda.is_available())),
                     "CUDA or Triton interpreter required")
class ArgmaxTests(unittest.TestCase):
    def test_every_cluster_probed_is_the_head_s_argmax(self):
        from engine.kernels.dense import ivf_head
        weight = head(device=DEVICE)
        index = ivf_head.build(weight, clusters=16, probes=16, iters=3)
        h = torch.randn(5, 256, generator=torch.Generator().manual_seed(3)).bfloat16().to(DEVICE)
        for start, valid in ((0, 640), (1000, 633)):
            got = ivf_head.argmax_key(index, h, start, valid)
            want = ivf_head.exact_key(weight, h, start, valid)
            with self.subTest(start=start, valid=valid):
                self.assertEqual(ids(got), ids(want))
                if not INTERPRET:                 # the interpreter truncates BF16 casts: its scores differ in the last bit
                    self.assertTrue(torch.equal(got, want))

    def test_a_row_past_valid_is_never_drafted(self):
        from engine.kernels.dense import ivf_head
        weight = head(device=DEVICE)
        index = ivf_head.build(weight, clusters=8, probes=8, iters=2)
        wq, _ = weight
        h = wq[630:634].float().bfloat16()                  # rows that point at themselves, all past `valid`
        self.assertTrue(all(i < 600 for i in ids(ivf_head.argmax_key(index, h, 0, 600))))

    def test_few_probes_answer_from_the_probed_clusters(self):
        from engine.kernels.dense import ivf_head
        weight = head(device=DEVICE)
        index = ivf_head.build(weight, clusters=16, probes=2, iters=3)
        h = torch.randn(4, 256, generator=torch.Generator().manual_seed(5)).bfloat16().to(DEVICE)
        got = ids(ivf_head.argmax_key(index, h, 0, 640))
        scores = h.float() @ index.centroids.float().t()
        for row, token in enumerate(got):
            probed = set(scores[row].topk(2).indices.tolist())
            owner = int((index.members == token).nonzero()[0, 0])
            self.assertIn(owner, probed)


@unittest.skipUnless(torch is not None, "torch required")
class ServingTests(unittest.TestCase):
    def test_the_net_drafts_through_the_index_only_when_prepared(self):
        from unittest import mock
        from engine.profiles.qwen38.net import Qwen38Net
        net = object.__new__(Qwen38Net)
        net.draft_index = None
        with mock.patch.object(Qwen38Net, "head_tokens", return_value=torch.tensor([7])) as full:
            self.assertEqual(net.draft_tokens(torch.zeros(1, 4)).tolist(), [7])
            full.assert_called_once()

    def test_every_draft_pick_goes_through_draft_tokens(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / "engine/profiles/qwen38"
        chain = (root / "decode_graphs.py").read_text()
        body = chain[chain.index("def draft_chain("):]
        self.assertNotIn("head_tokens", body)
        self.assertEqual(body.count("net.draft_tokens("), 2)
        adapter = (root / "adapter.py").read_text()
        head = adapter[adapter.index("    def _head(self"):adapter.index("    def _run_waiting")]
        self.assertIn("self.net.draft_tokens(hidden)", head)

    def test_the_fleet_reads_clusters_over_probes(self):
        from engine.profiles.qwen38.fleet import draft_index
        self.assertIsNone(draft_index(None))
        self.assertEqual(draft_index("1024/32"), (1024, 32))
        for bad in ("1024", "32/1024", "a/b"):
            with self.assertRaises(SystemExit):
                draft_index(bad)


if __name__ == "__main__":
    unittest.main()
