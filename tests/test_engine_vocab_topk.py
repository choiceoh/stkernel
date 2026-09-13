"""Candidate communication, masking and pinned CUDA topk tie compatibility."""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec('torch'):
    import torch


@unittest.skipUnless(torch is not None, 'requires PyTorch')
class VocabTopkTests(unittest.TestCase):
    def compare(self, full, k, decodable=None, exact_ids=True):
        from engine.base.comm import LocalTP
        from engine.modules.vocab import topk
        width = full.shape[-1] // 4
        before = full.clone()
        packets = []
        tp = LocalTP(4)
        def select(comm):
            gather = comm.all_gather
            def record(packet, dim=-1):
                packets.append((tuple(packet.shape), packet.dtype))
                return gather(packet, dim=dim)
            comm.all_gather = record
            return topk(full[..., comm.rank*width:(comm.rank+1)*width], comm, comm.rank*width, k, decodable)
        actual = tp.run(select)
        masked = full.float().clone()
        if decodable is not None:
            masked[..., decodable:] = float('-inf')
        expected = masked.topk(k, dim=-1)
        for values, ids in actual:
            torch.testing.assert_close(values, expected.values, rtol=0, atol=0, equal_nan=True)
            if exact_ids:
                self.assertTrue(torch.equal(ids, expected.indices), (ids, expected.indices))
            self.assertTrue(torch.equal(ids, actual[0].indices))
            self.assertTrue(((ids >= 0) & (ids < full.shape[-1])).all())
        self.assertEqual(packets, [(tuple(full.shape[:-1])+(k,), torch.int64)]*4)
        torch.testing.assert_close(full, before, rtol=0, atol=0, equal_nan=True)

    def test_unique_strided_values_and_empty_shards_use_one_small_packet(self):
        gen = torch.Generator().manual_seed(347)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.stack([torch.randperm(96, generator=gen) for _ in range(5)]).to(dtype)[:, ::2] / 128
            for decodable in (None, 43, 23, 7):
                with self.subTest(dtype=dtype, decodable=decodable):
                    self.compare(x, 7, decodable)

    def test_padding_and_nonfinite_scores_never_produce_invalid_ids(self):
        x = torch.tensor([[float('-inf')]*32, [0., -0.]*16,
                          [float('nan'), float('inf'), 4., 3., 2., 1., 0., -1.]*4])
        # CPU tie indices are unspecified and differ from CUDA radix selection.
        for decodable in (None, 29, 7, 1):
            self.compare(x, 8, decodable, exact_ids=False)

    def test_invalid_arguments_fail_before_a_collective(self):
        from engine.modules.vocab import topk
        from types import SimpleNamespace
        comm = SimpleNamespace(world_size=4, all_gather=lambda *a, **k: self.fail('invalid collective'))
        for start, k, decodable in ((-1, 3, None), (1, 3, None), (32, 3, None),
                                    (0, 0, None), (0, 33, None), (0, True, None), (0, 3, 0)):
            with self.assertRaises(ValueError):
                topk(torch.zeros(2, 8), comm, start, k, decodable)
        with self.assertRaises(TypeError):
            topk(torch.zeros(2, 8, dtype=torch.int64), comm, 0, 3)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires CUDA')
    def test_actual_vocabulary_ties_and_masks_match_dense_cuda_topk(self):
        gen = torch.Generator(device='cuda').manual_seed(92)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x = torch.randn(5, 154880*2, device='cuda', generator=gen).to(dtype)[:, ::2]
            for decodable in (154880, 153880, 38710, 7):
                with self.subTest(dtype=dtype, decodable=decodable, mode='random'):
                    self.compare(x, 16, decodable)
            x.fill_(-1)
            x[:, ::3000] = 5
            x[1].zero_(); x[1, ::2] = -0.
            x[2].fill_(float('-inf'))
            x[3, ::3000] = float('inf')
            x[4, ::3000] = float('nan')
            for decodable in (154880, 153880, 38710, 7):
                with self.subTest(dtype=dtype, decodable=decodable, mode='ties'):
                    self.compare(x, 16, decodable)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires CUDA')
    def test_capture_replays_changing_candidates_without_rng_or_host_reads(self):
        from engine.base.comm import Comm
        from engine.modules.vocab import topk
        x = torch.zeros(5, 38720, device='cuda', dtype=torch.bfloat16)
        state = torch.cuda.get_rng_state().clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): topk(x, Comm(), 0, 16, 38710)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            values, ids = topk(x, Comm(), 0, 16, 38710)
        for offset in (0, 15000, 38680):
            x.fill_(-1); x[:, offset:offset+16] = 5; x[:, 38710:] = 100
            graph.replay()
            expected = x.float(); expected[:, 38710:] = float('-inf')
            oracle = expected.topk(16, dim=-1)
            self.assertTrue(torch.equal(ids, oracle.indices))
            self.assertTrue(torch.equal(values, oracle.values))
        self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
        graph.reset()




@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'the fused selection is the CUDA path')
class CandidateSelectionTests(unittest.TestCase):
    """`select` replaces the LOCAL step only (45차 §87).

    Its job is the set, not the order: the shard's k largest keys, which torch was asked for with
    `sorted=False`. The merge afterwards is still torch's dense topk, because that is what pins the tie order
    this module promises -- see the tests above, which compare ids against `full.topk(k)` itself.
    """
    def keys(self, rows, width, seed, span=1.0):
        from engine.kernels.common.vocab_candidates import pack
        gen = torch.Generator(device='cuda').manual_seed(seed)
        logits = (torch.randn(rows, width, device='cuda', generator=gen) * span).bfloat16()
        return pack(logits, 0, width)

    def test_it_is_the_same_set_torch_would_have_taken(self):
        from engine.kernels.common.vocab_candidates import select
        for rows, width, k in ((5, 38_720, 16), (1, 38_720, 16), (8, 4096, 4), (5, 17, 16), (3, 16, 16),
                               (5, 38_720, 1), (2, 2048, 32)):
            with self.subTest(rows=rows, width=width, k=k):
                for span in (1.0, 0.01):                 # a narrow span packs many equal bf16 scores together
                    keys = self.keys(rows, width, rows + width + k, span)
                    want = keys.topk(min(k, width), dim=-1, sorted=False).values.sort(-1).values
                    self.assertTrue(torch.equal(want, select(keys, min(k, width)).sort(-1).values))

    def test_it_returns_them_in_descending_order(self):
        from engine.kernels.common.vocab_candidates import select
        got = select(self.keys(4, 4096, 11), 16)
        self.assertTrue(bool((got[:, :-1] > got[:, 1:]).all()))       # keys are unique: strictly descending

    def test_a_shard_narrower_than_k_pads_with_the_sentinel(self):
        from engine.kernels.common.vocab_candidates import select
        got = select(self.keys(2, 8, 12), 16)
        self.assertEqual(tuple(got.shape), (2, 16))
        self.assertEqual(int((got == -(2**63)).sum()), 2 * 8)


if __name__ == '__main__':
    unittest.main()
