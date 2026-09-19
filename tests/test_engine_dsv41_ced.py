"""engine/modules/sparse_indexer's CED half: DSv4.1's key compression and its candidate stage, against the
implementation they came from.

The oracle here is the retired overlay's -- `overlay/modules/dsv41_model/dsv41_compressor.py` and
`dsv41_indexer.py`, removed with the vLLM stack (#1152) and read back out of git history at 11c779a^ -- quoted below,
because that is what the 09-10 probes held bit-identical to the vendor's own classes
(`probes/dsv41_compressor_diff.py`: prefill, ragged prefill, and decode across a group boundary;
`measurements/dsv41_indexer_20260910`). What this file judges is that the module form computes the same bytes over
those same cases, so the arithmetic survived the move rather than being rewritten on the way.

No accelerator, no checkpoint: the weights are random and the equality is exact.
"""
import importlib.util
import unittest

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


# --------------------------------------------------------------------------------------------------------------------
# the retired overlay's arithmetic, quoted (11c779a^)
# --------------------------------------------------------------------------------------------------------------------
class Compressor:
    """overlay/modules/dsv41_model/dsv41_compressor.py: stateless over prefill, stateful across decode steps."""

    def __init__(self, hidden, head_dim, compress_ratio, norm_eps, max_batch_size=1):
        self.ratio = int(compress_ratio)
        self.head_dim, self.eps, self.hidden = head_dim, norm_eps, hidden
        if self.ratio > 1:
            shape = (max_batch_size, self.ratio, head_dim)
            self.kv_state = torch.zeros(shape, dtype=torch.float32)
            self.score_state = torch.full(shape, -torch.inf, dtype=torch.float32)
        else:
            self.kv_state = self.score_state = None

    def _rms(self, x, weight):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * weight.float()).type_as(x)

    def forward(self, x, start_pos, wkv, wgate, norm_weight):
        F = torch.nn.functional
        bsz, seqlen, _ = x.shape
        ratio, dtype = self.ratio, x.dtype
        if ratio == 1:
            return self._rms(F.linear(x, wkv.to(x.dtype)), norm_weight)
        xf = x.float()
        kv = F.linear(xf, wkv.float())
        score = F.linear(xf, wgate.float())
        if start_pos == 0:
            should = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                kv, tail_kv = kv.split([cutoff, remainder], dim=1)
                score, tail_score = score.split([cutoff, remainder], dim=1)
                self.kv_state[:bsz, :remainder] = tail_kv
                self.score_state[:bsz, :remainder] = tail_score
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should:
            return None
        return self._rms(kv.to(dtype), norm_weight)


def select_candidate_ids(full_scores, compress_lens, topk_blocks=2048, block_size=8, *, return_mask=False):
    """overlay/modules/dsv41_model/dsv41_indexer.py: pad -> amax -> newest-block +inf -> torch.topk."""
    F = torch.nn.functional
    width = full_scores.shape[-1]
    if isinstance(compress_lens, int):
        lengths = torch.full((*full_scores.shape[:-1], 1), compress_lens, dtype=torch.int64)
    else:
        lengths = torch.broadcast_to(compress_lens, (*full_scores.shape[:-1], 1))
    if width == 0:
        ids = torch.empty_like(full_scores, dtype=torch.int32)
        return (ids, torch.empty_like(full_scores, dtype=torch.bool)) if return_mask else ids
    scores = F.pad(full_scores, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    last = (lengths - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    reachable = top.values > -torch.inf
    positions = top.indices.unsqueeze(-1) * block_size + torch.arange(block_size)
    valid = reachable.unsqueeze(-1) & (positions < width)
    positions = positions.masked_fill(~valid, width).flatten(-2).sort(dim=-1).values
    positions = positions[..., : min(width, topk_blocks * block_size)]
    ids = positions.masked_fill(positions == width, -1).to(torch.int32).contiguous()
    if not return_mask:
        return ids
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, reachable)
    return ids, keep.repeat_interleave(block_size, dim=-1)[..., :width]


# --------------------------------------------------------------------------------------------------------------------


@unittest.skipUnless(torch is not None, "requires torch")
class CompressorTests(unittest.TestCase):
    H, D, EPS = 24, 8, 1e-20                                   # the checkpoint's rms_norm_eps is 1e-20

    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.wkv = (torch.randn(self.D, self.H, generator=g) * 0.1).to(torch.bfloat16)
        self.wgate = (torch.randn(self.D, self.H, generator=g) * 0.1).to(torch.bfloat16)
        self.norm = (torch.randn(self.D, generator=g) * 0.2 + 1).to(torch.bfloat16)
        self.x = (torch.randn(40, self.H, generator=g) * 0.5).to(torch.bfloat16)

    def ours(self, rows, ratio, tail=None):
        from engine.modules.sparse_indexer import ced_compress
        return ced_compress(rows, self.wkv, self.wgate if ratio > 1 else None, self.norm,
                            ratio=ratio, eps=self.EPS, tail=tail)

    def theirs(self, ratio):
        return Compressor(self.H, self.D, ratio, self.EPS)

    def test_a_whole_prefill_pools_every_group_the_same_way(self):
        for ratio in (2, 4):
            with self.subTest(ratio=ratio):
                rows = self.x[: 8 * ratio]
                want = self.theirs(ratio).forward(rows[None], 0, self.wkv, self.wgate, self.norm)
                keys, tail = self.ours(rows, ratio)
                self.assertIsNone(tail)
                self.assertEqual((keys.dtype, tuple(keys.shape)), (torch.bfloat16, (8, self.D)))
                self.assertTrue(torch.equal(keys, want[0]))

    def test_a_ragged_prefill_keeps_the_tail_and_the_next_call_completes_it(self):
        """The retired class parks the remainder in its state and the following decode steps fill it; ours hands the
        same rows back as `tail`, so the group that completes is the same group."""
        ratio, cut = 2, 9
        theirs = self.theirs(ratio)
        want = theirs.forward(self.x[:cut][None], 0, self.wkv, self.wgate, self.norm)
        keys, tail = self.ours(self.x[:cut], ratio)
        self.assertTrue(torch.equal(keys, want[0]))
        self.assertEqual(tuple(tail[0].shape), (1, self.D))                      # one row parked
        nxt = theirs.forward(self.x[cut:cut + 1][None], cut, self.wkv, self.wgate, self.norm)
        keys, tail = self.ours(self.x[cut:cut + 1], ratio, tail=tail)
        self.assertTrue(torch.equal(keys, nxt[0]))
        self.assertIsNone(tail)

    def test_decode_yields_a_key_only_where_the_group_closes(self):
        ratio = 4
        theirs = self.theirs(ratio)
        tail = None
        for pos in range(13):
            got, tail = self.ours(self.x[pos:pos + 1], ratio, tail=tail)
            want = theirs.forward(self.x[pos:pos + 1][None], pos, self.wkv, self.wgate, self.norm)
            closes = (pos + 1) % ratio == 0
            with self.subTest(pos=pos):
                self.assertEqual(got.shape[0], 1 if closes else 0)
                self.assertEqual(want is not None, closes)
                if closes:
                    self.assertTrue(torch.equal(got, want[0]))
                    self.assertIsNone(tail)

    def test_prefill_then_decode_is_the_same_as_one_prefill(self):
        ratio = 2
        whole, tail = self.ours(self.x[:12], ratio)
        self.assertIsNone(tail)
        first, tail = self.ours(self.x[:7], ratio)
        rest, tail = self.ours(self.x[7:12], ratio, tail=tail)
        self.assertTrue(torch.equal(torch.cat([first, rest]), whole))
        self.assertIsNone(tail)

    def test_ratio_one_is_the_other_path(self):
        """No gate, no fp32 pooling, no state -- and the reference builds no wgate for those layers."""
        rows = self.x[:5]
        want = self.theirs(1).forward(rows[None], 0, self.wkv, self.wgate, self.norm)
        keys, tail = self.ours(rows, 1)
        self.assertIsNone(tail)
        self.assertTrue(torch.equal(keys, want[0]))
        with self.assertRaises(ValueError):
            self.ours(rows, 1, tail=(torch.zeros(1, self.D), torch.zeros(1, self.D)))

    def test_it_refuses_a_ratio_below_one_and_a_missing_gate(self):
        from engine.modules.sparse_indexer import ced_compress
        with self.assertRaises(ValueError):
            ced_compress(self.x[:4], self.wkv, self.wgate, self.norm, ratio=0, eps=self.EPS)
        with self.assertRaisesRegex(ValueError, "wgate"):
            ced_compress(self.x[:4], self.wkv, None, self.norm, ratio=2, eps=self.EPS)


@unittest.skipUnless(torch is not None, "requires torch")
class CandidateTests(unittest.TestCase):
    def logits(self, queries, positions, seed=3):
        g = torch.Generator().manual_seed(seed)
        s = torch.randn(1, queries, positions, generator=g)
        cols = torch.arange(positions)
        return s.masked_fill(cols[None, None, :] > torch.arange(queries)[None, :, None], float("-inf"))

    def check(self, logits, lengths, *, topk_blocks, block_size):
        from engine.modules.sparse_indexer import ced_candidate_blocks
        want_ids, want_mask = select_candidate_ids(logits, lengths, topk_blocks, block_size, return_mask=True)
        ids = ced_candidate_blocks(logits, lengths, topk_blocks=topk_blocks, block_size=block_size)
        both = ced_candidate_blocks(logits, lengths, topk_blocks=topk_blocks, block_size=block_size, mask=True)
        self.assertTrue(torch.equal(ids, want_ids))
        self.assertTrue(torch.equal(both[0], want_ids) and torch.equal(both[1], want_mask))
        return ids

    def test_the_ids_are_the_retired_selections(self):
        for positions, topk_blocks, block_size in ((64, 4, 8), (64, 2, 8), (37, 3, 8), (16, 99, 8), (24, 2, 4)):
            with self.subTest(positions=positions, topk=topk_blocks, block=block_size):
                self.check(self.logits(6, positions), positions, topk_blocks=topk_blocks, block_size=block_size)

    def test_a_per_query_length_broadcasts_like_the_references(self):
        queries, positions = 5, 48
        lengths = torch.arange(1, queries + 1, dtype=torch.int64).reshape(1, queries, 1) * 8
        self.check(self.logits(queries, positions), lengths, topk_blocks=2, block_size=8)

    def test_the_ids_are_ascending_minus_one_padded_and_hold_the_newest_block(self):
        positions, block = 40, 8
        lengths = 33                                                   # the newest block is 33 // 8 == 4
        ids = self.check(self.logits(6, positions), lengths, topk_blocks=2, block_size=block)
        row = ids[0, 5]
        real = row[row >= 0]
        self.assertTrue(torch.equal(real, real.sort().values))         # ascending
        self.assertTrue(bool((row[row < 0] == -1).all()))              # only -1 pads
        self.assertTrue(set(range(32, 40)).issubset(set(real.tolist())))
        self.assertEqual(tuple(ids.shape), (1, 6, min(positions, 2 * block)))

    def test_an_empty_width_and_the_argument_contracts(self):
        from engine.modules.sparse_indexer import ced_candidate_blocks
        empty = torch.zeros(1, 3, 0)
        self.assertEqual(tuple(ced_candidate_blocks(empty, 0, topk_blocks=2, block_size=8).shape), (1, 3, 0))
        with self.assertRaises(ValueError):
            ced_candidate_blocks(torch.zeros(4, dtype=torch.int32), 4, topk_blocks=2, block_size=8)
        with self.assertRaises(ValueError):
            ced_candidate_blocks(self.logits(2, 8), 8, topk_blocks=0, block_size=8)


if __name__ == "__main__":
    unittest.main()
