"""CPU lifecycle/dispatch tests; arithmetic differential probe is separate.

The small source fixture retains the official Indexer.forward and selector
verbatim (HF model SHA 4e9ae236...). Only the test patches the expected whole-
file SHA; the public adapter API has no source-pin override. No GPU is used.
"""
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "overlay/modules/dsv41_model"
sys.path.insert(0, str(MODULES))
import torch
import dsv41_indexer as core
import dsv41_reference_adapter as adapter

REFERENCE_FIXTURE = r'''import torch
import torch.nn.functional as F

class Indexer(torch.nn.Module):
    def forward(self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor, start_pos: int, offset: int):
        """`latent` is this layer's RoPE-free compressed latent, None when this layer does not
        compress or when its current group is still incomplete. An index-key owner turns it into
        index keys here, which has to happen before Attention overwrites that same storage with
        the RoPE'd, quantized values."""
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        # latent is None while a group is still filling up, so there is nothing to publish yet
        if self.owns_k and latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = self.k_norm(self.wk(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant(k, fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            shared_attn.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        fp4_act_quant(q, fp4_block_size, True)

        index_k = shared_attn.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            dist.all_reduce(index_score)

        # how many compressed positions each query can see: a block becomes visible once the query
        # has passed its last token. One query per decode step, so there it is just a number.
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared_attn.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            # level two: score with our own weights, but only inside the source's candidate blocks
            index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)

        # top-k by score, re-sorted into position order; unreachable -> -1, rest shifted by offset
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is [..., n_positions] with positions the query cannot reach already at -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a plain int during
    decode, or broadcasts against logits' leading dims during prefill. Returns a bool mask shaped
    like `logits`, so the layers consuming it just mask and never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
'''

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.py"
        self.path.write_text(REFERENCE_FIXTURE)
        self.ref = ModuleType("official_reference_fixture")
        self.ref.__file__ = str(self.path)
        exec(compile(REFERENCE_FIXTURE, str(self.path), "exec"), vars(self.ref))
        self.events = []
        self.reductions = []
        self.ref.world_size = 1
        self.ref.fp4_block_size = 32
        self.ref.shared_attn = SimpleNamespace(index_k=None, candidates=None)
        self.ref.apply_rotary_emb = lambda value, freqs: self.events.append(("rope", tuple(value.shape)))
        self.ref.fp4_act_quant = lambda value, *args: self.events.append(("quant", tuple(value.shape)))
        self.ref.dist = SimpleNamespace(all_reduce=lambda value: self.reductions.append(value.shape))
        self.model = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(indexer=None)) for _ in range(40)])
        for layer in (20, 24, 28, 32, 36):
            instance = self.ref.Indexer()
            attrs = dict(compress_ratio=1, dim=5120, n_heads=32, n_local_heads=32,
                         index_head_dim=128, rope_head_dim=64, index_topk=512,
                         q_lora_rank=1280, candidate_topk_blocks=2048, candidate_block_size=8,
                         owns_k=layer == 20, is_candidate_source=layer == 20,
                         uses_candidates=layer != 20, softmax_scale=128**-.5)
            for name, value in attrs.items():
                setattr(instance, name, value)
            instance.freqs_cis = torch.zeros(16464, 32)
            instance.wq_b = lambda qr: torch.zeros((*qr.shape[:2], 32 * 128), dtype=torch.bfloat16)
            instance.weights_proj = lambda x: torch.ones((*x.shape[:2], 32), dtype=torch.bfloat16)
            instance.wk = lambda x: x[..., :128]
            instance.k_norm = lambda x: x
            self.model.layers[layer].attn.indexer = instance
        self.pin = patch.object(adapter, "REFERENCE_SHA256", hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.handle = None
        self.addCleanup(self.restore)

    def restore(self):
        if self.handle is not None and self.handle.active:
            self.handle.restore()

    def install(self):
        self.handle = adapter.install_reference_indexer(self.model, self.ref, enabled=True)
        return self.handle

    def indexer(self, layer):
        return self.model.layers[layer].attn.indexer

    def inputs(self, width=16401, batch=1, queries=1, start=None):
        cache = torch.zeros(batch, width + 31, 128, dtype=torch.bfloat16)
        self.ref.shared_attn.index_k = cache
        self.indexer(20).k_cache = cache
        x = torch.zeros(batch, queries, 5120, dtype=torch.bfloat16)
        qr = torch.zeros(batch, queries, 1280, dtype=torch.bfloat16)
        return x, qr, None, width - queries if start is None else start, 128

    def test_default_off_source_and_loaded_code_fail_closed(self):
        before = self.indexer(20).forward
        inactive = adapter.install_reference_indexer(None, None)
        self.assertFalse(inactive.active)
        inactive.restore()
        self.assertEqual(before, self.indexer(20).forward)
        with self.assertRaises(ValueError):
            adapter.install_reference_indexer(self.model, self.ref, enabled=1)
        self.path.write_text(REFERENCE_FIXTURE + "# changed\n")
        with self.assertRaises(ValueError):
            self.install()
        self.path.write_text(REFERENCE_FIXTURE)
        self.ref.Indexer.forward = lambda *args: None
        with self.assertRaises(ValueError):
            self.install()

    def test_instance_only_install_restore_and_exact_geometry(self):
        original = self.ref.Indexer.forward
        self.indexer(24).index_topk = 511
        with self.assertRaises(ValueError):
            self.install()
        self.assertNotIn("forward", self.indexer(20).__dict__)
        self.indexer(24).index_topk = 512
        h = self.install()
        self.assertIs(self.ref.Indexer.forward, original)
        with self.assertRaises(RuntimeError):
            adapter.install_reference_indexer(self.model, self.ref, enabled=True)
        self.assertTrue(all("forward" in self.indexer(layer).__dict__ for layer in (20, 24, 28, 32, 36)))
        h.restore()
        h.restore()
        self.assertTrue(all("forward" not in self.indexer(layer).__dict__ for layer in (20, 24, 28, 32, 36)))
        self.assertIs(self.indexer(20).forward.__func__, original)

    def test_long_decode_actual_dispatch_all_layers_and_strided_key_cache(self):
        args = self.inputs(batch=1)
        baseline = {}
        for layer in (20, 24, 28, 32, 36):
            baseline[layer] = self.indexer(layer)(*args)
        h = self.install()
        observed = []
        original_scores = core.compact_index_scores
        def spy(q, key, weights, ids, *rest, **kwargs):
            observed.append((tuple(key.shape), key.stride(), key.data_ptr(), ids.shape[-1]))
            return original_scores(q, key, weights, ids, *rest, **kwargs)
        with patch.object(core, "compact_index_scores", spy):
            for layer in (20, 24, 28, 32, 36):
                self.assertTrue(torch.equal(self.indexer(layer)(*args), baseline[layer]))
        self.assertEqual(h.counters, dict(source_compact_steps=1, consumer_compact_calls=4, dense_fallback_calls=0))
        self.assertEqual(len(observed), 4)
        for shape, stride, ptr, capacity in observed:
            self.assertEqual(shape, (1, 16401, 128))
            self.assertNotEqual(stride[0], 16401 * 128)
            self.assertEqual(ptr, self.ref.shared_attn.index_k.data_ptr())
            self.assertEqual(capacity, 16384)
        self.assertIsNone(h._state)

    def test_short_prefill_and_multiquery_keep_original_and_clear_state(self):
        h = self.install()
        self.indexer(20)(*self.inputs())
        self.assertIsNotNone(h._state)
        for args in (self.inputs(width=127), self.inputs(width=2, queries=2, start=0),
                     self.inputs(width=16401, queries=2), self.inputs(batch=2)):
            result = self.indexer(20)(*args)
            expected = self.ref.Indexer.forward(self.indexer(20), *args)
            self.assertTrue(torch.equal(result, expected))
            self.assertIsNone(h._state)
        self.assertEqual(h.counters["source_compact_steps"], 1)
        self.assertEqual(h.counters["consumer_compact_calls"], 0)

    def test_stale_step_order_cache_and_candidate_poison_use_dense(self):
        h = self.install()
        for poison in ("step", "order", "cache", "ids", "mask"):
            args = self.inputs()
            self.indexer(20)(*args)
            layer = 24
            changed = args
            if poison == "step":
                changed = (*args[:3], args[3] - 1, args[4])
            elif poison == "order":
                layer = 28
            elif poison == "cache":
                self.ref.shared_attn.index_k.add_(1)
            elif poison == "ids":
                h._state["ids"].zero_()
            elif poison == "mask":
                self.ref.shared_attn.candidates.zero_()
            if poison == "step":
                # The original itself rejects a stale dense mask width. The
                # adapter must preserve that failure, not invent a valid mask.
                with self.assertRaises(RuntimeError):
                    self.indexer(layer)(*changed)
            else:
                self.indexer(layer)(*changed)
            self.assertIsNone(h._state)
        self.assertEqual(h.counters["consumer_compact_calls"], 0)
        self.assertEqual(h.counters["dense_fallback_calls"], 5)

    def test_original_key_rope_quant_publication_precedes_query(self):
        h = self.install()
        args = self.inputs()
        latent = torch.ones(1, 1, 512, dtype=torch.bfloat16)
        args = (*args[:2], latent, *args[3:])
        self.indexer(20)(*args)
        self.assertEqual(self.events, [("rope", (1, 1, 64)), ("quant", (1, 1, 128)),
                                      ("rope", (1, 1, 32, 64)), ("quant", (1, 1, 32, 128))])
        self.assertTrue(torch.equal(self.ref.shared_attn.index_k[:, args[3]], latent[:, 0, :128]))
        self.events.clear()
        self.indexer(24)(*args)
        self.assertEqual(self.events, [("rope", (1, 1, 32, 64)), ("quant", (1, 1, 32, 128))])
        self.assertEqual(h.counters["consumer_compact_calls"], 1)

    def test_tp4_local_state_mismatch_never_switches_collective_width(self):
        self.ref.world_size = 4
        for layer in (20, 24, 28, 32, 36):
            instance = self.indexer(layer)
            instance.n_local_heads = 8
            instance.wq_b = lambda qr: torch.zeros((*qr.shape[:2], 8 * 128), dtype=torch.bfloat16)
            instance.weights_proj = lambda x: torch.ones((*x.shape[:2], 8), dtype=torch.bfloat16)
        h = self.install()
        args = self.inputs()
        for poison in ("ids", "order"):
            self.indexer(20)(*args)
            before = list(self.reductions)
            if poison == "ids":
                h._state["ids"].zero_()
            with self.assertRaisesRegex(RuntimeError, "before the TP collective"):
                self.indexer(24 if poison == "ids" else 28)(*args)
            self.assertEqual(self.reductions, before)
            self.assertIsNone(h._state)
        self.assertEqual(h.counters["dense_fallback_calls"], 0)
        short = self.inputs(width=127)
        self.indexer(20)(*short)
        self.indexer(24)(*short)
        self.assertEqual(self.reductions[-2:], [torch.Size([1, 1, 127])] * 2)
        before = list(self.reductions)
        self.ref.world_size = 1
        with self.assertRaisesRegex(RuntimeError, "shared runtime or TP size changed"):
            self.indexer(20)(*short)
        self.ref.world_size = 4
        self.ref.shared_attn = SimpleNamespace(index_k=None, candidates=None)
        with self.assertRaisesRegex(RuntimeError, "shared runtime or TP size changed"):
            self.indexer(24)(*short)
        self.assertEqual(self.reductions, before)

    def test_capture_rejected_and_external_forward_not_clobbered(self):
        h = self.install()
        h._state = {"old": True}
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True), \
                patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()):
            with self.assertRaises(RuntimeError):
                self.indexer(20)(SimpleNamespace(is_cuda=True, device="cuda:1"), None, None, 1, 0)
        self.assertIsNone(h._state)
        installed = self.indexer(24).__dict__["forward"]
        self.indexer(24).forward = lambda *args: None
        with self.assertRaises(RuntimeError):
            h.restore()
        self.indexer(24).forward = installed
        h.restore()


if __name__ == "__main__":
    unittest.main()
