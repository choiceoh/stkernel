"""DFlash context retention and selection across the device/host boundary."""
import importlib.util
from types import SimpleNamespace
import unittest
from pathlib import Path

from tests.image_kernels import fused_sdpa

DRAFTER_HEAD_DIM = 4           # DrafterFacts(head_dim=...) below; the shape the pinned backends refuse here

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@unittest.skipUnless(torch is not None and fused_sdpa(DRAFTER_HEAD_DIM),
                     "requires PyTorch with a fused SDPA backend at the drafter's head size")
class DrafterTests(unittest.TestCase):
    def make_drafter(self):
        from engine.profiles.glm53.drafter import Drafter, DrafterFacts
        F = DrafterFacts(layers=1, hidden=16, heads=2, kv_heads=1, head_dim=4,
                         inter=16, rms_eps=1e-6, rope_theta=10000., window=8,
                         block=4, mask_id=20, conv_taps=2, conv_group=4,
                         sel_rank=4, sel_top_k=3, target_layers=(1,), k=3)
        return Drafter(F, SimpleNamespace(), 21)

    def test_long_observation_retains_the_newest_window(self):
        from engine.profiles.glm53.drafter import norm, norm_rope   # the fused forms the observe itself runs
        d = self.make_drafter()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(6)
        rand = lambda *shape: torch.randn(*shape, device=dev, generator=gen).bfloat16()
        d.p = {"fc.weight": rand(16, 16), "hidden_norm.weight": torch.ones(16, device=dev, dtype=torch.bfloat16),
               "layers.0.self_attn.k_proj.weight": rand(4, 16),
               "layers.0.self_attn.k_norm.weight": torch.ones(4, device=dev, dtype=torch.bfloat16),
               "layers.0.self_attn.v_proj.weight": rand(4, 16)}
        ring = torch.zeros(1, 2, 8, 1, 4, device=dev, dtype=torch.bfloat16)
        positions = torch.arange(3, 38, device=dev)
        aux = rand(35, 16)
        d.observe(ring, positions, aux)
        # Compute the retained positions directly, without repeated scatter ids.
        h = norm(torch.nn.functional.linear(aux[-8:], d.p["fc.weight"]),
                 d.p["hidden_norm.weight"], d.F.rms_eps).bfloat16()
        key = torch.nn.functional.linear(h, d.p["layers.0.self_attn.k_proj.weight"]).view(8, 1, 4)
        key = norm_rope(key, d.p["layers.0.self_attn.k_norm.weight"], d.F.rms_eps,
                        positions[-8:], d.F.rope_theta)
        value = torch.nn.functional.linear(h, d.p["layers.0.self_attn.v_proj.weight"]).view(8, 1, 4)
        self.assertTrue(torch.equal(ring[0, 0, positions[-8:] % 8], key))
        self.assertTrue(torch.equal(ring[0, 1, positions[-8:] % 8], value))

    def test_device_selector_matches_the_scalar_greedy_walk(self):
        d = self.make_drafter()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(71)
        rand = lambda *shape: torch.randn(*shape, device=dev, generator=gen).bfloat16()
        hidden, logits = rand(4, 16), rand(3, 21).float()
        d.block = lambda *args: hidden
        from engine.base.comm import Comm
        d.target.head_local = lambda h: logits.clone()
        d.target.comm, d.target.rank, d.target.vp = Comm(), 0, 21
        d.target.head = lambda h: self.fail("drafter gathered full-vocabulary logits")
        prefix = "candidate_selector."
        d.p = {prefix + "hidden_projection.weight": rand(4, 16),
               prefix + "predecessor_codebook": rand(21, 4),
               prefix + "successor_codebook": rand(21, 4)}
        anchor = 7
        unary, cand = logits.topk(3, dim=-1)
        projection = torch.nn.functional.linear(hidden[1:], d.p[prefix + "hidden_projection.weight"]).float()
        expected, previous = [], anchor
        for step in range(3):
            pred = d.p[prefix + "predecessor_codebook"][previous].float()
            succ = d.p[prefix + "successor_codebook"][cand[step]].float()
            score = unary[step] + ((pred * projection[step])[None, :] * succ).sum(-1)
            previous = int(cand[step, score.argmax()])
            expected.append(previous)
        actual = d.propose_tensor(torch.full((1,), anchor, device=dev, dtype=torch.int64), 9,
                                   torch.empty(0, device=dev))
        self.assertEqual(actual.tolist(), expected)

    def test_the_sampled_walk_hands_back_the_distribution_it_drew_from(self):
        """The accept test divides by this, so it has to be the one the pick came out of."""
        d = self.make_drafter()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(71)
        rand = lambda *shape: torch.randn(*shape, device=dev, generator=gen).bfloat16()
        hidden, logits = rand(4, 16), rand(3, 21).float()
        d.block = lambda *args: hidden
        from engine.base.comm import Comm
        d.target.head_local = lambda h: logits.clone()
        d.target.comm, d.target.rank, d.target.vp = Comm(), 0, 21
        prefix = "candidate_selector."
        d.p = {prefix + "hidden_projection.weight": rand(4, 16),
               prefix + "predecessor_codebook": rand(21, 4),
               prefix + "successor_codebook": rand(21, 4)}
        ring = torch.empty(0, device=dev)
        ids, dists = d.propose_sampled(7, 9, ring, 0.8, torch.Generator(device=dev).manual_seed(5), 21)
        same, same_dists = d.propose_sampled_tensor(torch.full((1,), 7, device=dev, dtype=torch.int64), 9, ring,
                                                    0.8, torch.Generator(device=dev).manual_seed(5), 21)
        self.assertEqual(ids, same.tolist(), "the host walk is the device walk, read back once at the end")
        self.assertTrue(torch.equal(dists, same_dists))
        for step, token in enumerate(ids):
            self.assertGreater(float(dists[step, token]), 0.0, "the pick has mass in what the verifier is given")
            self.assertAlmostEqual(float(dists[step].sum()), 1.0, places=4)
            self.assertLessEqual(int((dists[step] > 0).sum()), 3, "and the mass sits only on the candidates")

    def make_full_drafter(self, seed=3):
        """The tiny drafter with every checkpoint tensor random, a vocabulary-shaped fake target and a draft field."""
        from engine.base.comm import Comm
        from engine.profiles.glm53.drafter import ring_cells, specs
        d = self.make_drafter()
        F = d.F
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(seed)
        rand = lambda *shape: (torch.randn(*shape, device=dev, generator=gen) * 0.3).bfloat16()
        d.p = {s.name: (torch.ones(*s.shape, device=dev, dtype=torch.bfloat16) if s.name.endswith("norm.weight") else rand(*s.shape))
               for s in specs(F)}
        vocab = 21
        table, head = rand(vocab, F.hidden), rand(vocab, F.hidden)
        d.target = SimpleNamespace(embed=lambda ids: table[ids], head_local=lambda h: torch.nn.functional.linear(h, head),
                                   comm=Comm(), rank=0, vp=vocab)
        field = rand(4, F.layers, 2, ring_cells(F), F.kv_heads, F.head_dim)
        return d, field, dev

    def test_rows_propose_matches_the_per_row_walk(self):
        """Three rows in three slots -- two with a context shorter than the window, one longer -- at once: the same
        drafts as three per-row walks, and the block's hidden within bf16 of the per-row block."""
        d, field, dev = self.make_full_drafter()
        F = d.F
        slots = torch.tensor([1, 3, 2], device=dev)
        ctx = torch.tensor([5, 12, 3], device=dev)
        anchors = torch.tensor([7, 3, 9], device=dev)
        expect = [d.propose_tensor(anchors[i:i + 1], int(ctx[i]), field[int(slots[i])].clone()) for i in range(3)]
        got = d.propose_rows(field, slots, anchors, ctx)
        self.assertEqual(got.tolist(), [e.tolist() for e in expect])
        t = F.k + 1
        ids = torch.cat([anchors.view(3, 1), torch.full((3, F.k), F.mask_id, device=dev)], 1).reshape(-1)
        pos = (ctx.view(3, 1) + torch.arange(t, device=dev)).reshape(-1)
        rows = d.block_rows(ids, pos, slots, ctx, field, 3, t).view(3, t, -1)
        for i in range(3):
            one = d.block(ids.view(3, t)[i], pos.view(3, t)[i], field[int(slots[i])], int(ctx[i]))
            torch.testing.assert_close(rows[i].float(), one.float(), atol=6e-2, rtol=6e-2)
        self.assertTrue(torch.isfinite(rows.float()).all())

    def test_rows_observe_writes_the_valid_cells_in_place(self):
        d, field, dev = self.make_full_drafter(seed=4)
        F = d.F
        t = F.k + 1
        slots = torch.tensor([2, 1], device=dev)
        positions = torch.tensor([[5, 6, 7, 8], [13, 14, 15, 16]], device=dev)
        valid = torch.tensor([3, 0], device=dev)
        aux = (torch.randn(2 * t, F.hidden, device=dev) * 0.3).bfloat16()
        expect = field.clone()
        for i in range(2):
            d.observe_masked(expect[int(slots[i])], positions[i], aux[i * t:(i + 1) * t], valid[i])
        d.observe_rows(field, slots, positions, aux, valid)
        torch.testing.assert_close(field.float(), expect.float(), atol=2e-2, rtol=2e-2)
        self.assertTrue(torch.equal(field[3], expect[3]), "an untouched slot is untouched")
        self.assertTrue(torch.equal(field[1], expect[1]), "a row with no valid position writes nothing")

    def test_rows_sampled_walk_keeps_greedy_rows_greedy_and_hands_back_distributions(self):
        d, field, dev = self.make_full_drafter(seed=5)
        F = d.F
        slots = torch.tensor([1, 2, 3], device=dev)
        ctx = torch.tensor([9, 4, 30], device=dev)
        anchors = torch.tensor([2, 7, 11], device=dev)
        temps = torch.tensor([0.8, 0.0, 0.5], device=dev)
        greedy = d.propose_rows(field, slots, anchors, ctx)
        drafts, cand, q = d.propose_rows(field, slots, anchors, ctx, temps=temps, generator=torch.Generator(device=dev).manual_seed(9), vocab=21)
        # the candidates and their mass, not a row per position: the walk puts nothing outside them, so the
        # vocabulary-wide form was 21 (154,880 in production) numbers to carry sel_top_k of them
        self.assertEqual(tuple(cand.shape), (3, F.k, F.sel_top_k))
        self.assertEqual(tuple(q.shape), (3, F.k, F.sel_top_k))
        self.assertEqual(drafts[1].tolist(), greedy[1].tolist(), "a row at temperature 0 walks greedily")
        for r in range(3):
            for s in range(F.k):
                at = (cand[r, s] == drafts[r, s]).nonzero()
                self.assertTrue(len(at), "the pick is one of the candidates it was drawn from")
                self.assertGreater(float(q[r, s][at[0]]), 0.0)
                self.assertAlmostEqual(float(q[r, s].sum()), 1.0, places=4)
        # a greedy row is a point mass: all of its mass sits on the candidate that is the pick
        self.assertTrue(torch.equal((q[1] > 0).sum(-1), torch.ones(F.k, dtype=torch.int64)), "one candidate carries it")
        self.assertTrue(torch.equal(cand[1].gather(1, (q[1] > 0).to(torch.int64).argmax(-1, keepdim=True)).squeeze(1),
                                    greedy[1]), "and it is the greedy pick")

    def test_native_rows_do_not_read_retired_weights_or_use_a_scratch_ring_tail(self):
        """Exercise the merged batched interface with packed readers and no BF16 sources.

        CPU attention/write oracles replace only the existing CUDA ring kernels;
        the batched projections, convolution and slot/position plumbing are real.
        """
        import copy
        import sys
        from types import ModuleType
        d, field, dev = self.make_full_drafter(seed=19)
        reference = copy.deepcopy(d)
        F = d.F
        linear = torch.nn.functional.linear
        d.context_kv = torch.cat([d.p[f"layers.{L}.self_attn.{s}_proj.weight"]
                                  for L in range(F.layers) for s in ("k", "v")])
        d.context_norm = torch.stack([d.p[f"layers.{L}.self_attn.k_norm.weight"] for L in range(F.layers)])
        packed = {"fc.weight": d.p["fc.weight"]}
        for L in range(F.layers):
            q = f"layers.{L}."
            packed[q+"self_attn.qkv"] = torch.cat([d.p[q+f"self_attn.{s}_proj.weight"] for s in ("q", "k", "v")])
            packed[q+"mlp.gate_up"] = torch.cat([d.p[q+f"mlp.{s}_proj.weight"] for s in ("gate", "up")])
            for key in ("self_attn.o_proj.weight", "mlp.down_proj.weight",
                        "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight"):
                packed[q+key] = d.p[q+key]
        d.dense = {name: (lambda x, w=w.clone(): linear(x, w)) for name, w in packed.items()}
        for name in list(d.p):
            if name in packed or name.endswith(("_proj.weight", "kernel_projection.weight")):
                d.p[name] = None
        d.fast_attention = True
        field = field[:, :, :, :F.window].contiguous()  # native ring has no SDPA scratch tail
        slots = torch.tensor([2, 1], device=dev)
        ctx = torch.tensor([13, 3], device=dev)
        anchors = torch.tensor([7, 9], device=dev)
        t = F.k + 1
        ids = torch.cat([anchors[:, None], torch.full((2, F.k), F.mask_id, device=dev)], 1).reshape(-1)
        pos = (ctx[:, None] + torch.arange(t, device=dev)).reshape(-1)
        expected = torch.cat([reference.block(ids[r*t:(r+1)*t], pos[r*t:(r+1)*t], field[int(slots[r])], int(ctx[r]))
                              for r in range(2)])
        kernels = ModuleType("engine.kernels.draft_attention")
        def attention(q, k, v, rings, position, *, slot, layer):
            ring = rings[int(slot[0]), layer]
            absolute = position + torch.arange(-F.window, 0, device=dev)
            keys = torch.cat([ring[0, absolute % F.window], k]).repeat_interleave(F.heads//F.kv_heads, 1)
            vals = torch.cat([ring[1, absolute % F.window], v]).repeat_interleave(F.heads//F.kv_heads, 1)
            scores = torch.einsum("bhd,nhd->bhn", q.float(), keys.float()) * F.head_dim**-.5
            valid = torch.cat([absolute >= 0, torch.ones(len(q), device=dev, dtype=torch.bool)])
            scores.masked_fill_(~valid[None, None], -float("inf"))
            return torch.einsum("bhn,nhd->bhd", scores.softmax(-1), vals.float()).to(q.dtype)
        def write(rings, slot, layer, positions, k, v, *, valid):
            count = int(valid)
            rows = positions[:count] % F.window
            rings[int(slot[0]), layer, 0, rows] = k[:count]
            rings[int(slot[0]), layer, 1, rows] = v[:count]
        def write_rows(rings, slots, layer, positions, k, v, *, valid):
            """The batched form the fast path uses now: one call for every row of the step. The stub keeps the
            per-row one beside it so the two are still being asked to agree."""
            for r in range(len(slots)):
                write(rings, slots[r:r + 1], layer, positions[r], k[r], v[r], valid=valid[r])
        def attend(q, k, v, rings, positions, *, slot, layer):
            """The batched form the fast path uses now: the rows are a grid dimension, and the stub keeps the
            per-row one beside it so the two are still being asked to agree."""
            return torch.stack([attention(q[r], k[r], v[r], rings, positions[r], slot=slot[r:r+1], layer=layer)
                                for r in range(len(slot))])
        kernels.draft_attention, kernels.write_draft_kv = attention, write
        kernels.write_draft_kv_rows, kernels.attend_rows = write_rows, attend
        observe_kernel = ModuleType("engine.kernels.draft_observe")
        def write_context(rings, slots, positions, context, weights, valid, eps, theta):
            from engine.kernels.norm_rope import norm_rope
            n, t, layers, _, heads, dim = context.shape
            for layer in range(layers):
                key = norm_rope(context[:, :, layer, 0].reshape(n*t, heads, dim), weights[layer], eps,
                                positions.reshape(-1), theta).reshape(n, t, heads, dim)
                write_rows(rings, slots, layer, positions, key, context[:, :, layer, 1], valid=valid)
        observe_kernel.write_context = write_context
        prior_observe = sys.modules.get(observe_kernel.__name__)
        sys.modules[observe_kernel.__name__] = observe_kernel
        prior = sys.modules.get(kernels.__name__)
        sys.modules[kernels.__name__] = kernels
        try:
            actual = d.block_rows(ids, pos, slots, ctx, field, 2, t)
            torch.testing.assert_close(actual.float(), expected.float(), atol=.06, rtol=.06)
            aux = torch.randn(2*t, F.hidden, device=dev).bfloat16()
            valid = torch.tensor([3, 0], device=dev)
            expect_ring = field.clone()
            for r in range(2):
                reference.observe_masked(expect_ring[int(slots[r])], pos.view(2,t)[r], aux[r*t:(r+1)*t], valid[r])
            d.observe_rows(field, slots, pos.view(2,t), aux, valid)
            torch.testing.assert_close(field.float(), expect_ring.float(), atol=.02, rtol=.02)
            self.assertTrue(torch.equal(field[1], expect_ring[1]))
        finally:
            if prior_observe is None:
                sys.modules.pop(observe_kernel.__name__, None)
            else:
                sys.modules[observe_kernel.__name__] = prior_observe
            if prior is None:
                sys.modules.pop(kernels.__name__, None)
            else:
                sys.modules[kernels.__name__] = prior

    def test_the_masks_of_the_batched_paths_reach_the_packs(self):
        """A calibrating pack sums only the rows the drafter says are real: `observe_rows` passes the kept positions,
        `propose_rows` the alive rows, `observe_masked` the valid count; a one-row path passes nothing (every row real)."""
        d, field, dev = self.make_full_drafter(seed=8)
        F, t = d.F, d.F.k + 1
        linear = torch.nn.functional.linear
        seen = {}

        class Fake:                                                           # a calibrating pack recording the masks it was handed
            def __init__(self, name, w):
                self.name, self.w, self.cols = name, w, w.shape[1]
                self.observer = object()                                      # what marks a pack as calibrating

            def __call__(self, x, rows_ok=None):
                seen.setdefault(self.name, []).append(None if rows_ok is None else rows_ok.clone())
                return linear(x, self.w)
        packed = {"fc.weight": d.p["fc.weight"]}
        for L in range(F.layers):
            q = f"layers.{L}."
            packed[q + "self_attn.qkv"] = torch.cat([d.p[q + f"self_attn.{s}_proj.weight"] for s in ("q", "k", "v")])
            packed[q + "mlp.gate_up"] = torch.cat([d.p[q + f"mlp.{s}_proj.weight"] for s in ("gate", "up")])
            for key in ("self_attn.o_proj.weight", "mlp.down_proj.weight", "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight"):
                packed[q + key] = d.p[q + key]
        d.dense = {name: Fake(name, w) for name, w in packed.items()}
        d.context_kv = torch.cat([d.p[f"layers.{L}.self_attn.{s}_proj.weight"] for L in range(F.layers) for s in ("k", "v")])
        slots = torch.tensor([1, 2, 3], device=dev); positions = torch.tensor([[5, 6, 7, 8], [13, 14, 15, 16], [2, 3, 4, 5]], device=dev)
        valid = torch.tensor([3, 0, 4], device=dev)
        aux = (torch.randn(3 * t, F.hidden * len(F.target_layers), device=dev) * 0.3).bfloat16()
        d.observe_rows(field, slots, positions, aux, valid)
        self.assertEqual(seen["fc.weight"][-1].tolist(), [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        d.observe_masked(field[1], positions[0], aux[:t], valid[0])
        self.assertEqual(seen["fc.weight"][-1].tolist(), [1, 1, 1, 0])
        d.observe(field[1], positions[0], aux[:t])
        self.assertIsNone(seen["fc.weight"][-1])
        anchors = torch.tensor([7, 3, 9], device=dev); ctx = torch.tensor([5, 12, 3], device=dev)
        alive = torch.tensor([True, False, True], device=dev)
        d.propose_rows(field, slots, anchors, ctx, alive=alive)
        for name in ("layers.0.attention_conv.kernel_projection.weight", "layers.0.mlp_conv.kernel_projection.weight",
                     "layers.0.mlp.down_proj.weight"):                        # the stock attention path here; the native one adds qkv, o, gate_up
            self.assertEqual(seen[name][-1].tolist(), [1] * t + [0] * t + [1] * t, name)
        d.propose_rows(field, slots, anchors, ctx)
        self.assertIsNone(seen["layers.0.mlp.down_proj.weight"][-1])
        d.propose_tensor(anchors[:1], 5, field[1])
        self.assertIsNone(seen["layers.0.mlp.down_proj.weight"][-1])

    def test_dense_shapes_name_every_prepared_weight_at_this_tp(self):
        from engine.profiles.glm53.drafter import dense_shapes, store_name
        F = self.make_drafter().F
        shapes = dense_shapes(F, 1)
        self.assertEqual(len(shapes), 1 + 6 * F.layers)
        self.assertEqual(shapes["fc.weight"], (F.hidden, F.hidden * len(F.target_layers)))
        self.assertEqual(shapes["layers.0.self_attn.qkv"], ((F.heads + 2 * F.kv_heads) * F.head_dim, F.hidden))
        self.assertEqual(store_name("layers.0.self_attn.qkv"), "DFlash2Qwen3ForCausalLM/model.layers.0.self_attn.qkv_proj")
        self.assertEqual(store_name("fc.weight"), "DFlash2Qwen3ForCausalLM/model.fc")
        half = dense_shapes(F, 2)
        self.assertEqual(half["layers.0.self_attn.o_proj.weight"], (F.hidden, F.heads // 2 * F.head_dim))

    def test_the_host_walk_crosses_once_not_twice_a_position(self):
        source = (Path(__file__).resolve().parents[1] / "engine/profiles/glm53/drafter.py").read_text()
        body = source[source.index("    def propose_sampled(self"):source.index("    def propose_sampled_tensor(")]
        self.assertNotIn(".item()", body)
        self.assertIn("propose_sampled_tensor(", body)


@unittest.skipUnless(torch.cuda.is_available(), "observe's write phase is a device kernel")
class ObserveOverlapTests(unittest.TestCase):
    """observe splits into a compute phase and a write phase so the compute can overlap the target's tail
    (45차 §96). The split has to be free -- byte-for-byte the single call -- and the compute has to be
    independent of the accepted count, or starting it before the commit would be wrong.
    """
    def fast_drafter(self):
        from engine.profiles.glm53.drafter import Drafter, DrafterFacts
        F = DrafterFacts(layers=3, hidden=256, heads=8, kv_heads=2, head_dim=128, inter=256, rms_eps=1e-5,
                         rope_theta=10000., window=64, block=8, mask_id=20, conv_taps=2, conv_group=16,
                         sel_rank=4, sel_top_k=3, target_layers=(1, 2, 3), k=5)
        d = Drafter(F, SimpleNamespace(), 64)
        d.fast_attention = True
        d.local_kv_heads = F.kv_heads
        gen = torch.Generator(device="cuda").manual_seed(3)
        rand = lambda *shape: torch.randn(*shape, device="cuda", generator=gen).bfloat16()
        d.p = {"fc.weight": rand(F.hidden, F.hidden * len(F.target_layers)),
               "hidden_norm.weight": torch.ones(F.hidden, device="cuda", dtype=torch.bfloat16)}
        for L in range(F.layers):
            d.p[f"layers.{L}.self_attn.k_norm.weight"] = torch.ones(F.head_dim, device="cuda", dtype=torch.bfloat16)
        d.context_kv = rand(F.layers * 2 * F.kv_heads * F.head_dim, F.hidden)
        d.linear = lambda x, name, mask=None: torch.nn.functional.linear(x, d.p[name])   # no pack: reference lane
        return d

    def inputs(self, d, accepted):
        from engine.profiles.glm53.drafter import ring_cells
        F = d.F
        n, t = 1, F.k + 1
        field = torch.zeros(2, F.layers, 2, ring_cells(F), F.kv_heads, F.head_dim, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([1], device="cuda")
        positions = (torch.arange(n * t, device="cuda") + 100).view(n, t)
        gen = torch.Generator(device="cuda").manual_seed(7)
        aux = torch.randn(n * t, F.hidden * len(F.target_layers), device="cuda", generator=gen).bfloat16()
        return field, slots, positions, aux, torch.tensor([accepted], device="cuda")

    def test_the_split_is_byte_for_byte_the_single_pass(self):
        from engine.profiles.glm53.drafter import norm, norm_rope
        from engine.kernels.draft_attention import write_draft_kv_rows
        d = self.fast_drafter()
        F = d.F
        n, t = 1, F.k + 1
        field, slots, positions, aux, valid = self.inputs(d, accepted=4)

        got = field.clone()
        d.observe_rows(got, slots, positions, aux, valid)

        want = torch.zeros_like(field)
        c = norm(torch.nn.functional.linear(aux, d.p["fc.weight"]), d.p["hidden_norm.weight"], F.rms_eps)
        context = torch.nn.functional.linear(c, d.context_kv).reshape(n, t, F.layers, 2, F.kv_heads, F.head_dim)
        for L in range(F.layers):
            k = norm_rope(context[:, :, L, 0].reshape(n * t, F.kv_heads, F.head_dim),
                          d.p[f"layers.{L}.self_attn.k_norm.weight"], F.rms_eps,
                          positions.reshape(-1), F.rope_theta).reshape(n, t, F.kv_heads, F.head_dim)
            write_draft_kv_rows(want, slots, L, positions, k, context[:, :, L, 1], valid=valid)
        self.assertTrue(torch.equal(got, want))

    def test_the_compute_phase_does_not_depend_on_the_accepted_count(self):
        """The overlap starts observe_kv before the commit decides the count. That is only correct because the
        K/V it produces are the same whatever the count turns out to be -- the count masks the write, nothing
        more. (On a calibrating boot the mask feeds the calibration sum, so the overlap is a serving-boot
        thing; the K/V values here are identical regardless.)"""
        d = self.fast_drafter()
        _, _, positions, aux, _ = self.inputs(d, accepted=0)
        full = d.observe_kv(positions, aux, torch.tensor([d.F.k + 1], device="cuda"))
        part = d.observe_kv(positions, aux, torch.tensor([2], device="cuda"))
        self.assertTrue(all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(full, part)))


if __name__ == "__main__":
    unittest.main()
