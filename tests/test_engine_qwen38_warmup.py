"""engine/profiles/qwen38/warmup: before the door, the largest prefill chunk is held to the memory ceiling and every
prefill kernel family runs once -- each width at context 0 and a short step on every context bucket's edge -- through
the target and the MTP head; every rank votes each pass; the slot, blocks and caches are given back whatever happens."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class FakeCaches:
    def __init__(self, blocks=1400, block=768, snapshots=8):
        self.device = "cpu"
        self.snapshots = snapshots
        self.block_table = torch.empty(4, blocks) if torch is not None else None
        self.pool = NS(num_blocks=blocks, rows_in_use=0, reserved={}, released=[])
        self.pool.reserve = lambda seq, tokens: self.pool.reserved.__setitem__(seq, tokens)
        self.pool.release = lambda seq: self.pool.released.append(seq)
        self.slots = NS(owner=[-1] * 5, given=[])
        self.slots.take = lambda seq: 1
        self.slots.give = lambda slot: self.slots.given.append(slot)
        self.prepared, self.slot_resets, self.resets = [], 0, 0

    def prepare(self, step):
        self.prepared.append(step)

    def reset_slot(self, slot):
        self.slot_resets += 1

    def reset(self):
        self.resets += 1


class FakeNet:
    def __init__(self, fail_at=None, nan_at=None, peer_bad=False):
        self.F = NS(block=768)
        self.steps, self.mtp_steps, self.votes = [], [], []
        self.fail_at, self.nan_at, self.peer_bad = fail_at, nan_at, peer_bad
        net = self

        class Comm:
            def all_reduce_max(self, t):
                net.votes.append(int(t.item()))
                return torch.ones_like(t) if net.peer_bad else t
        self.comm = Comm()

    def takes_mark(self, offset):
        return offset > 0 and offset % 64 == 0

    def forward(self, step, caches, *, streams=False):
        self.steps.append(step)
        n = step.ids.numel()
        if self.fail_at == len(self.steps):
            raise RuntimeError("an illegal memory access")
        value = float("nan") if self.nan_at == len(self.steps) else 0.0
        return torch.full((n, 4), value), torch.zeros(n, 8)

    def head(self, h):
        return h

    def mtp_forward(self, step, given, caches, *, last_hidden_only=True):
        self.mtp_steps.append((step, given.shape[0]))
        return torch.zeros(1, 4), None


@unittest.skipUnless(torch is not None, "requires torch")
class PlanTests(unittest.TestCase):
    def test_the_memory_pass_leads_and_every_bucket_edge_is_reached(self):
        from engine.profiles.qwen38 import warmup
        from engine.profiles.qwen38.net import bucket_blocks
        passes = warmup.plan(NS(block=768), chunk=32256, capacity=262144, top=1400)
        self.assertEqual(passes[0], ("memory", 32256, 0))
        self.assertEqual([p for p in passes if p[2] == 0][1:], [("kernels", w, 0) for w in warmup.WIDTHS])
        reached = {bucket_blocks(768, -(-(context + tokens) // 768), 1400) for _, tokens, context in passes}
        self.assertEqual(reached, {6, 11, 22, 43, 86, 171, 342})
        self.assertTrue(all(context + tokens <= 262144 for _, tokens, context in passes))

    def test_the_head_passes_cover_every_eager_observation_at_every_rung(self):
        from engine.profiles.qwen38 import warmup
        passes = warmup.plan(NS(block=768), chunk=32256, capacity=262144, top=1400, head=4)
        heads = [p for p in passes if p[0] == "head"]
        edges = warmup.rungs(768, 262144, 1400)
        self.assertEqual(len(heads), 4 * (1 + len(edges)))
        self.assertEqual({t for _, t, _ in heads}, {1, 2, 3, 4})       # K+1 observed positions down to a chain step
        for t in (1, 2, 3, 4):
            self.assertEqual([c for _, n, c in heads if n == t], [0] + [end - t for end in edges])
        self.assertEqual(passes[:len(passes) - len(heads)], warmup.plan(NS(block=768), chunk=32256, capacity=262144,
                                                                        top=1400))   # after the prefill passes, which stay
        self.assertEqual(warmup.plan(NS(block=768), chunk=32256, capacity=262144, top=1400, head=0),
                         [p for p in passes if p[0] != "head"])

    def test_the_decode_sized_widths_are_back(self):
        from engine.profiles.qwen38 import warmup
        self.assertEqual(warmup.WIDTHS[:2], (1, 8))

    def test_a_small_pool_caps_the_passes(self):
        from engine.profiles.qwen38 import warmup
        passes = warmup.plan(NS(block=768), chunk=32256, capacity=10 * 768, top=10)
        self.assertEqual(passes[0], ("memory", 7680, 0))
        self.assertTrue(all(context + tokens <= 7680 for _, tokens, context in passes))


@unittest.skipUnless(torch is not None, "requires torch")
class WarmupTests(unittest.TestCase):
    def run_warmup(self, net, caches, memory=None, mtp=True):
        from engine.profiles.qwen38 import warmup
        return warmup.warmup(net, caches, memory=memory, chunk=32256, max_context=262144, mtp=mtp)

    def test_every_pass_runs_the_target_and_the_head_and_is_voted(self):
        from engine.profiles.qwen38 import warmup
        net, caches = FakeNet(), FakeCaches()
        rows = []
        memory = NS(checkpoint=lambda name, release_cache=False: rows.append((name, release_cache)))
        paid = self.run_warmup(net, caches, memory)
        passes = warmup.plan(net.F, chunk=32256, capacity=262144, top=1400)
        self.assertEqual([(s.ids.numel(), s.segments[0].ctx) for s in net.steps], [(n, c) for _, n, c in passes])
        self.assertEqual([(s.ids.numel(), n) for s, n in net.mtp_steps], [(n, n) for _, n, _ in passes])
        self.assertEqual(net.votes, [0] * len(passes))
        self.assertEqual([name for name, _ in rows], [f"prefill/{n}/{c}/{k}" for k, n, c in passes])
        self.assertTrue(all(release for _, release in rows))
        self.assertEqual(list(paid), [f"{k}/{n}/{c}" for k, n, c in passes])
        self.assertEqual(caches.pool.reserved, {0: 262144})

    def test_only_the_memory_pass_carries_prefix_marks(self):
        net, caches = FakeNet(), FakeCaches(snapshots=8)
        self.run_warmup(net, caches)
        first, rest = net.steps[0], net.steps[1:]
        self.assertEqual(first.marks, tuple((768 * (i + 1), i) for i in range(8)))
        self.assertTrue(all(not s.marks for s in rest))

    def test_without_a_head_the_target_alone_runs(self):
        net, caches = FakeNet(), FakeCaches()
        self.run_warmup(net, caches, mtp=False)
        self.assertEqual(net.mtp_steps, [])

    def test_a_rank_that_raises_votes_first_and_everything_is_given_back(self):
        net, caches = FakeNet(fail_at=2), FakeCaches()
        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            self.run_warmup(net, caches)
        self.assertEqual(net.votes, [0, 1])            # the failing pass still reached the collective, with its flag
        self.assertEqual((caches.pool.released, caches.slots.given, caches.resets), ([0], [1], 1))

    def test_non_finite_output_or_a_peers_flag_stops_every_rank(self):
        net, caches = FakeNet(nan_at=1), FakeCaches()
        with self.assertRaises(FloatingPointError):
            self.run_warmup(net, caches)
        net, caches = FakeNet(peer_bad=True), FakeCaches()
        with self.assertRaisesRegex(FloatingPointError, "or a peer"):
            self.run_warmup(net, caches)
        self.assertEqual(caches.resets, 1)

    def test_a_head_pass_runs_the_head_alone_and_picks_as_the_drafter_does(self):
        from engine.profiles.qwen38 import warmup
        net, caches = FakeNet(), FakeCaches()
        net.F = NS(block=768, hc=4, hidden=2)
        picked = []
        net.draft_tokens = lambda h: picked.append(h.shape[0]) or torch.zeros(h.shape[0], dtype=torch.int64)
        paid = self.run_warmup_head(net, caches, head=4)
        passes = warmup.plan(net.F, chunk=32256, capacity=262144, top=1400, head=4)
        prefill = [p for p in passes if p[0] != "head"]
        heads = [p for p in passes if p[0] == "head"]
        self.assertEqual(len(net.steps), len(prefill))                  # the target runs the prefill passes only
        head_steps = net.mtp_steps[len(prefill):]
        self.assertEqual([(s.ids.numel(), s.segments[0].ctx, given) for s, given in head_steps],
                         [(t, c, t) for _, t, c in heads])              # given: the zeros it is handed, one row a token
        self.assertEqual(len(picked), len(heads))
        self.assertEqual(net.votes, [0] * len(passes))
        self.assertIn("head/4/0", paid)

    def test_without_a_drafter_there_are_no_head_passes(self):
        net, caches = FakeNet(), FakeCaches()
        net.F = NS(block=768, hc=4, hidden=2)
        self.run_warmup_head(net, caches, head=4, mtp=False)
        self.assertEqual(net.mtp_steps, [])

    def test_a_non_finite_head_stops_the_boot(self):
        net, caches = FakeNet(), FakeCaches()
        net.F = NS(block=768, hc=4, hidden=2)
        net.draft_tokens = lambda h: None
        forward = net.mtp_forward
        net.mtp_forward = lambda step, given, caches, last_hidden_only=True: (
            (torch.full((1, 4), float("nan")), None) if step.ids.numel() == 3 else forward(step, given, caches))
        with self.assertRaisesRegex(FloatingPointError, "head/3/0"):
            self.run_warmup_head(net, caches, head=4)

    def run_warmup_head(self, net, caches, head, mtp=True):
        from engine.profiles.qwen38 import warmup
        return warmup.warmup(net, caches, memory=None, chunk=32256, max_context=262144, mtp=mtp, head=head)

    def test_a_live_request_refuses_the_warmup(self):
        caches = FakeCaches()
        caches.pool.rows_in_use = 1
        with self.assertRaises(ValueError):
            self.run_warmup(FakeNet(), caches)


class BootOrderTests(unittest.TestCase):
    def test_the_boot_warms_after_the_prelude_and_before_the_capture(self):
        source = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        build = source[source.index("def build("):source.index("def write_dumps(")]
        warm = build.index('with recorder.phase("warm prefill")')
        self.assertLess(build.index("prelude.take()"), warm)
        self.assertLess(warm, build.index('with recorder.phase("capture decode")'))
        self.assertIn("mtp=model.drafter is not None, head=k + 1", build[warm:])

    def test_the_boot_warms_the_eager_moe_first_and_everything_before_the_capture(self):
        # before the prefill passes: their 1- and 8-token widths route a few pairs to a rank, and a first call below
        # the ceiling would build the workspace -- and kernels -- at its own smaller capacity
        source = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        build = source[source.index("def build("):source.index("def write_dumps(")]
        eager = build.index('with recorder.phase("warm eager moe")')
        self.assertLess(build.index("prelude.take()"), eager)
        self.assertLess(eager, build.index('with recorder.phase("warm prefill")'))
        self.assertLess(build.index('with recorder.phase("warm prefill")'), build.index('with recorder.phase("capture decode")'))
        self.assertIn("eager_moe(net)", build[eager:])


class EagerNet:
    """The fields `eager_moe` reads: F, first_expert, _experts, p, comm -- the experts a recorder of each launch's
    pairs, as lanes.moe's compact path counts them (the routes that fall in this rank's range)."""

    def __init__(self, *, rank=1, local=128, experts=512, topk=10, fail_at=None, peer_bad=False, mtp_only=False,
                 mtp_experts="nvfp4"):
        self.F = NS(experts=experts, topk_experts=topk, hidden=16)
        self.first_expert, self.local, self.mtp_experts = rank * local, local, mtp_experts
        self.launches, self.votes, self.fail_at, self.peer_bad = [], [], fail_at, peer_bad
        prefixes = ["mtp.L0."] if mtp_only else ["L0.", "L1.", "mtp.L0."]
        self._experts = {prefix: (lambda prefix: lambda x, ids, w, compact: self.launch(prefix, x, ids, w, compact))(prefix)
                         for prefix in prefixes}
        self.p = {prefix + "moe.w13": torch.empty(local, 2, 1) for prefix in prefixes}
        net = self

        class Comm:
            def all_reduce_max(self, t):
                net.votes.append(int(t.item()))
                return torch.ones_like(t) if net.peer_bad else t
        self.comm = Comm()

    def launch(self, prefix, x, ids, weights, compact):
        if self.fail_at == len(self.launches) + 1:
            raise RuntimeError("a CuTe DSL compile failed")
        shifted = ids.to(torch.int64) - self.first_expert
        local = (shifted >= 0) & (shifted < self.local)
        self.launches.append(dict(prefix=prefix, rows=x.shape[0], pairs=int(local.sum()), compact=compact,
                                  local_in_first_route=bool(local[:, 0].all()), ids=ids, weights=weights,
                                  dtypes=(x.dtype, ids.dtype, weights.dtype)))
        return torch.zeros_like(x)


@unittest.skipUnless(torch is not None, "requires torch")
class EagerMoeTests(unittest.TestCase):
    def test_the_ceiling_first_then_every_count_below_it(self):
        from engine.profiles.qwen38.warmup import EAGER_PAIRS, eager_counts
        self.assertEqual(eager_counts(), [8, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(sorted(eager_counts()), list(range(1, EAGER_PAIRS + 1)))

    def test_it_is_the_dispatchers_micro_ceiling(self):
        source = (ROOT / "engine/kernels/b12x/moe_dispatch.py").read_text(encoding="utf-8")
        from engine.profiles.qwen38.warmup import EAGER_PAIRS
        self.assertIn(f"_MICRO_MAX_TOKENS = {EAGER_PAIRS}", source.splitlines())
        # above it a one-route launch over every local expert is the dynamic kernel's, free of the count
        self.assertIn("num_experts == num_local_experts > 1 and num_tokens > _MICRO_MAX_TOKENS", source)

    def test_each_launch_keeps_exactly_its_count_of_this_ranks_pairs(self):
        from engine.profiles.qwen38.warmup import eager_moe
        for rank in range(4):
            net = EagerNet(rank=rank)
            paid = eager_moe(net)
            with self.subTest(rank=rank):
                self.assertEqual([l["pairs"] for l in net.launches], [8, 1, 2, 3, 4, 5, 6, 7])
                self.assertEqual([l["rows"] for l in net.launches], [8, 1, 2, 3, 4, 5, 6, 7])
                self.assertTrue(all(l["compact"] and l["local_in_first_route"] for l in net.launches))
                self.assertEqual({l["prefix"] for l in net.launches}, {"L0."})        # a target layer's experts
                self.assertEqual(net.launches[0]["dtypes"], (torch.bfloat16, torch.int32, torch.float32))
                ids = net.launches[0]["ids"]
                self.assertTrue(all(len(set(row.tolist())) == row.numel() for row in ids))   # distinct routes a row
                self.assertEqual(len(set(ids[:, 0].tolist())), 8)                               # distinct experts
                self.assertTrue(bool(((ids >= 0) & (ids < 512)).all()))
                self.assertEqual(list(paid), [f"eager/{m}" for m in (8, 1, 2, 3, 4, 5, 6, 7)])
                self.assertEqual(net.votes, [0])

    def test_the_mtp_head_when_there_is_no_target_layer_and_nothing_on_fp8_experts(self):
        from engine.profiles.qwen38.warmup import eager_moe
        net = EagerNet(mtp_only=True)
        eager_moe(net)
        self.assertEqual({l["prefix"] for l in net.launches}, {"mtp.L0."})
        net = EagerNet(mtp_only=True, mtp_experts="fp8")
        self.assertEqual(eager_moe(net), {})
        self.assertEqual((net.launches, net.votes), ([], []))

    def test_a_rank_that_fails_or_a_peer_that_failed_stops_the_boot(self):
        from engine.profiles.qwen38.warmup import eager_moe
        net = EagerNet(fail_at=3)
        with self.assertRaisesRegex(RuntimeError, "CuTe DSL"):
            eager_moe(net)
        self.assertEqual(net.votes, [1])                   # it voted before raising: its peers stop at the same vote
        net = EagerNet(peer_bad=True)
        with self.assertRaisesRegex(RuntimeError, "on a peer"):
            eager_moe(net)


if __name__ == "__main__":
    unittest.main()
