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
        self.assertIn("mtp=model.drafter is not None", build[warm:])


if __name__ == "__main__":
    unittest.main()
