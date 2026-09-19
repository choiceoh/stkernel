"""The MTP head's draft chain (fleet --spec-k K, K > 1): decode_graphs.draft_chain runs the head over the padded
observation, then K-1 single-position steps, each at the position after the row's last observed one, taking the row's
last pick as its token and the head's own streams as its state; DraftGraphs sizes a row by its observation and the
chain; adapter.ServedMTP replays a waiting row when its reservation holds the replay's positions, runs the head eagerly
over the observed positions alone when it does not (a park), and proposes K picks either way. K=1 keeps its old shape:
one launch, the verify width, the graph at a park.
"""
import importlib.util
import unittest
from types import SimpleNamespace

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class FakeNet:
    """mtp_forward records (ids, positions, given rows) per step and answers hidden rows that carry each row's token
    and position; head_tokens maps a token t to 10*t + 1, so a chain's picks are distinct and checkable."""

    def __init__(self):
        self.steps = []

    def mtp_forward(self, step, given, caches, *, last_hidden_only=True):
        ids = step.ids
        if getattr(step, "captured", False):
            positions = (step.contexts[:, None] + torch.arange(step.tokens)).reshape(-1)
        else:
            positions = torch.cat([torch.arange(s.ctx, s.ctx + s.length) for s in step.segments])
        self.steps.append((ids.tolist(), positions.tolist(), int(given.shape[0]),
                           step.tokens if getattr(step, "captured", False) else None))
        hidden = torch.stack([ids.to(torch.float32), positions.to(torch.float32)], dim=1)
        streams = hidden * 2
        if last_hidden_only:
            last = torch.tensor([s.start + s.length - 1 for s in step.segments])
            return hidden.index_select(0, last), streams.index_select(0, last)
        return hidden, streams

    def head_tokens(self, h, decodable=None):
        return (h[:, 0] * 10 + 1).to(torch.int64)

    def draft_tokens(self, h):                   # the drafter's picks (net.draft_tokens: the head's, or its index's)
        return self.head_tokens(h)


def fake_caches(reserved, owners):
    prepared = []
    caches = SimpleNamespace(pool=SimpleNamespace(tokens=list(reserved)), slots=SimpleNamespace(owner=list(owners)),
                             prepare=lambda step: prepared.append([(s.seq, s.ctx, s.length) for s in step.segments]))
    caches.prepared = prepared
    return caches


class FakeGraphs:
    """DraftGraphs' surface ServedMTP reads: tokens (the verify width), extent, run."""

    def __init__(self, k: int):
        from engine.profiles.qwen38.decode_graphs import DraftGraphs
        self.k, self.tokens, self.ran = k, k + 1, []
        self.extent = lambda observed: DraftGraphs.extent(self, observed)

    def run(self, rows):
        self.ran.append([(seq, slot, ctx, list(ids), int(streams.shape[0])) for seq, slot, ctx, ids, streams in rows])
        return [[100 + 10 * i + j for j in range(self.k)] for i in range(len(rows))]


@unittest.skipUnless(torch is not None, "requires torch")
class DraftChainTests(unittest.TestCase):
    def test_the_chain_feeds_each_step_the_last_pick_at_the_next_position(self):
        from engine.profiles.qwen38.decode_graphs import draft_chain
        from engine.profiles.qwen38.net import DeviceStep
        net, k, t = FakeNet(), 3, 4
        # two rows: one observed all four positions, one two (padded with its last token)
        ids = torch.tensor([11, 12, 13, 14, 21, 22, 22, 22])
        step = DeviceStep(ids, torch.tensor([10, 20]), torch.tensor([1, 2]), torch.tensor([0, 1]), t, 4)
        last, counts = torch.tensor([3, 5]), torch.tensor([4, 2])
        picks = draft_chain(net, None, step, torch.zeros(8, 2), last, counts, k)
        self.assertEqual(picks.shape, (2, k))
        self.assertEqual(picks.tolist(), [[141, 1411, 14111], [221, 2211, 22111]])
        self.assertEqual(len(net.steps), k)
        self.assertEqual(net.steps[0][1], [10, 11, 12, 13, 20, 21, 22, 23])
        # the chain: the row's pick, one position a row after its observed ones, the head's streams at those rows
        self.assertEqual(net.steps[1], ([141, 221], [14, 22], 2, 1))
        self.assertEqual(net.steps[2], ([1411, 2211], [15, 23], 2, 1))

    def test_k1_is_one_launch(self):
        from engine.profiles.qwen38.decode_graphs import draft_chain
        from engine.profiles.qwen38.net import DeviceStep
        net = FakeNet()
        step = DeviceStep(torch.tensor([5, 6]), torch.tensor([3]), torch.tensor([1]), torch.tensor([0]), 2, 4)
        picks = draft_chain(net, None, step, torch.zeros(2, 2), torch.tensor([1]), torch.tensor([2]), 1)
        self.assertEqual(picks.tolist(), [[61]])
        self.assertEqual(len(net.steps), 1)

    def test_extent_is_the_verify_width_or_the_chain_past_the_observation(self):
        from engine.profiles.qwen38.decode_graphs import DraftGraphs
        g3 = SimpleNamespace(tokens=4, k=3)
        self.assertEqual([DraftGraphs.extent(g3, m) for m in (1, 2, 3, 4)], [4, 4, 5, 6])
        g1 = SimpleNamespace(tokens=2, k=1)
        self.assertEqual([DraftGraphs.extent(g1, m) for m in (1, 2)], [2, 2])


@unittest.skipUnless(torch is not None, "requires torch")
class RowsPublishTests(unittest.TestCase):
    def rows(self, tokens, reach=None, reserved=(0, 16)):
        from engine.profiles.qwen38.decode_graphs import _Rows
        net = SimpleNamespace(F=SimpleNamespace(block=768, spec_k=3), comm=SimpleNamespace(graph_capture_safe=True))
        caches = fake_caches(reserved, [-1, -1, -1])
        caches.block_table = torch.zeros(2, 8, dtype=torch.int32)
        return _Rows(net, caches, 2, tokens, 4096, reach), caches

    def test_publish_takes_each_rows_width_within_its_reservation_and_the_reach(self):
        rows, caches = self.rows(4, reach=6)
        rows.publish([(1, 1, 10, 6), (1, 1, 12, 4)])
        self.assertEqual(caches.prepared, [[(1, 10, 6), (1, 12, 4)]])
        with self.assertRaisesRegex(ValueError, "passes its reservation"):
            rows.publish([(1, 1, 11, 6)])
        with self.assertRaisesRegex(ValueError, "1..6 positions"):
            rows.publish([(1, 1, 0, 7)])

    def test_reach_defaults_to_the_width_and_sizes_the_buckets(self):
        from engine.profiles.qwen38.decode_graphs import bucket_ladder
        rows, _ = self.rows(4)
        self.assertEqual(rows.reach, 4)
        self.assertEqual(rows.buckets, bucket_ladder(768, 8, 4096, 4))
        with self.assertRaises(ValueError):
            self.rows(4, reach=3)


@unittest.skipUnless(torch is not None, "requires torch")
class ServedMTPTests(unittest.TestCase):
    def drafter(self, k, reserved, owners=(-1, 7, -1)):
        from engine.profiles.qwen38.adapter import ServedMTP
        net, caches = FakeNet(), fake_caches(reserved, owners)
        store = SimpleNamespace(slot_of={7: 1})
        mtp = ServedMTP(net, caches, store, k)
        mtp.graphs = FakeGraphs(k)
        return mtp, net, caches

    def test_a_waiting_row_replays_with_its_chain_when_the_horizon_is_reserved(self):
        # after the verify step at 10 kept 4 tokens, the next step reserves the horizon 14 + 1 + K = 18
        mtp, net, _ = self.drafter(3, reserved=[0] * 7 + [18])
        mtp.observe(7, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        self.assertIn(7, mtp._waiting)
        self.assertEqual(mtp.propose([7]), [[100, 101, 102]])
        self.assertEqual(mtp.graphs.ran, [[(7, 1, 10, [1, 2, 3, 4], 4)]])
        self.assertEqual(net.steps, [])                                  # nothing eager

    def test_a_park_runs_the_head_eagerly_when_the_chain_would_pass_the_reservation(self):
        # parked after the verify step at 10 kept 4 tokens: the reservation is that step's 10 + K + 1 = 14, and the
        # replay would write the chain at 14 and 15 -- the head runs its four observed rows and nothing more
        mtp, net, _ = self.drafter(3, reserved=[0] * 7 + [14])
        mtp.observe(7, 10, [1, 2, 3, 4], torch.zeros(4, 2))
        mtp.forget(7)
        self.assertEqual(mtp.graphs.ran, [])
        self.assertEqual(net.steps, [([1, 2, 3, 4], [10, 11, 12, 13], 4, None)])
        self.assertNotIn(7, mtp._next)
        self.assertNotIn(7, mtp._waiting)

    def test_a_park_with_a_short_observation_still_replays(self):
        mtp, net, _ = self.drafter(3, reserved=[0] * 7 + [14])
        mtp.observe(7, 10, [1, 2], torch.zeros(2, 2))                    # extent 4: positions 10..13 are reserved
        mtp.forget(7)
        self.assertEqual(mtp.graphs.ran, [[(7, 1, 10, [1, 2], 2)]])
        self.assertEqual(net.steps, [])

    def test_a_released_row_is_dropped(self):
        mtp, net, _ = self.drafter(3, reserved=[0] * 7 + [18], owners=(-1, -1, -1))
        mtp.observe(7, 10, [1, 2], torch.zeros(2, 2))
        mtp.forget(7)
        self.assertEqual((mtp.graphs.ran, net.steps, mtp._waiting), ([], [], {}))

    def test_k1_parks_through_the_graph_as_before(self):
        mtp, net, _ = self.drafter(1, reserved=[0] * 7 + [12])
        mtp.observe(7, 10, [1, 2], torch.zeros(2, 2))
        mtp.forget(7)
        self.assertEqual(mtp.graphs.ran, [[(7, 1, 10, [1, 2], 2)]])
        self.assertEqual(net.steps, [])

    def test_a_prompt_observation_runs_eagerly_and_the_chain_follows_at_propose(self):
        mtp, net, _ = self.drafter(3, reserved=[0] * 7 + [40])
        mtp.observe(7, 0, [1, 2, 3, 4, 5, 6], torch.zeros(6, 2))         # 6 > K+1: the head at once
        self.assertEqual(net.steps, [([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5], 6, None)])
        self.assertEqual(mtp.propose([7]), [[61, 611, 6111]])
        # the chain: one position each, after the observed ones, from the last pick
        self.assertEqual(net.steps[1:], [([61], [6], 1, None), ([611], [7], 1, None)])
        self.assertEqual(mtp.graphs.ran, [])

    def test_a_second_observation_runs_the_waiting_row_first(self):
        mtp, net, _ = self.drafter(3, reserved=[0] * 7 + [12])
        mtp.observe(7, 0, [1, 2, 3, 4], torch.zeros(4, 2))               # a prompt's tail: waits
        mtp.observe(7, 4, [5, 6], torch.zeros(2, 2))                     # the reservation (12) holds 0 + 6: replay
        self.assertEqual(mtp.graphs.ran, [[(7, 1, 0, [1, 2, 3, 4], 4)]])
        self.assertEqual(mtp._waiting[7][1:3], (4, [5, 6]))
        self.assertEqual(mtp.propose([7]), [[100, 101, 102]])
        self.assertEqual(len(mtp.graphs.ran), 2)

    def test_the_capture_takes_any_k(self):
        from engine.profiles.qwen38.adapter import ServedMTP
        mtp = ServedMTP(FakeNet(), fake_caches([0], [-1]), SimpleNamespace(slot_of={}), 3)
        self.assertEqual(mtp.k, 3)
        with self.assertRaises(ValueError):
            ServedMTP(FakeNet(), fake_caches([0], [-1]), SimpleNamespace(slot_of={}), 0)


class RingCheckTests(unittest.TestCase):
    def test_the_fixed_rings_hold_a_verify_step_up_to_k4(self):
        from engine.profiles.qwen38.caches import check_rings
        for k in (1, 2, 3, 4):
            check_rings(SimpleNamespace(idx_ratio=4, ngram_size=3, spec_k=k))
        with self.assertRaisesRegex(ValueError, "raw index-key ring"):
            check_rings(SimpleNamespace(idx_ratio=4, ngram_size=3, spec_k=5))
        with self.assertRaisesRegex(ValueError, "PLE id ring"):
            check_rings(SimpleNamespace(idx_ratio=2, ngram_size=6, spec_k=3))


class LauncherTests(unittest.TestCase):
    def test_st_spec_k_reaches_the_fleet_command(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath("launchers", "start-st-qwen38.sh").read_text()
        self.assertIn('SPEC_ARG="--spec-k $ST_SPEC_K"', text)
        exec_line = next(l for l in text.splitlines() if "-m engine.profiles.qwen38.fleet " in l)
        self.assertLess(exec_line.index("$HC_ARG $SPEC_ARG"), exec_line.index("--port"))

    def test_st_mtp_precision_reaches_the_fleet_command(self):
        from pathlib import Path
        text = Path(__file__).resolve().parents[1].joinpath("launchers", "start-st-qwen38.sh").read_text()
        self.assertIn('bf16|fp8|w4) MTP_ARG="--mtp-precision $ST_MTP_PRECISION" ;;', text)
        exec_line = next(l for l in text.splitlines() if "-m engine.profiles.qwen38.fleet " in l)
        self.assertLess(exec_line.index("$MTP_ARG"), exec_line.index("--port"))
        fleet = Path(__file__).resolve().parents[1].joinpath("engine", "profiles", "qwen38", "fleet.py").read_text()
        self.assertIn('ap.add_argument("--mtp-precision", choices=("bf16", "fp8", "w4"), default="bf16",', fleet)
        self.assertIn("mtp_precision=a.mtp_precision,", fleet)
        self.assertIn("draft_index=draft_index(a.draft_index))", fleet)


if __name__ == "__main__":
    unittest.main()
