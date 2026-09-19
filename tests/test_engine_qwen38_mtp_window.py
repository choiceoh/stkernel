"""The MTP head attends a sink and a recent window of groups instead of its scored selection (Windowed-MTP, arXiv
2607.21535; fleet --mtp-window SINK,RECENT, the launcher's ST_MTP_WINDOW; off by default).

The target keeps its QSA selection; only the head's layer changes. `window_pool_ids` (engine/modules/prefill_indexer)
writes the groups in a selection's form -- ascending, -1 after -- so `qsa_attend` reads them as it reads a scored
selection, and `Qwen38Net._qsa(window=...)` skips what only a scored layer needs: the index keys' write and the scoring
and top-k over every group the row sees, the part of a draft that grows with the context.

Held here: the ids against a plain reading of the rule; that a windowed layer writes no index key and scores nothing,
takes the covered launch only where the window is the whole budget, and that a scored layer is untouched; the fleet
flag's parsing; and, on the served kernels (a GPU, or TRITON_INTERPRET=1), that the block-form attention over window ids
is the attention over the window's positions and the row's incomplete group:

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_mtp_window
"""
import types
import unittest
from types import SimpleNamespace

from unittest import mock

from tests.test_engine_qwen38_kernels import (DEVICE, INTERPRET, RUNS, RUNS_REASON, W, Held, BF16_STEP, Launches,
                                              block_table, generator, host_meta, paged, randn, served_kernels, torch)


def expected(seen: int, width: int, sink: int, recent: int) -> "list[int]":
    """The rule read plainly: every group while they fit the window, else the first `sink` and the last `recent`."""
    groups = list(range(seen)) if seen <= sink + recent else list(range(sink)) + list(range(seen - recent, seen))
    return groups + [-1] * (width - len(groups))


@unittest.skipUnless(torch is not None, "requires torch")
class WindowIdsTests(unittest.TestCase):
    def test_the_ids_are_the_sink_and_the_latest_groups(self):
        from engine.modules.prefill_indexer import window_pool_ids
        width = 16
        for sink, recent in ((1, 15), (1, 5), (0, 6), (3, 3), (0, 16)):
            seen = torch.tensor([0, 1, 5, 6, 7, 15, 16, 17, 40, 1000], dtype=torch.int32)
            got = window_pool_ids(seen, width, sink, recent)
            with self.subTest(sink=sink, recent=recent):
                self.assertEqual(got.dtype, torch.int32)
                self.assertEqual(tuple(got.shape), (len(seen), width))
                self.assertEqual(got.tolist(), [expected(int(s), width, sink, recent) for s in seen])

    def test_the_whole_budget_is_the_covered_selection_where_it_covers(self):
        from engine.modules.prefill_indexer import covered_pool_ids, window_pool_ids
        seen = torch.arange(0, 17, dtype=torch.int32)
        self.assertTrue(torch.equal(window_pool_ids(seen, 16, 1, 15), covered_pool_ids(seen, 16)))

    def test_a_window_outside_the_width_is_refused(self):
        from engine.modules.prefill_indexer import window_pool_ids
        seen = torch.tensor([3], dtype=torch.int32)
        for sink, recent in ((1, 16), (-1, 4), (2, 0), (17, 1)):
            with self.subTest(sink=sink, recent=recent), self.assertRaises(ValueError):
                window_pool_ids(seen, 16, sink, recent)


class Calls:
    """A lane that records its calls and answers a fixed tensor."""

    def __init__(self, log, name, answer):
        self.log, self.name, self.answer = log, name, answer

    def __call__(self, *args, **kwargs):
        self.log.append((self.name, args, kwargs))
        return self.answer


@unittest.skipUnless(torch is not None, "requires torch")
class WindowedLayerTests(unittest.TestCase):
    """`Qwen38Net._qsa` over lanes that record: what a windowed layer calls, and what a scored one still does."""

    F = SimpleNamespace(heads_local=2, head_dim=4, kv_heads_local=1, idx_heads=1, idx_dim=2, idx_ratio=4, idx_budget=16,
                        index_blocks=4, rms_eps=1e-6, rope_theta=1e7, rotary_dim=2)

    def layer(self, positions, *, window, captured=False):
        from engine.profiles.qwen38.net import Qwen38Net
        F, N = self.F, len(positions)
        log = []
        width = F.heads_local * 2 * F.head_dim + 2 * F.kv_heads_local * F.head_dim + F.idx_heads * F.idx_dim + F.idx_dim
        attended = torch.ones(N, F.heads_local, F.head_dim)
        scored = torch.full((N, F.index_blocks), 7, dtype=torch.int32)
        lanes = SimpleNamespace(
            qsa_index_keys=Calls(log, "index_keys", None),
            qsa_inputs=Calls(log, "inputs", (torch.zeros(N, F.heads_local, F.head_dim),
                                             torch.zeros(N, F.idx_heads, F.idx_dim))),
            qsa_select=Calls(log, "select", scored),
            qsa_attend=Calls(log, "attend", attended),
            qsa_attend_covered=Calls(log, "covered", attended))
        norms = {f"mtp.L0.attn.{w}": None for w in ("q_norm", "k_norm", "idx_q_norm", "idx_k_norm")}
        stand_in = SimpleNamespace(F=F, p=norms, lanes=lanes, comm=SimpleNamespace(all_reduce=lambda t: t),
                                   linear=lambda x, name: torch.zeros(x.shape[0], width if "in_proj" in name else 3),
                                   _sharded_blocks=lambda *a: None, _score_runs=Qwen38Net._score_runs)
        stand_in._window_blocks = types.MethodType(Qwen38Net._window_blocks, stand_in)
        stand_in._covered_blocks = types.MethodType(Qwen38Net._covered_blocks, stand_in)
        pos = torch.tensor(positions, dtype=torch.int32)
        meta = SimpleNamespace(positions32=pos, positions=pos.to(torch.int64), lengths=None, page_table=None,
                               rows_req=torch.zeros(N, dtype=torch.int32), slot_table=None, starts=None, key_slots=None,
                               kv_slots=None, ring_slots=None, groups_seen=None, covered_blocks=None)
        if captured:
            step = SimpleNamespace(captured=True, tokens=1)
        else:
            step = SimpleNamespace(segments=[SimpleNamespace(ctx=positions[0], length=N)])
        caches = SimpleNamespace(kv=lambda L: (None, None), key_ring=lambda L: None, index_keys=lambda L: "keys")
        Qwen38Net._qsa(stand_in, 48, torch.zeros(N, 8), step, meta, caches, prefix="mtp.L0.attn.", cache_layer=48,
                       window=window)
        return log, scored

    def names(self, log):
        return [name for name, _, _ in log]

    def test_a_scored_layer_writes_its_keys_and_attends_its_selection(self):
        log, scored = self.layer([40, 41, 42], window=None)
        self.assertEqual(self.names(log), ["index_keys", "inputs", "select", "attend"])
        self.assertIs(log[-1][1][3], scored)

    def test_a_windowed_layer_writes_no_key_and_scores_nothing(self):
        from engine.modules.prefill_indexer import window_pool_ids
        for captured in (False, True):
            positions = [40, 41, 42]
            log, _ = self.layer(positions, window=(1, 2), captured=captured)
            with self.subTest(captured=captured):
                self.assertEqual(self.names(log), ["inputs", "attend"])
                seen = (torch.tensor(positions, dtype=torch.int32) + 1) // 4
                self.assertTrue(torch.equal(log[-1][1][3], window_pool_ids(seen, 4, 1, 2)))

    def test_the_covered_launch_only_where_the_window_is_the_whole_budget(self):
        log, _ = self.layer([1, 2, 3], window=(1, 3))                       # a host step the budget covers
        self.assertEqual(self.names(log), ["inputs", "covered"])
        log, _ = self.layer([12, 13, 14], window=(1, 1))                    # covered, but the window is narrower
        self.assertEqual(self.names(log), ["inputs", "attend"])
        self.assertEqual(log[-1][1][3].tolist(), [[0, 2, -1, -1], [0, 2, -1, -1], [0, 2, -1, -1]])


@unittest.skipUnless(torch is not None, "requires torch")
class FleetFlagTests(unittest.TestCase):
    def test_the_flag_parses_and_refuses(self):
        from engine.profiles.qwen38.fleet import mtp_window
        self.assertIsNone(mtp_window(None))
        self.assertIsNone(mtp_window("off"))
        self.assertEqual(mtp_window("1,511"), (1, 511))
        self.assertEqual(mtp_window("0,64"), (0, 64))
        for bad in ("1", "a,b", "1,0", "-1,4"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                mtp_window(bad)

    def test_the_fleet_serves_the_window_by_default(self):
        """The operator's decision of 2026-09-19 ("전부 켜"): the fleet's head attends 1,511; `off` scores. The net
        itself (a probe's, a test's) scores unless told."""
        from pathlib import Path
        from engine.profiles.qwen38.fleet import MTP_WINDOW
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(MTP_WINDOW, (1, 511))
        net = (root / "engine/profiles/qwen38/net.py").read_text()
        self.assertIn("self.mtp_window = None", net)
        self.assertIn("window=self.mtp_window", net)
        fleet = (root / "engine/profiles/qwen38/fleet.py").read_text()
        self.assertIn('ap.add_argument("--mtp-window", default=f"{MTP_WINDOW[0]},{MTP_WINDOW[1]}"', fleet)
        launcher = (root / "launchers/start-st-qwen38.sh").read_text()
        self.assertIn("^[0-9]+,[1-9][0-9]*$|^off$", launcher)
        probe = (root / "probes/engine_qwen38_step.py").read_text()
        self.assertIn("net.mtp_window = MTP_WINDOW", probe)


class ProbeTests(unittest.TestCase):
    """probes/engine_qwen38_mtp_window: the lane's arms are the flag's windows, and it is the lane's to run."""

    def test_the_arms_are_the_served_head_and_windows_the_flag_takes(self):
        from probes.engine_qwen38_mtp_window import ARMS
        self.assertIsNone(ARMS["scored"])
        for arm, window in ARMS.items():
            if window is not None:
                self.assertEqual(arm, f"window-{window[1]}")
                self.assertTrue(window[0] >= 0 and window[1] > 0 and sum(window) <= 512)

    def test_the_saving_is_the_scored_head_less_each_window_per_bucket(self):
        from probes.engine_qwen38_mtp_window import saving
        arm = lambda wall, dev, launches, qsa: {"wall_us": wall, "device_us": dev, "launches": launches, "qsa_us": qsa}
        report = {"arms": {"scored": None, "window-511": (1, 511)},
                  "buckets": {"6": {"context": 4601, "scored": arm(7000.0, 6900.0, 127.0, 170.0),
                                    "window-511": arm(6900.5, 6850.0, 124.0, 90.0)}}}
        self.assertEqual(saving(report), {"6": {"context": 4601, "window-511": {"wall_us": 99.5, "device_us": 50.0,
                                                                                "launches": 3.0, "qsa_us": 80.0}}})

    def test_the_arms_are_captured_from_one_net_and_replayed_in_turns(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "probes/engine_qwen38_mtp_window.py").read_text()
        self.assertIn("net.mtp_window = ARMS[arm]", source)
        self.assertIn("for arm in (arms if turn % 2 == 0 else tuple(reversed(arms))):", source)

    def test_the_kernel_check_dispatches_the_lane(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "probes/engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_mtp_window'", source)
        self.assertIn("from probes.engine_qwen38_mtp_window import run", source)


@unittest.skipUnless(RUNS, RUNS_REASON)
class WindowKernelTests(unittest.TestCase):
    """kernels/qsa_window: the ids in one launch a step, byte for byte the torch form's."""

    def test_one_launch_writes_what_the_torch_form_does(self):
        from engine.kernels import qsa_window
        from engine.modules.prefill_indexer import window_pool_ids
        g = torch.Generator().manual_seed(9)
        positions = torch.cat([torch.arange(0, 12), torch.randint(0, 5000, (40,), generator=g)]).to(torch.int32)
        launches = Launches(qsa_window._window_rows)
        cases = ((16, 1, 15), (16, 1, 5), (16, 0, 6), (512, 1, 511), (512, 1, 127))
        for width, sink, recent in cases:
            with served_kernels(), mock.patch.object(qsa_window, "_window_rows", launches):
                got = qsa_window.window_ids(positions.to(DEVICE), 4, width, sink, recent)
            want = window_pool_ids((positions + 1) // 4, width, sink, recent)
            with self.subTest(width=width, sink=sink, recent=recent):
                self.assertEqual(got.dtype, torch.int32)
                self.assertTrue(torch.equal(got.cpu(), want))
        self.assertEqual(launches.grids, [(len(positions),)] * len(cases))


@unittest.skipUnless(RUNS, RUNS_REASON)
class WindowedAttentionTests(Held):
    """qsa_sparse_paged_attention_blocks over window ids against modules/sparse_attention.gqa_sparse over the positions
    they name and the row's incomplete group, read through the request's pages."""

    def test_the_attention_over_a_window_is_the_attention_over_its_positions(self):
        from engine.kernels import qsa
        from engine.modules.prefill_indexer import window_pool_ids
        from engine.modules.sparse_attention import gqa_sparse
        gen = generator(2607)
        D, page, ratio = W.head_dim, W.block, W.ratio
        budget = 64 if INTERPRET else W.budget
        groups = budget // ratio
        sink, recent = (1, 5) if INTERPRET else (1, 127)
        wide = (sink + recent + 3) * ratio + 2                                   # rows past the window, and inside it
        requests = ((0, 1, 0, 3), (1, 2, wide - 6, 6), (2, 3, 7, 1))
        pages = 8 * len(requests)
        blocks = max(-(-(ctx + length) // page) for _, _, ctx, length in requests)
        table = block_table(gen, requests, blocks, page, pages)
        meta = host_meta(requests, table, block=page, ratio=ratio)
        k_cache, _ = paged(gen, pages, page, W.kv_heads, D)
        v_cache, _ = paged(gen, pages, page, W.kv_heads, D)
        rows = meta.positions.numel()
        q = randn(gen, rows, W.heads, D)
        seen = (meta.positions32 + 1) // ratio
        ids = window_pool_ids(seen, groups, sink, recent)
        with served_kernels():
            got = qsa.qsa_sparse_paged_attention_blocks(q, k_cache, v_cache, ids, meta.positions32, meta.lengths, ratio,
                                                        budget, meta.page_table, meta.rows_req)
        width = budget + ratio - 1
        slots = torch.zeros(rows, width, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        pages_of, owners = table.cpu(), meta.rows_req.cpu().tolist()
        for r in range(rows):
            p = int(meta.positions32[r])
            held = [b * ratio + j for b in ids[r].tolist() if b >= 0 for j in range(ratio)]
            held += list(range((p + 1) // ratio * ratio, p + 1))                 # the incomplete group, always
            held = torch.tensor(held, dtype=torch.int64)
            seq = requests[owners[r]][0]
            slots[r, :len(held)] = (pages_of[seq, held // page] * page + held % page).to(torch.int32)
            valid[r] = len(held)
        reference = gqa_sparse(q, k_cache.reshape(pages * page, W.kv_heads, D), v_cache.reshape(pages * page, W.kv_heads, D),
                               slots.to(DEVICE), valid.to(DEVICE), D ** -0.5)
        self.assertGreater(int(seen.max()), sink + recent)                      # a window slid
        self.assertWithin(got, reference, (2 * BF16_STEP, BF16_STEP), "the windowed rows")


if __name__ == "__main__":
    unittest.main()
