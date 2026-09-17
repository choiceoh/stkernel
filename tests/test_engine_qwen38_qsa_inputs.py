"""A QSA layer's inputs in two launches (engine/QWEN38_CARRY.md Q3).

net._qsa normalised and rotated the query heads, the key head and the index query heads (three norm_rope_partial
launches), stored K, V, the index keys and the raw keys (four qsa_store_cache_rows launches), pooled the groups the step
closes (qsa_compress_groups_with_ratio) and normalised and rotated those (a fourth norm_rope_partial): nine launches a
layer. qsa.qsa_index_keys does the compression, its norm and rotation and the index key store in one; qsa.qsa_inputs the
rest in one, the key head normalised and rotated straight into K. The arithmetic and the addresses are the nine launches'
-- and the order that matters is kept: the index keys read the raw-key ring before this step's raw keys overwrite it
(a prefill longer than the ring writes every cell a continuing prefill's first groups read).

Both paths run on the served kernels -- on a GPU, or under TRITON_INTERPRET=1 with tests/test_engine_qwen38_kernels'
accommodations -- from the same cache bytes, and every output and every cache storage is compared byte for byte.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_qsa_inputs
"""
from pathlib import Path
from types import SimpleNamespace
import unittest

from tests.test_engine_qwen38_kernels import (DEVICE, EPS, RUNS, RUNS_REASON, THETA, W, block_table, generator, host_meta,
                                              paged, randn, served_kernels, torch)

ROOT = Path(__file__).resolve().parents[1]


def layer_case(seed):
    """A step's QSA layer: (seq, slot, ctx, length) -- a first prefill longer than the ring, a prefill continuing at
    context 6 past the ring's width (its first groups read ring members its last rows overwrite), a decode row closing
    a group from the ring, and a verify pair across a group boundary -- over paged K/V, index key and ring caches that
    start random and share their storage with other regions."""
    from engine.profiles.qwen38.caches import QSA_KEY_RING
    gen = generator(seed)
    D, Di, ratio, block = W.head_dim, W.idx_dim, W.ratio, W.block
    requests = ((0, 1, 0, 21), (1, 2, 6, 21), (2, 3, 11, 1), (3, 4, 26, 2))
    pages = 8 * len(requests)
    table = block_table(gen, requests, 8, block, pages)
    meta = host_meta(requests, table, block=block, ratio=ratio)
    rows = meta.positions.numel()
    idx_q = W.idx_heads * Di
    proj = randn(gen, rows, W.heads * 2 * D + 2 * W.kv_heads * D + idx_q + Di)
    caches = dict(k=paged(gen, pages, block, 1, D), v=paged(gen, pages, block, 1, D),
                  keys=paged(gen, pages, block // ratio, 1, Di), ring=paged(gen, 8, QSA_KEY_RING, 1, Di))
    weights = dict(q=randn(gen, D, scale=0.1), k=randn(gen, D, scale=0.1), iq=randn(gen, Di, scale=0.1),
                   ik=randn(gen, Di, scale=0.1))
    return SimpleNamespace(meta=meta, rows=rows, proj=proj, caches=caches, weights=weights)


def split(case):
    """The layer's in_proj split as net._qsa makes it: views of one row."""
    D, Di, n = W.head_dim, W.idx_dim, case.rows
    idx_q = W.idx_heads * Di
    qg, k, v, idx = case.proj.split([W.heads * 2 * D, D, D, idx_q + Di], dim=-1)
    return qg.view(n, W.heads, 2 * D), k.view(n, 1, D), v.view(n, 1, D), idx[:, :idx_q].view(n, W.idx_heads, Di), \
        idx[:, idx_q:]


def fresh(case):
    """Each cache view over its own copy of the case's starting storage: (views, storages)."""
    views, storages = {}, {}
    for name, (view, storage) in case.caches.items():
        copy = storage.clone()
        views[name] = copy.as_strided(view.shape, view.stride(), view.storage_offset())
        storages[name] = copy
    return views, storages


def nine_launches(case, c):
    from engine.kernels import qsa
    m, w, n = case.meta, case.weights, case.rows
    qg, k, v, iq_rows, ik = split(case)
    q = qsa.norm_rope_partial(qg[..., :W.head_dim], w["q"], EPS, m.positions, THETA, W.rotary)
    key = qsa.norm_rope_partial(k, w["k"], EPS, m.positions, THETA, W.rotary)
    qsa.qsa_store_cache_rows(c["k"], m.kv_slots, key.reshape(n, W.head_dim))
    qsa.qsa_store_cache_rows(c["v"], m.kv_slots, v.reshape(n, W.head_dim))
    iq = qsa.norm_rope_partial(iq_rows, w["iq"], EPS, m.positions, THETA, W.rotary)
    pooled, first = qsa.qsa_compress_groups_with_ratio(ik[:, None, :], m.positions[:, None, None].expand(n, 1, 3),
                                                       c["ring"], m.slot_table, m.rows_req, m.starts, m.positions,
                                                       m.key_slots, W.ratio)
    keys = qsa.norm_rope_partial(pooled, w["ik"], EPS, first[:, 0], THETA, W.rotary)
    qsa.qsa_store_cache_rows(c["keys"], m.key_slots, keys[:, 0])
    qsa.qsa_store_cache_rows(c["ring"], m.ring_slots, ik)
    return q, iq


def two_launches(case, c, *, ring_first=False):
    from engine.kernels import qsa
    m, w = case.meta, case.weights
    qg, k, v, iq_rows, ik = split(case)

    def index_keys():
        qsa.qsa_index_keys(ik, c["ring"], m.slot_table, m.rows_req, m.starts, m.positions, m.key_slots, W.ratio, w["ik"],
                           EPS, THETA, W.rotary, c["keys"])

    def inputs():
        return qsa.qsa_inputs(qg[..., :W.head_dim], k, v, iq_rows, ik, m.positions, w["q"], w["k"], w["iq"], EPS, THETA,
                              W.rotary, c["k"], c["v"], m.kv_slots, c["ring"], m.ring_slots)
    if ring_first:                                                   # the negative control: the ring written first
        out = inputs()
        index_keys()
        return out
    index_keys()
    return inputs()


@unittest.skipUnless(RUNS, RUNS_REASON)
class QsaInputsTests(unittest.TestCase):
    def test_two_launches_are_the_nine(self):
        case = layer_case(81)
        m = case.meta
        self.assertTrue(bool((m.ring_slots < 0).any()) and bool((m.key_slots >= 0).any()))
        want_views, want = fresh(case)
        got_views, got = fresh(case)
        with served_kernels():
            q_want, iq_want = nine_launches(case, want_views)
            q_got, iq_got = two_launches(case, got_views)
        self.assertTrue(torch.equal(q_got, q_want))
        self.assertTrue(torch.equal(iq_got, iq_want))
        for name in ("k", "v", "keys", "ring"):
            with self.subTest(cache=name):
                self.assertFalse(torch.equal(want[name], case.caches[name][1]))       # the step wrote into it
                self.assertTrue(torch.equal(got[name], want[name]))

    def test_the_ring_is_read_before_it_is_written(self):
        """The order is the fold's: writing this step's raw keys first changes the continuing prefill's index keys."""
        case = layer_case(82)
        want_views, want = fresh(case)
        got_views, got = fresh(case)
        with served_kernels():
            nine_launches(case, want_views)
            two_launches(case, got_views, ring_first=True)
        self.assertTrue(torch.equal(got["ring"], want["ring"]))
        self.assertFalse(torch.equal(got["keys"], want["keys"]))

    def test_the_entries_refuse_before_any_launch(self):
        from engine.kernels import qsa
        case = layer_case(83)
        c, _ = fresh(case)
        m, w = case.meta, case.weights
        qg, k, v, iq_rows, ik = split(case)
        call = lambda **kw: qsa.qsa_inputs(kw.get("q", qg[..., :W.head_dim]), k, v, kw.get("iq", iq_rows), ik,
                                           kw.get("positions", m.positions), w["q"], w["k"], w["iq"], EPS, THETA, W.rotary,
                                           c["k"], c["v"], m.kv_slots, c["ring"], m.ring_slots)
        if DEVICE == "cpu":
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                call()
        with served_kernels():
            with self.assertRaisesRegex(ValueError, "int64 positions"):
                call(positions=m.positions.to(torch.int32))
            with self.assertRaisesRegex(ValueError, "unit stride"):
                call(iq=iq_rows.transpose(1, 2).contiguous().transpose(1, 2))
            with self.assertRaisesRegex(ValueError, "unit stride"):
                qsa.qsa_index_keys(ik.t().contiguous().t(), c["ring"], m.slot_table, m.rows_req, m.starts, m.positions,
                                   m.key_slots, W.ratio, w["ik"], EPS, THETA, W.rotary, c["keys"])


class ServedLaneTests(unittest.TestCase):
    def test_the_layer_launches_the_index_keys_then_the_inputs(self):
        source = (ROOT / "engine/profiles/qwen38/net.py").read_text()
        body = source[source.index("    def _qsa("):source.index("    # -- MoE")]
        self.assertLess(body.index("lanes.qsa_index_keys("), body.index("lanes.qsa_inputs("))
        for gone in ("lanes.norm_rope(", "lanes.qsa_store(", "lanes.qsa_compress("):
            self.assertNotIn(gone, body)
        lanes = (ROOT / "engine/profiles/qwen38/lanes.py").read_text()
        self.assertIn("qsa_index_keys=on_main(qsa.qsa_index_keys)", lanes)
        self.assertIn("qsa_inputs=on_main(qsa.qsa_inputs)", lanes)


if __name__ == "__main__":
    unittest.main()
