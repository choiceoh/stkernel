"""engine/profiles/qwen38/ple_table and the served net's PLE path over it: the rank's table on the SSD, rows read by
id -- an eager step's gather (`_ple_embed`) and a captured step's staging (`stage_ple`, the graph's static rows).

The table's rows are the checkpoint's e4m3 bytes; a row this rank does not hold is zero, so the all-reduce sums the one
rank that holds each row. The staged rows of a captured step must be the rows an eager step would have hashed for
the same tokens and history (the ids ring's carried tokens, DEAD before the sequence)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
    import numpy as np
    from tests.test_engine_qwen38_preshard import MTP_BLOCK, TINY, nvidia_config, facts_of


def write_table(path: Path, rows: int, width: int, seed: int = 0) -> np.ndarray:
    """Random e4m3 rows, none of them NaN (0x7F and 0xFF): a NaN never compares equal to itself."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(rows, width), dtype=np.uint8)
    data[data == 0x7F] = 0x7E
    data[data == 0xFF] = 0xFE
    path.write_bytes(data.tobytes())
    return data


@unittest.skipUnless(torch is not None, "requires torch")
class TableTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "ple-r0of4.weight"
        self.rows, self.width = 1000, 8
        self.data = write_table(self.path, self.rows, self.width)

    def test_gather_returns_rows_in_order_with_repeats_on_both_read_paths(self):
        from engine.profiles.qwen38.ple_table import PLETable, SPLIT_AT
        table = PLETable(self.path, rows=self.rows, width=self.width, scale=0.5, threads=4)
        self.addCleanup(table.close)
        few = np.array([3, 999, 3, 0, 17], dtype=np.int64)
        self.assertTrue(np.array_equal(table.gather(few), self.data[few]))
        many = np.random.default_rng(1).integers(0, self.rows, size=SPLIT_AT * 5, dtype=np.int64)
        self.assertTrue(np.array_equal(table.gather(many), self.data[many]))
        self.assertEqual(table.gather(np.zeros(0, dtype=np.int64)).shape, (0, self.width))
        self.assertEqual(table.reads, 2)
        with self.assertRaises(IndexError):
            table.gather(np.array([self.rows], dtype=np.int64))
        with self.assertRaises(IndexError):
            table.gather(np.array([-1], dtype=np.int64))

    def test_a_gather_is_one_take_over_the_mapping_not_a_read_a_row(self):
        """The rows come out of a read-only mapping of the file (one C loop, the GIL released): no `pread` a row on
        either path, the counters count what was asked, and closing releases the mapping before the descriptor."""
        from unittest import mock
        from engine.profiles.qwen38 import ple_table
        table = ple_table.PLETable(self.path, rows=self.rows, width=self.width, scale=0.5, threads=4)
        many = np.random.default_rng(2).integers(0, self.rows, size=ple_table.SPLIT_AT * 9 + 5, dtype=np.int64)
        with mock.patch.object(ple_table.os, "pread", side_effect=AssertionError("a row was read by pread"), create=True):
            self.assertTrue(np.array_equal(table.gather(many[:7]), self.data[many[:7]]))
            self.assertTrue(np.array_equal(table.gather(many), self.data[many]))       # the pool's chunks, in order
        self.assertEqual((table.reads, table.rows_read), (2, 7 + many.shape[0]))
        got = table.gather(many[:3])
        table.close()
        self.assertTrue(np.array_equal(got, self.data[many[:3]]))                      # a gather is a copy, not a view
        self.assertEqual((table.fd, table._map, table._rows), (-1, None, None))
        self.path.unlink()                                                             # nothing holds the file

    def test_a_file_of_the_wrong_size_is_refused(self):
        from engine.profiles.qwen38.ple_table import PLETable
        with self.assertRaises(ValueError):
            PLETable(self.path, rows=self.rows + 1, width=self.width, scale=1.0)

    def test_local_rows_partition_the_table_across_ranks(self):
        from engine.profiles.qwen38.ple_table import local_rows
        rows = np.arange(0, 40, dtype=np.int64).reshape(5, 8)
        held = np.zeros(40, dtype=bool)
        for rank in range(4):
            local, mine = local_rows(rows, rank, 10)
            self.assertTrue(((local >= 0) & (local < 10)).all())
            self.assertTrue(np.array_equal(rows[mine], np.arange(rank * 10, (rank + 1) * 10)))
            held[rows[mine]] = True
        self.assertTrue(held.all())

    def test_open_checks_the_sidecar_against_the_profile(self):
        from engine.profiles.qwen38 import facts, specs
        from engine.profiles.qwen38.ple_table import PLETable, write_sidecar
        F = facts_of(nvidia_config())
        rank, L = 2, 1
        rows, width = F.ple_rows_per_rank, F.ple_head_dim
        d = Path(self.dir.name)
        data = write_table(d / facts.ple_file(rank), rows, width, seed=3)
        good = dict(layout=F.weight_layout, rank=rank, world=4, rows=rows, width=width, dtype="F8_E4M3",
                    shards=specs.ple_shards(F, L, rank), scale=0.0002)
        write_sidecar(d / facts.ple_sidecar(rank), **good)
        table = PLETable.open(d, rank, F)
        self.addCleanup(table.close)
        self.assertEqual((table.rows, table.width, table.scale), (rows, width, 0.0002))
        self.assertTrue(np.array_equal(table.gather(np.array([1, rows - 1], dtype=np.int64)), data[[1, rows - 1]]))
        for key, bad in (("layout", "st-qwen38-tep4-modelopt-v2"), ("rank", 1), ("rows", rows - 1), ("shards", good["shards"][::-1])):
            write_sidecar(d / facts.ple_sidecar(rank), **dict(good, **{key: bad}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                PLETable.open(d, rank, F)


@unittest.skipUnless(torch is not None, "requires torch")
class StagingTests(unittest.TestCase):
    def test_fill_zeroes_foreign_rows_and_upload_copies_the_filled_rows(self):
        from engine.profiles.qwen38.ple_table import PLEStaging, PLETable, local_rows
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.weight"
            data = write_table(path, 50, 4)
            table = PLETable(path, rows=50, width=4, scale=1.0)
            stage = PLEStaging(6, 3, 4, torch.device("cpu"))
            rows = np.array([[10, 60, 199], [0, 49, 120]], dtype=np.int64)     # rank 0 of 4 holds rows 0..49
            local, mine = local_rows(rows, 0, 50)
            self.assertEqual(stage.fill(table, local, mine), 2)
            stage.upload()
            got = stage.device[:2].numpy()
            self.assertTrue(np.array_equal(got[0, 0], data[10]) and np.array_equal(got[1, 0], data[0]) and np.array_equal(got[1, 1], data[49]))
            self.assertFalse(got[0, 1].any() or got[0, 2].any() or got[1, 2].any())
            with self.assertRaises(ValueError):
                stage.fill(table, np.zeros((7, 3), dtype=np.int64), np.zeros((7, 3), dtype=bool))
            table.close()


@unittest.skipUnless(torch is not None, "requires torch")
class NetPathTests(unittest.TestCase):
    """The served net's two PLE paths over a rank's table, on the CPU: the tensors `_ple_feature` binds, the hash, a
    table file for the rank's vocabulary range."""
    RANK = 1

    def setUp(self):
        from engine.modules.ngram_embedding import NGramHash
        from engine.profiles.qwen38 import facts
        from engine.profiles.qwen38.net import Qwen38Net
        from engine.profiles.qwen38.ple_table import PLETable
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        F = self.F = facts_of(nvidia_config())
        t = TINY
        L = 1
        H, hc, D = F.hidden, F.hc, F.ple_dim
        g = torch.Generator().manual_seed(0)
        bf = lambda *shape, scale=0.1: (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)
        made = NGramHash.splitmix(ngram_size=F.ngram_size, heads=F.heads_per_ngram, unigram_vocab=F.vocab, base=F.ngram_base,
                                  table_index=0, seed=F.seed, eos=F.eos)
        self.scale = 0.0002
        p = {f"L{L}.ple.kv_proj": bf(hc * H + H, D), f"L{L}.ple.norm_key": bf(hc * H, scale=0.3),
             f"L{L}.ple.norm_query": bf(hc * H, scale=0.3), f"L{L}.ple.norm_conv": bf(hc * H, scale=0.3),
             f"L{L}.ple.conv": (torch.randn(hc * H, F.ple_conv, generator=g) * 0.3), f"L{L}.ple.scale": torch.tensor([self.scale]),
             f"L{L}.ple.layer_multipliers": made.multipliers.clone(), f"L{L}.ple.heads_offsets": made.offsets.clone(),
             f"L{L}.ple.heads_vocab": made.sizes.clone()}
        net = self.net = object.__new__(Qwen38Net)
        net.F, net.p, net.rank, net.layers = F, p, self.RANK, list(range(F.layers))
        net.comm = SimpleNamespace(all_reduce=lambda x: x, rank=self.RANK, world_size=4)
        net._ple = net._ple_feature(L)
        net._ple_hash = net._ple.hashes(L)
        net._ple_scale = p[f"L{L}.ple.scale"].float().reshape(())
        net.ple_table = net.ple_stage = None
        path = Path(self.dir.name) / facts.ple_file(self.RANK)
        self.data = write_table(path, F.ple_rows_per_rank, F.ple_head_dim, seed=5)
        self.table = PLETable(path, rows=F.ple_rows_per_rank, width=F.ple_head_dim, scale=self.scale, threads=2)
        self.addCleanup(self.table.close)
        net.attach_ple(self.table, max_rows=8)

    def expected(self, rows: torch.Tensor) -> torch.Tensor:
        """What the rows [N, heads] must become on this rank: its rows' e4m3 bytes times the scale, other ranks' zero."""
        F = self.F
        local = rows.numpy() - self.RANK * F.ple_rows_per_rank
        mine = (local >= 0) & (local < F.ple_rows_per_rank)
        raw = np.zeros((*rows.shape, F.ple_head_dim), dtype=np.uint8)
        raw[mine] = self.data[local[mine]]
        return (torch.from_numpy(raw).view(torch.float8_e4m3fn).float() * self.scale).to(torch.bfloat16).flatten(-2)

    def test_attach_refuses_a_table_of_another_shape_or_scale(self):
        from engine.profiles.qwen38.ple_table import PLETable
        F = self.F
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.weight"
            write_table(path, F.ple_rows_per_rank, F.ple_head_dim)
            # a refused table is closed here: it maps its file, and a mapped file cannot be rewritten everywhere
            table = PLETable(path, rows=F.ple_rows_per_rank, width=F.ple_head_dim, scale=self.scale * 2)
            with self.assertRaisesRegex(ValueError, "scale"):
                self.net.attach_ple(table, max_rows=4)
            table.close()
            write_table(path, F.ple_rows_per_rank + 1, F.ple_head_dim)
            table = PLETable(path, rows=F.ple_rows_per_rank + 1, width=F.ple_head_dim, scale=self.scale)
            with self.assertRaisesRegex(ValueError, "rows"):
                self.net.attach_ple(table, max_rows=4)
            table.close()

    def test_an_eager_gather_is_the_rows_this_rank_holds_scaled_and_the_rest_zero(self):
        F = self.F
        rows = torch.randint(0, F.ple_rows_total, (7, F.ple_heads), dtype=torch.int64, generator=torch.Generator().manual_seed(2))
        rows[0, :] = self.RANK * F.ple_rows_per_rank + torch.arange(F.ple_heads)          # a row of this rank's
        rows[1, :] = 0                                                                     # rank 0's
        got = self.net._ple_embed(rows)
        self.assertEqual((got.dtype, tuple(got.shape)), (torch.bfloat16, (7, F.ple_heads * F.ple_head_dim)))
        self.assertTrue(torch.equal(got, self.expected(rows)))
        self.assertFalse(bool(got[1].any()))
        self.assertTrue(bool(got[0].any()))

    def test_staged_rows_are_the_eager_hash_of_each_row_with_its_carried_ids(self):
        """A captured step of n rows x t tokens: stage_ple gathers what `_ple_inject` would hash segment by segment --
        carried ids from the ring (DEAD before the sequence), then the tokens."""
        from engine.modules.ngram_embedding import DEAD
        F, net = self.F, self.net
        n, t, r_ids = 3, 2, 8
        slots, contexts = [1, 2, 3], [5, 0, 1]
        ids = [11, 12, 21, 22, 31, 32]
        ids_ring = torch.full((4, r_ids), -7, dtype=torch.int64)                         # stale cells: never read below 0
        ids_ring[1, (5 - 2) % r_ids], ids_ring[1, (5 - 1) % r_ids] = 101, 102
        ids_ring[3, 0] = 301
        caches = SimpleNamespace(ple_fields=lambda: (ids_ring, None))
        net.stage_ple(slots, contexts, ids, t, caches)
        net.ple_stage.upload()
        got = net._ple_values(net.ple_stage.device[:n * t])
        histories = [[101, 102, 11, 12], [DEAD, DEAD, 21, 22], [DEAD, 301, 31, 32]]
        rows = torch.cat([net._ple_hash.rows(torch.tensor(h, dtype=torch.int64), t) for h in histories])
        self.assertTrue(torch.equal(got, self.expected(rows)))
        self.assertEqual(net._ple_embed(rows).tolist(), got.tolist())
        # the same rows when a host that holds the carried tokens hands them over: the rings are not read at all
        staged = net.ple_stage.host[:n * t].clone()
        net.ple_stage.host.zero_()
        unread = SimpleNamespace(ple_fields=lambda: self.fail("the carried ids were handed over: no ring read"))
        net.stage_ple(slots, contexts, ids, t, unread, carried=[[101, 102], [DEAD, DEAD], [DEAD, 301]])
        self.assertTrue(torch.equal(net.ple_stage.host[:n * t], staged))


if __name__ == "__main__":
    unittest.main()
