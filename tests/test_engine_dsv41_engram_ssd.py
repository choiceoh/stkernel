"""engine/modules/lookup_table and the DSv4.1 profile's engram over it: the rank's table on the SSD, rows read by
local id off a read-only mapping, dequantised the way a resident table would be.

The reader this replaced (`dsv41_engram_io.ShardReader`, O_DIRECT a sector a row) left the tree with the vLLM overlay
(#1152), so `engram.attach` raised FileNotFoundError before it read anything; these hold the mapped form to what the
family's own dequantisation returns (modules/ngram_embedding.block_fp8_rows) rather than to the retired reader.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

np = None
torch = None
if importlib.util.find_spec("numpy") is not None:
    import numpy as np
if importlib.util.find_spec("torch") is not None:
    import torch

ROOT = Path(__file__).resolve().parents[1]


def write_table(path: Path, rows: int, width: int, seed: int = 0):
    """Random e4m3 rows, none of them NaN (0x7F and 0xFF): a NaN never compares equal to itself."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(rows, width), dtype=np.uint8)
    data[(data == 0x7F) | (data == 0xFF)] = 0x3C
    path.write_bytes(data.tobytes())
    return data


def write_scales(path: Path, rows: int, per_row: int, seed: int = 1):
    """Random e8m0 scale rows around 2^0, never 0xFF (e8m0's NaN)."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0x70, 0x88, size=(rows, per_row), dtype=np.uint8)
    path.write_bytes(data.tobytes())
    return data


@unittest.skipUnless(np is not None, "requires numpy")
class MappedTableTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "engram-l1-r0of4.weight"
        self.rows, self.width = 700, 8
        self.data = write_table(self.path, self.rows, self.width)

    def test_gather_returns_rows_in_order_with_repeats_on_both_read_paths(self):
        from engine.modules.lookup_table import MappedTable, SPLIT_AT
        table = MappedTable(self.path, width=self.width, threads=4)
        self.addCleanup(table.close)
        few = np.array([3, 699, 3, 0, 17], dtype=np.int64)
        self.assertTrue(np.array_equal(table.gather(few), self.data[few]))
        many = np.random.default_rng(1).integers(0, self.rows, size=SPLIT_AT * 5, dtype=np.int64)
        self.assertTrue(np.array_equal(table.gather(many), self.data[many]))
        self.assertEqual(table.gather(np.zeros(0, dtype=np.int64)).shape, (0, self.width))
        self.assertEqual((table.reads, table.rows_read), (2, few.shape[0] + many.shape[0]))
        with self.assertRaises(IndexError):
            table.gather(np.array([self.rows], dtype=np.int64))
        with self.assertRaises(IndexError):
            table.gather(np.array([-1], dtype=np.int64))

    def test_a_gather_is_one_take_over_the_mapping_not_a_read_a_row(self):
        """The rows come out of a read-only mapping (one C loop, the GIL released): no `pread` a row on either path,
        a gather is a copy that outlives the mapping, and closing releases the view before the descriptor."""
        from engine.modules import lookup_table
        table = lookup_table.MappedTable(self.path, width=self.width, threads=4)
        many = np.random.default_rng(2).integers(0, self.rows, size=lookup_table.SPLIT_AT * 9 + 5, dtype=np.int64)
        with mock.patch.object(lookup_table.os, "pread", side_effect=AssertionError("a row was read by pread"),
                               create=True):
            self.assertTrue(np.array_equal(table.gather(many[:7]), self.data[many[:7]]))
            self.assertTrue(np.array_equal(table.gather(many), self.data[many]))   # the pool's chunks, in order
        got = table.gather(many[:3])
        table.close()
        self.assertTrue(np.array_equal(got, self.data[many[:3]]))
        self.assertEqual((table.fd, table._map, table._rows), (-1, None, None))
        self.path.unlink()                                                         # nothing holds the file

    def test_the_row_count_is_the_file_and_a_partial_row_is_refused(self):
        from engine.modules.lookup_table import MappedTable
        with MappedTable(self.path, width=self.width) as table:
            self.assertEqual((table.rows, table.width), (self.rows, self.width))
        with self.assertRaises(ValueError):                                        # a width the file is not rows of
            MappedTable(self.path, width=self.width * 3)
        with self.assertRaises(ValueError):                                        # a row count that is not the file
            MappedTable(self.path, width=self.width, rows=self.rows - 1)
        with self.assertRaises(ValueError):
            MappedTable(self.path, width=0)


@unittest.skipUnless(torch is not None and np is not None, "requires torch")
class EngramLookupTests(unittest.TestCase):
    """The lookup's rows are the resident form's rows: modules/ngram_embedding.block_fp8_rows over the same bytes."""
    WIDTH, BLOCK, ROWS = 16, 4, 40

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        d = Path(self.dir.name)
        self.path = d / "engram-l1-r0of4.weight"
        self.data = write_table(self.path, self.ROWS, self.WIDTH, seed=4)
        self.scale_bytes = write_scales(d / "engram-l1-r0of4.scale", self.ROWS, self.WIDTH // self.BLOCK, seed=5)
        self.scale = torch.from_numpy(self.scale_bytes.copy()).view(torch.float8_e8m0fnu)
        self.weight = torch.from_numpy(self.data.copy()).view(torch.float8_e4m3fn)

    def lookup(self):
        from engine.modules.lookup_table import SSDEngramLookup
        look = SSDEngramLookup(self.path, self.scale, self.BLOCK, row_bytes=self.WIDTH)
        self.addCleanup(look.close)
        return look

    def test_rows_are_what_the_resident_table_would_have_returned(self):
        from engine.modules.ngram_embedding import block_fp8_rows
        look = self.lookup()
        rows = torch.tensor([[3, 7, 3], [0, self.ROWS - 1, 12]], dtype=torch.int64)
        got = look.rows(rows)
        self.assertEqual((got.dtype, tuple(got.shape)), (torch.bfloat16, (2, 3, self.WIDTH)))
        self.assertTrue(torch.equal(got, block_fp8_rows(self.weight, self.scale, rows, self.BLOCK)))
        self.assertEqual((look.calls, look.rows_read), (1, 5))                  # the repeat was read once
        self.assertEqual(look.n_rows, self.ROWS)

    def test_a_flat_index_and_an_empty_one_keep_their_shape(self):
        from engine.modules.ngram_embedding import block_fp8_rows
        look = self.lookup()
        flat = torch.tensor([9, 9, 9], dtype=torch.int64)
        self.assertTrue(torch.equal(look.rows(flat), block_fp8_rows(self.weight, self.scale, flat, self.BLOCK)))
        self.assertEqual(look.rows_read, 1)
        self.assertEqual(tuple(look.rows(torch.zeros(0, dtype=torch.int64)).shape), (0, self.WIDTH))

    def test_the_reader_needs_nothing_outside_this_tree(self):
        """The regression #1152 left: the engram's reader was imported from overlay/modules/dsv41_engram, which the
        decommission deleted, so the profile's constants and its reader live here now."""
        from engine.modules.lookup_table import SSDEngramLookup
        from engine.profiles.dsv41 import engram
        self.assertFalse((ROOT / "engine/overlay").exists())
        self.assertEqual((engram.ROW_BYTES, engram.SCALE_ROW_BYTES), (256, 8))   # 256 e4m3 elements, 8 e8m0 scales
        d = Path(self.dir.name)
        write_table(d / "wide.weight", 6, engram.ROW_BYTES, seed=6)
        scale = torch.from_numpy(write_scales(d / "wide.scale", 6, engram.SCALE_ROW_BYTES, seed=7).copy())
        look = SSDEngramLookup(d / "wide.weight", scale.view(torch.float8_e8m0fnu), 32, row_bytes=engram.ROW_BYTES)
        self.addCleanup(look.close)
        self.assertEqual((look.n_rows, look.row_bytes), (6, 256))
        self.assertEqual(tuple(look.rows(torch.tensor([5, 0], dtype=torch.int64)).shape), (2, 256))


@unittest.skipUnless(torch is not None and np is not None, "requires torch")
class AttachTests(unittest.TestCase):
    """`engram.attach` swaps the reference's tables for the SSD reader: the layer id comes from the module path, the
    shard's row count must be the module's, and a row of another rank's range is zero."""
    WIDTH, BLOCK, ROWS = 16, 4, 12

    def model(self, rows: int):
        from torch import nn

        class ParallelEngramEmbedding(nn.Module):                                # attach matches on this name
            def __init__(self, width, block, rows_held):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(0, width, dtype=torch.float8_e4m3fn), requires_grad=False)
                self.scale = nn.Parameter(torch.zeros(rows_held, width // block, dtype=torch.float8_e8m0fnu),
                                          requires_grad=False)
                self.block_size, self.part_num_embeddings = block, rows_held
                self.vocab_start_idx, self.vocab_end_idx = 0, rows_held

        layer = nn.Module()
        layer.engram = ParallelEngramEmbedding(self.WIDTH, self.BLOCK, rows)
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Module(), layer])
        return model, layer.engram

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        d = Path(self.dir.name)
        self.data = write_table(d / "engram-l1-r0of4.weight", self.ROWS, self.WIDTH, seed=8)
        self.scale_bytes = write_scales(d / "engram-l1-r0of4.scale", self.ROWS, self.WIDTH // self.BLOCK, seed=9)

    def test_attach_swaps_the_table_and_the_forward_reads_this_ranks_rows(self):
        from engine.modules.ngram_embedding import block_fp8_rows
        from engine.profiles.dsv41 import engram
        model, module = self.model(self.ROWS)
        with mock.patch.object(engram, "ROW_BYTES", self.WIDTH), \
             mock.patch.object(engram, "SCALE_ROW_BYTES", self.WIDTH // self.BLOCK):
            swapped = engram.attach(model, rank=0, world=4, engram_dir=self.dir.name)
        self.assertEqual(swapped, [("layers.1.engram", 1, self.ROWS)])
        self.addCleanup(module.ssd.close)
        ids = torch.tensor([[2, 11], [0, self.ROWS]], dtype=torch.int64)          # the last id is another rank's
        got = module.forward(ids)
        weight = torch.from_numpy(self.data.copy()).view(torch.float8_e4m3fn)
        scale = torch.from_numpy(self.scale_bytes.copy()).view(torch.float8_e8m0fnu)
        want = block_fp8_rows(weight, scale, ids.clamp(max=self.ROWS - 1), self.BLOCK)
        want[1, 1] = 0
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(torch.equal(module.scale, scale))                         # the resident half, read once

    def test_a_shard_that_is_not_the_modules_rows_is_refused(self):
        from engine.profiles.dsv41 import engram
        model, _ = self.model(self.ROWS - 1)
        with mock.patch.object(engram, "ROW_BYTES", self.WIDTH), \
             mock.patch.object(engram, "SCALE_ROW_BYTES", self.WIDTH // self.BLOCK), \
             self.assertRaisesRegex(ValueError, "rows"):
            engram.attach(model, rank=0, world=4, engram_dir=self.dir.name)

    def test_a_missing_shard_names_the_file(self):
        from engine.profiles.dsv41 import engram
        model, _ = self.model(self.ROWS)
        with self.assertRaises(FileNotFoundError):
            engram.attach(model, rank=2, world=4, engram_dir=self.dir.name)


if __name__ == "__main__":
    unittest.main()
