"""Rank file alignment and finite I/O failures; CUDA is not required."""
from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.base.loader import RankLoader

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class ReadTests(unittest.TestCase):
    def test_truncated_run_raises_instead_of_spinning_on_eof(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "truncated.safetensors"
            header = json.dumps({"weight": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}).encode()
            p.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
            reader = RankLoader(p)
            with self.assertRaisesRegex(EOFError, "expected 4 more bytes"):
                reader._read_run(reader.runs(["weight"])[0], memoryview(bytearray(8)))


@unittest.skipUnless(torch is not None, "requires PyTorch")
class AlignedRankTests(unittest.TestCase):
    def test_scalar_then_matrix_stays_aligned_in_one_upload(self):
        from engine.base.arena import ALIGN, Arena
        from engine.base.params import Spec
        from engine.base.preshard import RankWriter
        from safetensors import safe_open
        data = {"scale": torch.tensor([1., 2., 3.]),
                "matrix": torch.arange(24 * 32, dtype=torch.float32).view(24, 32),
                "byte": torch.tensor([17], dtype=torch.uint8),
                "half": torch.arange(20, dtype=torch.bfloat16)}
        specs = [Spec(k, tuple(t.shape), t.dtype) for k, t in data.items()]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rank.safetensors"
            writer = RankWriter(p, specs, {"test": "aligned"})
            try:
                for k, t in data.items():
                    writer.put(k, t)
            finally:
                writer.close()
            reader = RankLoader(p)
            runs = reader.runs(data)
            self.assertEqual(len(runs), 1)
            arena = Arena(writer.total, device="cpu")
            loaded = reader.load(data, device="cpu", arena=arena)
            self.assertEqual(len(arena.regions), 1)
            with safe_open(p, framework="pt") as f:
                for k, expected in data.items():
                    self.assertTrue(torch.equal(loaded[k], expected))
                    self.assertTrue(torch.equal(f.get_tensor(k), expected))
                    self.assertEqual((loaded[k].data_ptr() - arena.buf.data_ptr()) % ALIGN, 0)
                    self.assertEqual(loaded[k].data_ptr() % 16, 0)
                    self.assertEqual(loaded[k].untyped_storage().data_ptr(), arena.buf.data_ptr())
            # A run beginning at a later requested tensor must also align.
            later = reader.load(["matrix", "half"], device="cpu")
            self.assertTrue(all((t.data_ptr() - t.untyped_storage().data_ptr()) % ALIGN == 0 for t in later.values()))

    def test_short_writes_are_retried_and_zero_progress_fails(self):
        from engine.base.preshard import RankWriter
        writer = RankWriter.__new__(RankWriter)
        writer.fd = 77
        calls = []
        def partial(fd, view, offset):
            calls.append((bytes(view), offset))
            return min(2, len(view))
        with patch("engine.base.preshard.os.pwrite", side_effect=partial):
            writer._write_all(b"abcde", 10)
        self.assertEqual(calls, [(b"abcde", 10), (b"cde", 12), (b"e", 14)])
        with patch("engine.base.preshard.os.pwrite", return_value=0):
            with self.assertRaisesRegex(OSError, "short write"):
                writer._write_all(b"a", 10)


if __name__ == "__main__":
    unittest.main()
