"""Rank file alignment and finite I/O failures; CUDA is not required."""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.base.loader import RankLoader

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


def scratch():
    """A writable directory on a filesystem that may support O_DIRECT.

    /tmp can be tmpfs (no O_DIRECT) and the runtime image mounts the repo read-only,
    so try the repo, then the image's writable cache, then whatever tempfile picks;
    the direct-mode assertions skip themselves when none of them has O_DIRECT."""
    for candidate in (os.environ.get("ST_TEST_SCRATCH"), Path(__file__).resolve().parent, "/cache"):
        if candidate and os.access(candidate, os.W_OK):
            return str(candidate)
    return tempfile.gettempdir()


def write_file(path, tensors):
    """A safetensors file whose data offsets are deliberately not sector multiples."""
    header, off = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(raw for _, _, raw in tensors.values()))
    return path


class ReadTests(unittest.TestCase):
    def test_truncated_run_raises_instead_of_spinning_on_eof(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "truncated.safetensors"
            header = json.dumps({"weight": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}).encode()
            p.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
            reader = RankLoader(p)
            reader.direct = False
            with self.assertRaisesRegex(EOFError, "expected 4 more bytes"):
                reader._read_run(reader.runs(["weight"])[0], memoryview(bytearray(8)))

    def test_a_truncated_file_is_short_even_under_direct_reads(self):
        from engine.base.loader import SECTOR, staging
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            p = Path(d) / "truncated.safetensors"
            header = json.dumps({"weight": {"dtype": "U8", "shape": [3 * SECTOR],
                                            "data_offsets": [0, 3 * SECTOR]}}).encode()
            p.write_bytes(struct.pack("<Q", len(header)) + header + b"x" * SECTOR)
            reader = RankLoader(p)
            run = reader.runs(["weight"])[0]
            _owners, views = staging(reader.staging_bytes([run]), 1, "cpu")
            with self.assertRaisesRegex(EOFError, "short read"):
                reader._read_run(run, views[0])

    def test_direct_and_buffered_reads_agree_on_unaligned_offsets(self):
        from engine.base.loader import SECTOR, staging
        tensors = {"a": ("U8", (3,), b"abc"),                       # every later offset is odd
                   "b": ("U8", (5 * SECTOR + 7,), bytes(range(256)) * ((5 * SECTOR + 7) // 256 + 1))}
        tensors["b"] = ("U8", (5 * SECTOR + 7,), tensors["b"][2][: 5 * SECTOR + 7])
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            reader = RankLoader(write_file(Path(d) / "odd.safetensors", tensors))
            if not reader.direct:
                self.skipTest("this filesystem refuses O_DIRECT")
            runs = reader.runs(["a", "b"])
            self.assertGreaterEqual(reader.staging_bytes(runs), runs[0].nbytes + 2 * SECTOR)
            _owners, views = staging(reader.staging_bytes(runs), 1, "cpu")
            direct = [bytes(reader._read_run(r, views[0])) for r in runs]
            reader.direct = False
            buffered = [bytes(reader._read_run(r, views[0])) for r in runs]
            self.assertEqual(direct, buffered)
            self.assertEqual(direct[0][:3], b"abc")

    def test_staging_buffers_are_page_aligned_and_checked_for_size(self):
        from engine.base.loader import SECTOR, staging
        owners, views = staging(SECTOR + 17, 2, "cpu")
        self.assertEqual(len(owners), 2)
        for view in views:
            with memoryview(view) as m:
                self.assertEqual(len(m), SECTOR + 17)
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            reader = RankLoader(write_file(Path(d) / "one.safetensors", {"a": ("U8", (9,), b"123456789")}))
            run = reader.runs(["a"])[0]
            if reader.direct:
                with self.assertRaisesRegex(ValueError, "staging buffer holds"):
                    reader._read_run(run, memoryview(bytearray(16)))


@unittest.skipUnless(torch is not None, "requires PyTorch")
class StagingTests(unittest.TestCase):
    def test_a_cpu_target_does_not_alias_the_reused_staging_buffer(self):
        """Two runs share two buffers; the first run's tensors must keep their bytes."""
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            path = write_file(Path(d) / "two.safetensors",
                              {"first": ("U8", (2048,), b"\x11" * 2048),
                               "second": ("U8", (2048,), b"\x22" * 2048),
                               "third": ("U8", (2048,), b"\x33" * 2048)})
            reader = RankLoader(path)
            self.assertEqual(len(reader.runs(["first", "second", "third"], max_run=2048)), 3)
            # the cap bounds the host buffer, and the buffer is what pins memory while a rank
            # loads: a run can never be narrower than one tensor, so the cap's only job is to
            # stop coalescing from making the pair wider than that (45차 §49)
            wide = reader.runs(["first", "second", "third"], max_run=1 << 30)
            narrow = reader.runs(["first", "second", "third"], max_run=2048)
            self.assertLess(reader.staging_bytes(narrow), reader.staging_bytes(wide))
            self.assertGreaterEqual(reader.staging_bytes(narrow),
                                    max(reader.header[k]["data_offsets"][1] - reader.header[k]["data_offsets"][0]
                                        for k in ("first", "second", "third")))
            out = reader.load(["first", "second", "third"], device="cpu", max_run=2048)
            for name, value in (("first", 0x11), ("second", 0x22), ("third", 0x33)):
                self.assertTrue(bool((out[name] == value).all()), name)
            self.assertEqual(len({t.untyped_storage().data_ptr() for t in out.values()}), 3)

    def test_one_run_is_read_with_one_buffer(self):
        """The second buffer exists to overlap a read with the upload before it. With a single
        run there is nothing to overlap, and it would be half a gigabyte pinned for nothing."""
        from unittest.mock import patch
        import engine.base.loader as loader
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            path = write_file(Path(d) / "one.safetensors", {"only": ("I32", (2,), b"\x01\x00\x00\x00\x02\x00\x00\x00")})
            reader = loader.RankLoader(path)
            counted, real = [], loader.staging

            def counting(nbytes, count, device):
                counted.append(count)
                return real(nbytes, count, device)

            with patch.object(loader, "staging", counting):
                out = reader.load(["only"], device="cpu")
            self.assertEqual(counted, [1])
            self.assertEqual(out["only"].tolist(), [1, 2])

    def test_an_empty_selection_reads_nothing(self):
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            reader = RankLoader(write_file(Path(d) / "one.safetensors", {"a": ("U8", (4,), b"abcd")}))
            self.assertEqual(reader.load([], device="cpu"), {})


@unittest.skipUnless(torch is not None, "requires PyTorch")
class CheckpointTests(unittest.TestCase):
    """The preshard's side: a sharded checkpoint read as ranges, not tensor by tensor."""

    def build(self, root):
        shards = {"shard-0.safetensors": {"model.layers.0.w": ("U8", (600,), b"\x01" * 600),
                                          "model.layers.0.b": ("U8", (7,), b"\x02" * 7),
                                          "model.layers.1.w": ("U8", (600,), b"\x03" * 600)},
                  "shard-1.safetensors": {"model.layers.2.w": ("U8", (600,), b"\x04" * 600),
                                          "model.norm": ("U8", (9,), b"\x05" * 9)}}
        weight_map = {}
        for shard, tensors in shards.items():
            write_file(root / shard, tensors)
            weight_map.update({name: shard for name in tensors})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        return shards

    def test_layers_are_read_as_ranges_and_match_the_reference_reader(self):
        from engine.base.checkpoint import Checkpoint
        from safetensors import safe_open
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            root = Path(d)
            self.build(root)
            ck = Checkpoint(str(root))
            self.assertEqual(ck.num_layers, 3)
            keys = ck.keys_for([0, 2], include_shared=True)
            self.assertEqual(keys, ["model.layers.0.b", "model.layers.0.w", "model.layers.2.w", "model.norm"])
            got = ck.load(keys)
            self.assertEqual(sorted(got), keys)
            for shard in ("shard-0.safetensors", "shard-1.safetensors"):
                with safe_open(root / shard, framework="pt") as f:
                    for name in f.keys():
                        if name in got:
                            self.assertTrue(torch.equal(got[name], f.get_tensor(name)), name)
            # layer 0's two tensors are adjacent: one range covers both
            self.assertEqual(got["model.layers.0.w"].untyped_storage().data_ptr(),
                             got["model.layers.0.b"].untyped_storage().data_ptr())
            self.assertIs(ck.reader("shard-0.safetensors"), ck.reader("shard-0.safetensors"))

    @unittest.skipUnless(Path("/home/choiceoh/models/glm53-redhat-nvfp4/config.json").exists(),
                         "needs the GLM-5.3 checkpoint")
    def test_ranges_hand_the_preshard_the_same_bytes_as_the_per_tensor_reader(self):
        """The gate on changing how the preshard reads: same sources, same built ranks."""
        import os
        from collections import defaultdict
        from safetensors import safe_open
        from engine.base.checkpoint import Checkpoint
        from engine.profiles.glm53 import facts, specs as glm
        ck = Checkpoint("/home/choiceoh/models/glm53-redhat-nvfp4")
        F = facts.load()
        keys = sorted({k for s in glm.layer_specs(F, 0) for k in s.sources})
        by_shard = defaultdict(list)
        for name in keys:
            by_shard[ck.weight_map[name]].append(name)
        old = {}
        for shard, names in sorted(by_shard.items()):
            with safe_open(os.path.join(ck.path, shard), framework="pt", device="cpu") as fh:
                for name in names:
                    old[name] = fh.get_tensor(name)
        new = ck.load(keys)
        self.assertEqual(sorted(new), keys)
        for name in keys:
            self.assertTrue(torch.equal(old[name], new[name]), name)
        for rank in range(facts.TP):
            for spec in glm.layer_specs(F, 0):
                self.assertTrue(torch.equal(spec.build(old, rank, facts.TP), spec.build(new, rank, facts.TP)),
                                f"{spec.name} rank {rank}")

    def test_a_recorder_counts_shards_ranges_and_bytes(self):
        from engine.base.checkpoint import Checkpoint
        from engine.base.instruments import Recorder
        with tempfile.TemporaryDirectory(dir=scratch()) as d:
            root = Path(d)
            self.build(root)
            rec = Recorder("t")
            with rec.phase("load"):
                Checkpoint(str(root)).load(["model.layers.0.w", "model.layers.2.w"], recorder=rec)
            span = rec.root.children[0]
            self.assertEqual(span.counters["shards"], 2)
            self.assertEqual(span.counters["blocks"], 2)
            self.assertEqual(span.counters["bytes"], 1200)
            self.assertEqual(span.counters["tensors"], 2)


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
