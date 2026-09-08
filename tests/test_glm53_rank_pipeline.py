"""Real file/hash checks with deliberately deferred DMA to expose slot reuse races."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from test_glm53_startup_artifacts import import_file, torch


class DeferredStream:
    def __init__(self):
        self.jobs = []
        self.completed = 0
        self.drains = 0

    def complete(self, count):
        while self.completed < count:
            dst, src = self.jobs[self.completed]
            dst.copy_(src)  # reads the original buffer NOW, not at enqueue time
            self.completed += 1

    def synchronize(self):
        self.drains += 1
        self.complete(len(self.jobs))


class DeferredEvent:
    def record(self, stream):
        self.stream, self.until = stream, len(stream.jobs)

    def synchronize(self):
        self.stream.complete(self.until)


class DeferredTarget:
    def __init__(self, value, stream):
        self.value, self.stream = value, stream
        self.device = torch.device("cuda:0")

    def numel(self):
        return self.value.numel()

    def reshape(self, *shape):
        return DeferredTarget(self.value.reshape(*shape), self.stream)

    def view(self, dtype):
        return DeferredTarget(self.value.view(dtype), self.stream)

    def __getitem__(self, key):
        return DeferredTarget(self.value[key], self.stream)

    def copy_(self, source, non_blocking=False):
        assert non_blocking
        self.stream.jobs.append((self.value, source))


@unittest.skipIf(torch is None, "CPU torch required")
class RankPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        common = import_file("rank_pipeline_common", "glm53_startup_cache.py")
        self.patch_modules = patch.dict("sys.modules", {
            "vllm.model_executor.layers.glm53_startup_cache": common})
        self.patch_modules.start()
        self.addCleanup(self.patch_modules.stop)
        self.rank = import_file("rank_pipeline_test", "glm53_rank_cache.py")
        self.rank.CHUNK_BYTES = 16
        self.expected = torch.arange(7 * 16 + 9, dtype=torch.uint8)
        self.path = self.root / "weights.bin"
        self.path.write_bytes(bytes(self.expected.tolist()))
        chunks = []
        for start in range(0, self.expected.numel(), 16):
            data = bytes(self.expected[start:start+16].tolist())
            chunks.append(dict(name="weight", start=start, offset=start, size=len(data),
                               sha256=hashlib.sha256(data).hexdigest()))
        self.manifest = dict(size=self.expected.numel(), chunks=chunks, loaded=["weight"])
        self.actual = torch.full_like(self.expected, 255)
        self.stream = DeferredStream()
        self.state = {"weight": DeferredTarget(self.actual, self.stream)}

    def restore(self):
        empty = torch.empty
        def host_empty(*args, **kwargs):
            self.assertTrue(kwargs.pop("pin_memory"))
            return empty(*args, **kwargs)
        with patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE_PIPELINE": "1"}), \
             patch.object(torch, "empty", side_effect=host_empty), \
             patch.object(torch.cuda, "current_stream", return_value=self.stream), \
             patch.object(torch.cuda, "Event", DeferredEvent):
            return self.rank._restore(self.root, self.manifest, self.state)

    def test_two_slots_preserve_bytes_with_delayed_dma_and_tail(self):
        self.assertEqual(self.restore(), {"weight"})
        self.assertTrue(torch.equal(self.actual, self.expected))
        self.assertEqual(self.stream.completed, 8)
        self.assertEqual(self.stream.drains, 1)
        # Two storages only, regardless of the number of chunks.
        self.assertEqual(len({src.untyped_storage().data_ptr() for _, src in self.stream.jobs}), 2)

    def test_late_bad_checksum_never_publishes_bad_chunk_and_drains_dma(self):
        data = bytearray(self.path.read_bytes()); data[49] ^= 1; self.path.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.restore()
        self.assertTrue(torch.equal(self.actual[:48], self.expected[:48]))
        self.assertTrue(torch.all(self.actual[48:] == 255))
        self.assertEqual(self.stream.completed, len(self.stream.jobs))
        self.assertEqual(self.stream.drains, 1)

    def test_late_truncation_is_fatal_and_drains_earlier_copies(self):
        self.path.write_bytes(self.path.read_bytes()[:50])
        with self.assertRaisesRegex(RuntimeError, "short rank-cache read"):
            self.restore()
        self.assertTrue(torch.equal(self.actual[:48], self.expected[:48]))
        self.assertTrue(torch.all(self.actual[48:] == 255))
        self.assertEqual(self.stream.completed, len(self.stream.jobs))

    def test_partial_readinto_retries_until_full_chunk(self):
        class ShortReader:
            def __init__(self, stream): self.stream = stream
            def seek(self, *args): return self.stream.seek(*args)
            def fileno(self): return self.stream.fileno()
            def readinto(self, view): return self.stream.readinto(view[:3])
        chunk = self.manifest["chunks"][2]
        slot = torch.empty(16, dtype=torch.uint8)
        with self.path.open("rb", buffering=0) as source:
            self.rank._read_into_slot(ShortReader(source), chunk, slot)
        self.assertTrue(torch.equal(slot, self.expected[32:48]))

    def test_reader_failure_keeps_buffer_alive_until_prior_dma_drained(self):
        original = self.rank._read_into_slot
        def fail(source, chunk, buffer):
            if chunk["start"] == 16: raise OSError("injected disk failure")
            return original(source, chunk, buffer)
        with patch.object(self.rank, "_read_into_slot", side_effect=fail):
            with self.assertRaisesRegex(OSError, "injected disk failure"):
                self.restore()
        self.assertTrue(torch.equal(self.actual[:16], self.expected[:16]))
        self.assertEqual(self.stream.completed, len(self.stream.jobs))

    def test_pin_allocation_failure_falls_back_before_any_write(self):
        with patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE_PIPELINE": "1"}), \
             patch.object(torch, "empty", side_effect=RuntimeError("pin unavailable")), \
             patch.object(self.rank, "_restore_serial") as serial, \
             patch.object(self.rank, "_restore_pipeline") as pipeline:
            self.rank._restore(self.root, self.manifest, self.state)
        serial.assert_called_once(); pipeline.assert_not_called()
        self.assertEqual(self.stream.jobs, [])

    def test_cpu_targets_keep_exact_serial_path(self):
        with patch.dict(os.environ, {"VLLM_GLM53_RANK_CACHE_PIPELINE": "1"}), \
             patch.object(self.rank, "_restore_pipeline", side_effect=AssertionError("CUDA path")):
            self.rank._restore(self.root, self.manifest, {"weight": self.actual})
        self.assertTrue(torch.equal(self.actual, self.expected))


if __name__ == "__main__":
    unittest.main()
