"""The NVMe tier for cold KV (base). Contiguous per sequence, O_DIRECT, survives boots.

D16's second layer. A sequence that goes idle has its blocks written as ONE
contiguous run to a file named for it; a sequence that comes back has them
read straight into freshly reserved blocks. Two properties are load-bearing:

  contiguous     the drive does 5.1 GB/s sequential and ~0.6 GB/s random
                 (157K IOPS x 4 KiB). A 128K-token sequence is 1.59 GiB
                 (Qwen3.8); contiguous is 0.3 s, scattered is 2.7 s.
  O_DIRECT       the page cache is the same pool as the arena on this box.
                 Reading 1.6 GiB through it would evict something that
                 mattered; direct I/O touches only the staging buffer.

Both directions stage through one PINNED, page-aligned host buffer sized to
`stage_bytes` and reused, and every device-side move happens on the tier's
OWN CUDA stream: one gather kernel per staging window (index_select over the
block storage viewed as [num_blocks, block_bytes]) and one async copy. The
first version did per-block Python `copy_` calls on the default stream and
slowed a decode-shaped loop by 1.50x (42nd ledger) -- thousands of launches
competing for the GIL and the stream a decoder was using. The tier holds no
memory the budget did not declare: the staging buffer and the device scratch
are both fixed at construction.

Not in this file, on purpose: WHEN to demote. That is the scheduler's call
(D10: never while a decoder would wait), and the tier only promises that a
demotion or promotion in flight never blocks a step -- it runs on its own
thread and the runner asks `done()`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path

SECTOR = 4096


class NvmeTier:
    def __init__(self, directory: "str | Path", block_bytes: int, stage_bytes: int = 64 << 20):
        if not isinstance(block_bytes, int) or block_bytes <= 0 or block_bytes % SECTOR:
            raise ValueError(f"block_bytes {block_bytes} must be a positive multiple of {SECTOR} for O_DIRECT")
        if not isinstance(stage_bytes, int) or stage_bytes < block_bytes:
            raise ValueError("stage_bytes must hold at least one complete block")
        if stage_bytes % block_bytes:
            stage_bytes = (stage_bytes // block_bytes) * block_bytes
        import torch

        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)
        self.block_bytes, self.stage_bytes = block_bytes, stage_bytes
        self.per = stage_bytes // block_bytes
        # pinned host staging (cudaHostAlloc is page-aligned, which O_DIRECT needs)
        self.stage_t = torch.empty(stage_bytes, dtype=torch.uint8, pin_memory=True)
        self.stage = memoryview(self.stage_t.numpy())
        # device scratch for one window's gather, and the tier's own stream
        self.scratch = torch.empty(stage_bytes, dtype=torch.uint8, device="cuda")
        self.stream = torch.cuda.Stream()
        self.manifest = self.dir / "manifest.json"
        self.index = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        self.lock = threading.Lock()
        self._transfer_lock = threading.Lock()    # one staging buffer/stream, even across async callers
        self.bytes_written = self.bytes_read = 0

    def _path(self, seq: int) -> Path:
        return self.dir / f"seq-{seq}.kv"

    def has(self, seq: int) -> bool:
        return str(seq) in self.index

    def _save_manifest(self, index: dict) -> None:
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(index)); os.replace(tmp, self.manifest)
        self.index = index                       # publish only after the disk commit succeeds

    def demote(self, seq: int, storage, block_ids: "list[int]", tokens: int) -> int:
        """Write blocks `block_ids` of `storage` ([num_blocks * block_bytes] uint8)
        contiguously. Returns bytes. Device work runs on the tier stream only."""
        with self._transfer_lock:
            return self._demote(seq, storage, block_ids, tokens)

    def _demote(self, seq: int, storage, block_ids: "list[int]", tokens: int) -> int:
        import torch

        table = storage.view(-1, self.block_bytes)
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=table.device)
        self.stream.wait_stream(torch.cuda.current_stream(table.device))
        fd = os.open(self._path(seq), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        written = 0
        try:
            for i in range(0, len(block_ids), self.per):
                n_blk = min(self.per, len(block_ids) - i); n = n_blk * self.block_bytes
                with torch.cuda.stream(self.stream):
                    torch.index_select(table, 0, ids[i:i + n_blk],
                                       out=self.scratch[:n].view(n_blk, self.block_bytes))
                    self.stage_t[:n].copy_(self.scratch[:n], non_blocking=True)
                self.stream.synchronize()
                off = 0
                while off < n:
                    got = os.pwritev(fd, [self.stage[off:n]], written + off)
                    if got <= 0:
                        raise OSError(f"short write at {written + off}")
                    off += got
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)
        with self.lock:
            self._save_manifest({**self.index, str(seq): {
                "blocks": len(block_ids), "tokens": tokens, "bytes": written, "at": time.time()}})
        self.bytes_written += written
        return written

    def promote(self, seq: int, storage, block_ids: "list[int]") -> int:
        """Read the sequence back into blocks `block_ids` of `storage`. Returns bytes."""
        with self._transfer_lock:
            return self._promote(seq, storage, block_ids)

    def _promote(self, seq: int, storage, block_ids: "list[int]") -> int:
        import torch

        meta = self.index[str(seq)]
        if len(block_ids) != meta["blocks"]:
            raise ValueError(f"seq {seq}: {meta['blocks']} blocks on disk, {len(block_ids)} given")
        table = storage.view(-1, self.block_bytes)
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=table.device)
        self.stream.wait_stream(torch.cuda.current_stream(table.device))
        fd = os.open(self._path(seq), os.O_RDONLY | os.O_DIRECT)
        read = 0
        try:
            for i in range(0, len(block_ids), self.per):
                n_blk = min(self.per, len(block_ids) - i); n = n_blk * self.block_bytes
                off = 0
                while off < n:
                    got = os.preadv(fd, [self.stage[off:n]], read + off)
                    if got <= 0:
                        raise OSError(f"short read at {read + off}")
                    off += got
                with torch.cuda.stream(self.stream):
                    self.scratch[:n].copy_(self.stage_t[:n], non_blocking=True)
                    table.index_copy_(0, ids[i:i + n_blk], self.scratch[:n].view(n_blk, self.block_bytes))
                self.stream.synchronize()
                read += n
        finally:
            os.close(fd)
        self.bytes_read += read
        return read

    def forget(self, seq: int) -> None:
        with self._transfer_lock:
            with self.lock:
                index = dict(self.index)
                index.pop(str(seq), None)
                self._save_manifest(index)
            try:
                os.unlink(self._path(seq))
            except FileNotFoundError:
                pass

    def run_async(self, fn, *args) -> Future:
        """Off-thread I/O. Poll `.done()`, then `.result()` to surface failures.

        Transfers on this tier serialize on its one staging buffer. Running
        decoders do not acquire that lock or wait on these futures.
        Preserve the caller's CUDA stream in the worker so a transfer waits
        for its producer even when submission came from a nondefault stream.
        """
        future = Future()
        context = nullcontext
        if hasattr(self, "stream"):
            import torch
            producer = torch.cuda.current_stream(self.stream.device)
            context = lambda: torch.cuda.stream(producer)

        def work():
            if not future.set_running_or_notify_cancel():
                return
            try:
                with context():
                    result = fn(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        threading.Thread(target=work, daemon=True).start()
        return future


def _selfcheck() -> None:
    import tempfile
    import torch

    GIB = 1 << 30
    block_bytes = 51 * SECTOR
    n_blocks = (1 * GIB) // block_bytes
    storage = torch.randint(0, 256, (n_blocks * block_bytes,), dtype=torch.uint8, device="cuda")
    keep = storage.clone()
    ids = list(range(n_blocks))[::-1]               # scattered order: the gather must honour it
    with tempfile.TemporaryDirectory(dir="/home/choiceoh") as d:
        tier = NvmeTier(d, block_bytes)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        wrote = tier.demote(7, storage, ids, tokens=n_blocks * 16); t_w = time.perf_counter() - t0
        storage.zero_()
        t0 = time.perf_counter(); got = tier.promote(7, storage, ids); torch.cuda.synchronize(); t_r = time.perf_counter() - t0
        assert wrote == got == n_blocks * block_bytes and torch.equal(storage, keep), "round trip must be exact"
        tier2 = NvmeTier(d, block_bytes); assert tier2.has(7); tier2.forget(7); assert not tier2.has(7)
        print(f"  kv_tier v2: {wrote / GIB:.2f} GiB demote {t_w:.2f}s ({wrote / GIB / t_w:.2f} GiB/s), "
              f"promote {t_r:.2f}s ({got / GIB / t_r:.2f} GiB/s), scattered ids exact, own stream + pinned staging OK")


if __name__ == "__main__":
    _selfcheck()
