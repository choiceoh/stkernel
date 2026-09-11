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

Both directions stage through one page-aligned host buffer (an anonymous
mmap), sized to `stage_bytes` and reused: the tier holds no memory that the
budget did not declare.

Not in this file, on purpose: WHEN to demote. That is the scheduler's call
(D10: never while a decoder would wait), and the tier only promises that a
demotion or promotion in flight never blocks a step -- it runs on its own
thread and the runner asks `done()`.
"""
from __future__ import annotations

import json
import mmap
import os
import threading
import time
from pathlib import Path

SECTOR = 4096


class NvmeTier:
    def __init__(self, directory: "str | Path", block_bytes: int, stage_bytes: int = 64 << 20):
        if block_bytes % SECTOR:
            raise ValueError(f"block_bytes {block_bytes} must be a multiple of {SECTOR} for O_DIRECT")
        if stage_bytes % block_bytes:
            stage_bytes = (stage_bytes // block_bytes) * block_bytes
        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)
        self.block_bytes, self.stage_bytes = block_bytes, stage_bytes
        self.stage = mmap.mmap(-1, stage_bytes)           # page-aligned, reused
        self.manifest = self.dir / "manifest.json"
        self.index = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        self.lock = threading.Lock()
        self.bytes_written = self.bytes_read = 0

    def _path(self, seq: int) -> Path:
        return self.dir / f"seq-{seq}.kv"

    def has(self, seq: int) -> bool:
        return str(seq) in self.index

    def _save_manifest(self) -> None:
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index)); os.replace(tmp, self.manifest)

    def demote(self, seq: int, blocks: "list", tokens: int) -> int:
        """Write `blocks` (uint8 device views, in order) contiguously. Returns bytes."""
        import torch

        per = self.stage_bytes // self.block_bytes
        fd = os.open(self._path(seq), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        staged = torch.frombuffer(self.stage, dtype=torch.uint8)
        written = 0
        try:
            for i in range(0, len(blocks), per):
                chunk = blocks[i:i + per]
                n = len(chunk) * self.block_bytes
                staged[:n].copy_(torch.cat(chunk) if len(chunk) > 1 else chunk[0])
                off = 0
                while off < n:
                    off += os.pwritev(fd, [memoryview(self.stage)[off:n]], written + off)
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)
        with self.lock:
            self.index[str(seq)] = {"blocks": len(blocks), "tokens": tokens, "bytes": written,
                                    "at": time.time()}
            self._save_manifest()
        self.bytes_written += written
        return written

    def promote(self, seq: int, into: "list") -> int:
        """Read the sequence back into `into` (uint8 device views). Returns bytes."""
        import torch

        meta = self.index[str(seq)]
        if len(into) != meta["blocks"]:
            raise ValueError(f"seq {seq}: {meta['blocks']} blocks on disk, {len(into)} views given")
        per = self.stage_bytes // self.block_bytes
        fd = os.open(self._path(seq), os.O_RDONLY | os.O_DIRECT)
        staged = torch.frombuffer(self.stage, dtype=torch.uint8)
        read = 0
        try:
            for i in range(0, len(into), per):
                chunk = into[i:i + per]
                n = len(chunk) * self.block_bytes
                off = 0
                while off < n:
                    got = os.preadv(fd, [memoryview(self.stage)[off:n]], read + off)
                    if got <= 0:
                        raise OSError(f"short read at {read + off}")
                    off += got
                for j, view in enumerate(chunk):
                    view.copy_(staged[j * self.block_bytes:(j + 1) * self.block_bytes])
                read += n
        finally:
            os.close(fd)
        self.bytes_read += read
        return read

    def forget(self, seq: int) -> None:
        with self.lock:
            self.index.pop(str(seq), None); self._save_manifest()
        try:
            os.unlink(self._path(seq))
        except FileNotFoundError:
            pass

    def run_async(self, fn, *args):
        """A demotion/promotion on its own thread; the runner polls `.done()`."""
        t = threading.Thread(target=fn, args=args, daemon=True); t.start()
        return t


def _selfcheck() -> None:
    import tempfile
    import torch

    GIB = 1 << 30
    block_bytes = 51 * SECTOR                       # 16 tokens x 12.75 KiB (Qwen3.8), sector-rounded
    n_blocks = (1 * GIB) // block_bytes             # ~1 GiB of KV
    src = torch.randint(0, 256, (n_blocks * block_bytes,), dtype=torch.uint8, device="cuda")
    blocks = [src[i * block_bytes:(i + 1) * block_bytes] for i in range(n_blocks)]
    with tempfile.TemporaryDirectory(dir="/home/choiceoh") as d:
        tier = NvmeTier(d, block_bytes)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        wrote = tier.demote(7, blocks, tokens=n_blocks * 16)
        t_w = time.perf_counter() - t0
        dst = torch.zeros_like(src)
        into = [dst[i * block_bytes:(i + 1) * block_bytes] for i in range(n_blocks)]
        t0 = time.perf_counter()
        got = tier.promote(7, into)
        torch.cuda.synchronize(); t_r = time.perf_counter() - t0
        assert wrote == got == n_blocks * block_bytes and torch.equal(src, dst), "round trip must be exact"
        # survives a "boot": a new tier object over the same directory knows the sequence
        tier2 = NvmeTier(d, block_bytes)
        assert tier2.has(7) and tier2.index["7"]["blocks"] == n_blocks
        tier2.forget(7); assert not tier2.has(7)
        print(f"  kv_tier: {wrote / GIB:.2f} GiB demote {t_w:.2f}s ({wrote / GIB / t_w:.2f} GiB/s), "
              f"promote {t_r:.2f}s ({got / GIB / t_r:.2f} GiB/s), exact, manifest survives, O_DIRECT OK")


if __name__ == "__main__":
    _selfcheck()
