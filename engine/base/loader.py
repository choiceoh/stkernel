"""Read a presharded rank the way the device can actually deliver it.

Measured 2026-09-11 on srv4, one layer of dsv41's rank0 (606 tensors,
1.839 GiB), against `safetensors.safe_open` + per-tensor `get_tensor`:

    per-tensor (safetensors)            0.16 GiB/s
    one contiguous read + slice + H2D   1.95 GiB/s      12x

and the disk itself does 5.1 GB/s O_DIRECT, so the old path was running at a
thirtieth of the hardware. The reason is not I/O: the layer's tensors are
PERFECTLY contiguous in the file (0.0 MiB of holes across 606 of them), so the
per-tensor path was paying 606 syscalls, 606 allocations and 606 H2D copies for
one sequential range.

That makes the loader's shape obvious and it is the opposite of a tensor loop:

    coalesce  tensors into contiguous byte runs (they already are)
    read      each run in large blocks into a reusable host buffer
    upload    the whole block once, as bytes
    view      every tensor out of the device block -- no per-tensor copy at all

The last step is why this is not just "batched I/O": a tensor is a VIEW into
the uploaded block, so its bytes are never copied twice and the block is the
allocation. That is also what CHARTER D1 wants -- the engine holding one arena
it declared, instead of 26,961 allocations the caching allocator rounds up.

Reading and uploading are balanced (3.61 vs 4.23 GiB/s), so they are overlapped
with two host buffers: block n uploads while block n+1 is read.

The reads are O_DIRECT, for the reason base/kv_tier gives for its own: the page
cache is the same pool as the arena on this box. A buffered load of a 44.5 GiB
rank file leaves 44.5 GiB of clean pages behind at the exact moment the engine
is asking the driver for its arena -- the pressure implicated in the first
45-layer boot's failed 55.4 GiB allocation. Dropping them afterwards (fadvise)
cleans up; not making them is better, and the same NVMe reads at 8.3 GB/s with
O_DIRECT against 3.38 GB/s through the cache (40th ledger). O_DIRECT wants whole
sectors, so a run is read as the sector window around it and the run is the slice
inside; filesystems without it (overlayfs, tmpfs) fall back to buffered reads and
the fadvise that path needs.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

GIB = 1 << 30
SECTOR = 4096                # O_DIRECT's alignment on this fleet (base/kv_tier uses the same)

# safetensors dtype names -> torch. Only what this checkpoint actually holds;
# an unknown name raises rather than guessing a width.
_DTYPES = {
    "U8": "uint8",        # NVFP4 packed weights, two e2m1 per byte (Qwen3.8)
    "I8": "int8",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E8M0": "float8_e8m0fnu",
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "I32": "int32",
    "I64": "int64",
    "BOOL": "bool",
}


def staging(nbytes: int, count: int, device: str):
    """`count` page-aligned host buffers, and what owns them.

    O_DIRECT needs the alignment. Pinned memory has it (cudaHostAlloc is page
    aligned) and also spares the driver its own staging copy on the upload, so a
    CUDA target prefers it; anything else gets anonymous mmaps, which are aligned
    too. The owners must outlive the views.
    """
    if device.startswith("cuda"):
        try:
            import torch
            owners = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(count)]
            return owners, [memoryview(o.numpy()) for o in owners]
        except Exception:                     # noqa: BLE001 -- no CUDA here: an mmap is still aligned
            pass
    owners = [mmap.mmap(-1, nbytes) for _ in range(count)]
    return owners, [memoryview(o) for o in owners]


@dataclass(frozen=True)
class Run:
    """A contiguous byte range of the file and the tensors inside it."""
    start: int          # absolute file offset
    end: int
    keys: tuple         # (name, offset_within_run, nbytes)

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class RankLoader:
    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            self.header = json.loads(handle.read(size))
        self.metadata = self.header.pop("__metadata__", None)
        self.data_base = 8 + size
        self.direct = self._direct_works()

    def _direct_works(self) -> bool:
        """Ask the filesystem once: overlayfs and tmpfs refuse O_DIRECT."""
        try:
            os.close(os.open(self.path, os.O_RDONLY | os.O_DIRECT))
            return True
        except OSError:
            return False

    def staging_bytes(self, runs) -> int:
        """What one host buffer must hold: the widest run plus the sectors around it."""
        return max(r.nbytes for r in runs) + (2 * SECTOR if self.direct else 0)

    def keys(self) -> "list[str]":
        return list(self.header)

    def nbytes(self, keys) -> int:
        return sum(self.header[k]["data_offsets"][1] - self.header[k]["data_offsets"][0]
                   for k in keys)

    def runs(self, keys, max_gap: int = 1 << 20, max_run: int = 1 << 30) -> "list[Run]":
        """Coalesce `keys` into contiguous ranges.

        `max_gap` tolerates a hole rather than splitting the read -- reading a
        megabyte we discard is cheaper than a second seek. `max_run` bounds the
        host buffer, which is what makes this streamable instead of a slurp.
        """
        ordered = sorted(keys, key=lambda k: self.header[k]["data_offsets"][0])
        out, start, end, members = [], None, None, []
        for name in ordered:
            lo, hi = self.header[name]["data_offsets"]
            if start is None:
                start, end, members = lo, hi, []
            elif lo - end > max_gap or hi - start > max_run:
                out.append(Run(start, end, tuple(members)))
                start, end, members = lo, hi, []
            members.append((name, lo - start, hi - lo))
            end = max(end, hi)
        if start is not None:
            out.append(Run(start, end, tuple(members)))
        return out

    # -- the read half -------------------------------------------------------

    def _read_run(self, run: Run, into: memoryview) -> memoryview:
        """The run's bytes, as a slice of `into` (size it with `staging_bytes`).

        Under O_DIRECT the read is the sector window that contains the run: the
        offset, the buffer and every request length stay sector-aligned, and the
        leading bytes of the first sector are read and ignored. A short read can
        only happen past the file's end, which is past the run by construction --
        so it is a truncated file, not a partial sector.
        """
        start = self.data_base + run.start
        lead = start % SECTOR if self.direct else 0
        base, want = start - lead, lead + run.nbytes
        span = -(-want // SECTOR) * SECTOR if self.direct else want
        if len(into) < span:
            raise ValueError(f"staging buffer holds {len(into)} B, this run needs {span}")
        fd = os.open(self.path, os.O_RDONLY | (os.O_DIRECT if self.direct else 0))
        try:
            view, got = into[:span], 0
            while got < want:
                n = os.preadv(fd, [view[got:]], base + got)
                if n <= 0:
                    raise EOFError(f"{self.path}: short read at {base + got}; expected {want - got} more bytes")
                got += n
        finally:
            os.close(fd)
        return into[lead:lead + run.nbytes]

    # -- the whole thing -----------------------------------------------------

    def load(self, keys, device: str = "cuda", max_run: int = 1 << 30,
             recorder=None, arena=None) -> dict:
        """{name: tensor} with every tensor a view into an uploaded block.

        With `arena`, the block is carved from it and filled with one H2D
        copy -- the engine then holds NO allocation the arena does not know
        about (D16). Without it, the block is its own allocation.
        """
        import torch

        runs = self.runs(keys, max_run=max_run)
        if not runs:
            return {}
        owners, buffers = staging(self.staging_bytes(runs), 2, device)
        out, blocks, staged = {}, [], None

        def read(index):
            return self._read_run(runs[index], buffers[index % 2])

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(read, 0)
                for i, run in enumerate(runs):
                    host = pending.result()
                    if i + 1 < len(runs):
                        # the next read cannot reuse this buffer, hence two.
                        pending = pool.submit(read, i + 1)
                    staged = torch.frombuffer(host, dtype=torch.uint8)
                    if arena is not None:
                        block = arena.carve(run.nbytes, f"weights/{run.start}")
                        block.copy_(staged, non_blocking=False)
                    else:
                        # always a copy: the next run overwrites this staging buffer
                        block = staged.to(device, non_blocking=False, copy=True)
                    if not self.direct:
                        # A buffered read left clean pages behind and a blocking
                        # upload has consumed this run. They must not compete
                        # with the resident arena on UMA.
                        with self.path.open("rb") as stream:
                            os.posix_fadvise(stream.fileno(), self.data_base + run.start,
                                             run.nbytes, os.POSIX_FADV_DONTNEED)
                    blocks.append(block)
                    for name, offset, size in run.keys:
                        entry = self.header[name]
                        dtype = getattr(torch, _DTYPES[entry["dtype"]])
                        out[name] = (block[offset:offset + size]
                                     .view(dtype).reshape(entry["shape"]))
                    if recorder is not None:
                        recorder.count("blocks")
                        recorder.count("bytes", run.nbytes)
        finally:
            staged = None                     # the staging buffers are no one else's
            buffers.clear()
            owners.clear()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        # the caller must keep `blocks` alive; views reference them, and Python
        # does that for us through the tensors' storage.
        return out


def _main(argv=None) -> int:
    import argparse
    import re
    import time

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank-file",
                        default="/home/choiceoh/models/DeepSeek-V4.1-Flash-tp4/rank0of4.safetensors")
    parser.add_argument("--layers", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-run-mib", type=int, default=1024)
    parser.add_argument("--buffered", action="store_true", help="read through the page cache instead of O_DIRECT")
    args = parser.parse_args(argv)

    loader = RankLoader(args.rank_file)
    if args.buffered:
        loader.direct = False
    if "-" in args.layers:
        lo, hi = args.layers.split("-")
        want = set(range(int(lo), int(hi) + 1))
    else:
        want = {int(x) for x in args.layers.split(",") if x}
    # dsv41 rank files name a layer `layers.N.`, GLM's specs name it `LN.`
    layer = re.compile(r"(?:^|\.)(?:layers\.|L)(\d+)\.")
    keys = [k for k in loader.keys() if (m := layer.search(k)) and int(m.group(1)) in want]
    if not keys:
        raise SystemExit(f"no tensors for layers {sorted(want)} in {args.rank_file}")

    runs = loader.runs(keys, max_run=args.max_run_mib << 20)
    total = loader.nbytes(keys)
    holes = sum(r.nbytes for r in runs) - total
    started = time.perf_counter()
    tensors = loader.load(keys, device=args.device, max_run=args.max_run_mib << 20)
    elapsed = time.perf_counter() - started

    print(f"  {len(keys):,} tensors, {total / GIB:.3f} GiB, {len(runs)} runs, "
          f"{holes / (1 << 20):.1f} MiB of holes, reads {'O_DIRECT' if loader.direct else 'buffered'}")
    print(f"  {elapsed:.2f} s -> {total / GIB / elapsed:.2f} GiB/s to {args.device}")
    import torch
    if args.device.startswith("cuda"):
        print(f"  torch allocated {torch.cuda.memory_allocated() / GIB:.3f} GiB, "
              f"reserved {torch.cuda.memory_reserved() / GIB:.3f} GiB "
              f"(slack {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / GIB:.3f})")
    sample = next(iter(tensors))
    print(f"  e.g. {sample}: {tuple(tensors[sample].shape)} {tensors[sample].dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
