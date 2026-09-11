#!/usr/bin/env python3
"""Does the O_DIRECT row reader survive a row width that straddles sectors?

DeepSeek-V4.1's engram rows are 256 bytes. 256 divides the 512-byte logical
block, so a row lies entirely inside one sector and one aligned sector read per
row is both correct and minimal. The reader was written to that.

Qwen3.8-Flash-Next's PLE n-gram rows are **160 bytes** (`F8_E4M3 [2500012,
160]`, 128 shards). 160 does not divide 512, so rows DO straddle:

    row 3 lives at 480..640 -- sectors 0 and 1

A reader that assumes otherwise reads sector 0, takes 160 bytes from offset
480, and gets 32 real bytes followed by 128 bytes of whatever the reused
buffer last held. Real data, right file, wrong place, no error.

This builds a shard at a given row width, fills each row with a pattern that
identifies it, and requires every row read back to be exactly its own. Then it
checks the guard: a `read_bytes` too small for the width must be REFUSED, not
silently truncated.

    python3 probes/qwen38_ple_rows.py [--row-bytes 160] [--rows 20000]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "overlay", "modules", "dsv41_engram"))

from dsv41_engram_io import ShardReader, min_read_bytes  # noqa: E402


def row_pattern(row: int, width: int) -> bytes:
    """Unique per row and per width, so a row from the wrong offset shows."""
    seed = hashlib.sha256(f"{row}:{width}".encode()).digest()
    return (seed * ((width // len(seed)) + 1))[:width]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-bytes", type=int, default=160,
                    help="160 = Qwen PLE, 256 = DeepSeek engram")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--queue-depth", type=int, default=16)
    args = ap.parse_args()
    width, n = args.row_bytes, args.rows
    ok = True

    straddle = [r for r in range(min(n, 512))
                if (r * width) % 512 + width > 512]
    need = min_read_bytes(width)
    print(f"  row_bytes {width}: {'straddles' if straddle else 'never straddles'}"
          f" sectors (e.g. rows {straddle[:4]}), minimum read {need} B")

    with tempfile.TemporaryDirectory(prefix="ple-rows-") as tmp:
        path = os.path.join(tmp, "shard.bin")
        with open(path, "wb") as fh:
            for r in range(n):
                fh.write(row_pattern(r, width))
        size = os.path.getsize(path)
        print(f"  wrote {n:,} rows, {size:,} B")

        with ShardReader(path, queue_depth=args.queue_depth,
                         row_bytes=width) as reader:
            if reader.n_rows != n:
                print(f"  FAIL: reader sees {reader.n_rows} rows, wrote {n}")
                return 1
            print(f"  reader     n_rows {reader.n_rows:,}, read_bytes "
                  f"{reader.read_bytes}")
            # every straddling row, plus a spread of the rest
            want = sorted(set(straddle[:64]
                              + list(range(0, n, max(1, n // 512)))
                              + [0, 1, n - 2, n - 1]))
            got = reader.gather(want)
            bad = [r for r, g in zip(want, got) if g != row_pattern(r, width)]
            if bad:
                r = bad[0]
                print(f"  FAIL: {len(bad)} of {len(want)} rows wrong; row {r} "
                      f"at offset {r * width} returned "
                      f"{got[want.index(r)][:8].hex()} want "
                      f"{row_pattern(r, width)[:8].hex()}")
                return 1
            print(f"  content    {len(want)} rows exact, including "
                  f"{len([r for r in want if r in set(straddle)])} that "
                  f"straddle a sector")

        # the guard: a window too small must be refused, not truncated
        if need > 512:
            try:
                ShardReader(path, row_bytes=width, read_bytes=512).close()
                print("  FAIL: read_bytes=512 accepted for a straddling width")
                ok = False
            except ValueError as exc:
                print(f"  refuses    read_bytes below the minimum "
                      f"({str(exc).split('.')[0][:56]})")
        try:
            ShardReader(path, row_bytes=width, read_bytes=need + 1).close()
            print("  FAIL: a non-block-multiple read_bytes was accepted")
            ok = False
        except ValueError:
            print("  refuses    a read_bytes that is not a block multiple")

    print("\n" + ("PLE ROWS PASS" if ok else "PLE ROWS FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
