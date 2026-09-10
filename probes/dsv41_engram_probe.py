#!/usr/bin/env python3
"""Does a rank's engram fetch fit inside a decode step? Answered without a GPU.

Builds a synthetic shard, proves ShardReader returns the rows it was asked for,
then times the real per-step access pattern. The pattern is not invented: it
falls out of the checkpoint's config.

    engram_max_ngram_size 4   -> 3 n-gram sizes (2-gram, 3-gram, 4-gram)
    engram_n_heads        8
    engram_layer_ids  [1, 14] -> 2 tables
    3 * 8 * 2 = 48 rows per token, 12 per rank once the 24 disjoint prime
    bucket ranges are split 6-per-rank across TP=4

A batch-32 decode step therefore asks one rank for 32 * 12 = 384 rows. The
verdict is that number against the step budget, so the probe prints the fetch as
a percentage of it rather than as bare IOPS -- IOPS is what the drive can do,
and the question is what the step can afford.

    python3 probes/dsv41_engram_probe.py [--shard-gib 2] [--step-ms 20]

Nothing here touches the model, the fleet or a serving container, so it runs on
any node at any time. It DOES do sustained O_DIRECT reads, which share the drive
with whatever else is on it -- a run taken while the checkpoint is downloading
measures the contended case, which is the pessimistic one and worth saying in
the log rather than hiding.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "overlay", "modules", "dsv41_engram"))

from dsv41_engram_io import EMB_ROW_BYTES, ShardReader  # noqa: E402

ROWS_PER_TOKEN_PER_RANK = 12          # 3 n-gram sizes x 8 heads x 2 tables / 4 ranks
TABLE_ROWS = 384_006_168              # engram_num_embeddings[0]


def build_shard(path: str, n_rows: int) -> None:
    """Row i is filled with the low byte of i, so a wrong row is detectable."""
    chunk_rows = 1 << 14
    with open(path, "wb") as fh:
        written = 0
        while written < n_rows:
            rows = min(chunk_rows, n_rows - written)
            fh.write(b"".join(bytes([(written + i) & 0xFF]) * EMB_ROW_BYTES
                              for i in range(rows)))
            written += rows
    os.sync()


def check_correctness(reader: ShardReader, rng: random.Random) -> None:
    # Deliberately includes adjacent rows (which share a 512 B sector and so
    # exercise the coalescing path) and a repeat (which must be read once and
    # returned twice, in order).
    probe = [0, 1, 2, 3, reader.n_rows - 1, reader.n_rows - 2, 7, 7]
    probe += [rng.randrange(reader.n_rows) for _ in range(64)]
    got = reader.gather(probe)
    assert len(got) == len(probe), f"{len(got)} rows back for {len(probe)} asked"
    for row, data in zip(probe, got):
        want = bytes([row & 0xFF]) * EMB_ROW_BYTES
        assert data == want, f"row {row}: got {data[:8]!r}, want {want[:8]!r}"
    print(f"  correctness ... OK ({len(probe)} rows, incl. adjacent + repeat)")


def make_plans(reader: ShardReader, rng: random.Random, batch: int,
               steps: int) -> list[list[int]]:
    per_step = batch * ROWS_PER_TOKEN_PER_RANK
    return [[rng.randrange(reader.n_rows) for _ in range(per_step)]
            for _ in range(steps)]


def time_blocking(reader: ShardReader, plans) -> list[float]:
    """Submit and wait with nothing in between: the fetch's own cost."""
    reader.gather(plans[0])                      # warm the pool's threads/fds
    out = []
    for plan in plans:
        t0 = time.perf_counter()
        reader.gather(plan)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def time_overlapped(reader: ShardReader, plans, cover_ms: float) -> list[float]:
    """Residual stall after `cover_ms` of compute -- the number that decides it.

    The addresses are known at step entry, so the submit goes at layer 0 and the
    wait at the engram layer. What the step actually pays is whatever is LEFT
    when the compute in between is done, and for a cover long enough that is
    zero rather than "small".

    `sleep` stands in for the compute because that is what the host does during
    it: the work is on the GPU and the Python thread is not holding the GIL,
    which is the same condition the reader threads see in serving.
    """
    reader.gather(plans[0])
    out = []
    for plan in plans:
        t0 = time.perf_counter()
        pending = reader.submit(plan)
        time.sleep(cover_ms / 1e3)
        t1 = time.perf_counter()
        pending.wait()
        out.append((time.perf_counter() - t1) * 1e3)
        del t0
    return out


def time_pipelined(reader: ShardReader, plans, cover_ms: float) -> list[float]:
    """Steady state: step N+1 is issued before step N is collected.

    Serving does not pause between steps, so the drive is never idle and a
    submit lands on a queue that still holds the previous step's tail. Measuring
    only the quiesced case would flatter the design.
    """
    reader.gather(plans[0])
    out, inflight = [], reader.submit(plans[0])
    for plan in plans[1:]:
        time.sleep(cover_ms / 1e3)
        nxt = reader.submit(plan)          # issue N+1 before collecting N
        t0 = time.perf_counter()
        inflight.wait()
        out.append((time.perf_counter() - t0) * 1e3)
        inflight = nxt
    inflight.wait()
    return out


def time_drafted(reader: ShardReader, plans, step_ms: float,
                 accept: float, rng: random.Random) -> "tuple[list[float], float]":
    """Layer 1 fetched a whole step early, off the DSpark draft block.

    Layer 1 sits one layer into a forty-layer step, so nothing inside the step
    can cover it. The rows are addressable a step early anyway: DSpark drafts
    `dspark_block_size` 5 tokens, and a draft token's hash ids are computable
    from the draft ids alone. So step N issues step N+1's layer-1 rows alongside
    its own layer-14 rows, and step N+1 finds them already there.

    The cost is not latency, it is waste: a rejected draft token was fetched for
    nothing. `accept` is the acceptance rate, and the returned second value is
    the fraction of reads thrown away -- reads the drive still had to do.
    """
    per_token = ROWS_PER_TOKEN_PER_RANK // 2      # half the rows are layer 1's
    reader.gather(plans[0])
    ahead = reader.submit(plans[0][:len(plans[0]) // 2])
    out, issued, wasted = [], 0, 0
    for i, plan in enumerate(plans[1:], 1):
        half = len(plan) // 2
        l14, l1_next = plan[:half], plans[(i + 1) % len(plans)][half:]
        pending14 = reader.submit(l14)
        # layer 1: submitted during the previous step, so a full step of cover
        t0 = time.perf_counter()
        ahead.wait()
        stall = (time.perf_counter() - t0) * 1e3
        # a rejected draft's rows were fetched and are discarded
        missed = sum(1 for _ in range(len(l1_next) // per_token)
                     if rng.random() >= accept)
        issued += len(l1_next)
        wasted += missed * per_token
        ahead = reader.submit(l1_next)
        time.sleep(14 / 40 * step_ms / 1e3)
        t1 = time.perf_counter()
        pending14.wait()
        out.append(stall + (time.perf_counter() - t1) * 1e3)
    ahead.wait()
    return out, (wasted / issued if issued else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-gib", type=float, default=2.0)
    ap.add_argument("--step-ms", type=float, default=20.0,
                    help="decode step budget to judge the fetch against")
    ap.add_argument("--queue-depth", type=int, default=32)
    ap.add_argument("--read-bytes", type=int, default=512,
                    help="512 = one sector per row (2x amplification); "
                         "4096 = one page (16x)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--accept", type=float, default=0.75,
                    help="DSpark draft acceptance; sets how many layer-1 "
                         "prefetches are thrown away")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--real", action="store_true",
                    help="the path is a REAL shard, not this probe's synthetic "
                         "one: skip the pattern check (its rows are weights, "
                         "not row indices) and never delete the file. "
                         "Correctness for a real shard is "
                         "tools/dsv41_engram_shard.py verify, which compares "
                         "it against the source checkpoint")
    ap.add_argument("--path", default="/tmp/dsv41-engram-probe.bin")
    args = ap.parse_args()

    n_rows = int(args.shard_gib * (1 << 30)) // EMB_ROW_BYTES
    rng = random.Random(20260910)

    if args.real:
        if not os.path.exists(args.path):
            raise SystemExit(f"--real given but {args.path} does not exist")
        # a real shard's size is a fact, not a request: --shard-gib does not
        # apply, and the banner must not claim a size the file does not have
        n_rows = os.path.getsize(args.path) // EMB_ROW_BYTES
        args.shard_gib = n_rows * EMB_ROW_BYTES / (1 << 30)
        args.keep = True                 # never unlink a real shard
    print(f"engram shard probe -- {args.shard_gib:.2f} GiB, {n_rows:,} rows, "
          f"QD{args.queue_depth}, {args.read_bytes} B reads"
          + (f"\n  REAL shard {args.path}" if args.real else ""))
    if not args.real and (not os.path.exists(args.path)
                          or os.path.getsize(args.path)
                          < n_rows * EMB_ROW_BYTES):
        t0 = time.time()
        build_shard(args.path, n_rows)
        print(f"  built {args.path} in {time.time() - t0:.1f}s")

    per_step = args.batch * ROWS_PER_TOKEN_PER_RANK

    def report(label: str, ms: list[float], budget: float) -> float:
        med = statistics.median(ms)
        p95 = sorted(ms)[min(int(len(ms) * 0.95), len(ms) - 1)]
        verdict = "" if budget <= 0 else (
            "  FITS" if p95 <= budget else "  OVER by "
            f"{p95 - budget:.2f} ms at p95")
        print(f"  {label:22s} median {med:7.2f}  p95 {p95:7.2f}  "
              f"max {max(ms):7.2f} ms{verdict}")
        return med

    with ShardReader(args.path, queue_depth=args.queue_depth,
                     read_bytes=args.read_bytes) as reader:
        if args.real:
            print("  correctness ... skipped (real weights; see "
                  "dsv41_engram_shard.py verify)")
        else:
            check_correctness(reader, rng)
        plans = make_plans(reader, rng, args.batch, args.steps)
        print(f"  batch {args.batch} -> {per_step} rows/step over "
              f"{args.steps} steps\n")

        blocking = report("blocking fetch", time_blocking(reader, plans), 0)
        iops = per_step / (blocking / 1e3)
        print(f"  {'':22s} = {iops:,.0f} IOPS/rank, "
              f"{iops * args.read_bytes / 2**20:.0f} MiB/s\n")

        # The tables sit at layers 1 and 14 of 40. Layer 14 has 13 layers of
        # compute between the submit and the wait; layer 1 has one. Both are
        # reported because a design that only clears the easy half is not a
        # pass -- it is a design that needs the drafter to carry layer 1.
        for layer in (14, 1):
            cover = layer / 40 * args.step_ms
            report(f"layer {layer:<2d} (cover {cover:4.1f}ms)",
                   time_overlapped(reader, plans, cover), args.step_ms * 0.05)
        cover14 = 14 / 40 * args.step_ms
        report("layer 14 pipelined", time_pipelined(reader, plans, cover14),
               args.step_ms * 0.05)
        drafted, waste = time_drafted(reader, plans, args.step_ms,
                                      args.accept, rng)
        report("both, L1 drafted", drafted, args.step_ms * 0.05)
        print(f"  {'':22s} = {waste * 100:.0f}% of layer-1 reads discarded at "
              f"{args.accept:.0%} draft acceptance")

    print(f"\n  budget    {args.step_ms:.0f} ms step; the residual stall is "
          f"judged against 5% of it ({args.step_ms * 0.05:.2f} ms)")
    print(f"  shard/rank at full size: "
          f"{TABLE_ROWS * 2 * EMB_ROW_BYTES / 4 / 2**30:.1f} GiB")

    if not args.keep:
        os.unlink(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
