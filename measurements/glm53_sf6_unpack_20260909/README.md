# SF6 unpack arithmetic reduction — 2026-09-09

The static decode and direct dynamic prefill kernels now reconstruct four
scale bytes together. The SF6 format, 1552-byte packed stage, scale ownership,
and raw-scale release remain unchanged. This change adds no persistent buffer.

The previous loop extracted and assembled every byte separately. The shared
`_sf6_unpack_u8x4` helper spreads four low nibbles and four high two-bit fields
into their byte positions, then adds one broadcast base word. The valid
lossless pack guarantees `base + code <= 255` for every byte, so the addition
cannot carry into a neighboring byte. Signed Int32 values preserve the same
low 32-bit pattern.

Both static CTA barriers remain necessary: the first prevents an expanded
word from overwriting another warp's unread packed input; the second publishes
expanded scales to other MMA consumers. Dynamic loads, shared writes, and
producer publication ordering also remain unchanged. This reduces unpack
arithmetic, not shared-memory traffic or synchronization count.

## Validation

- Local core: 71,161 checks and 50 megakernel regressions; sensitivity: 7 checks.
- Targeted pack, ownership, dispatch, static, dynamic and original runner checks:
  52 tests. The updated runner separately passes 8 tests, including rejection of
  missing/duplicate comparison arms and damaged assembly artifacts.
- Static tests exercise the actual helper AST over every low-plane pattern,
  every high-plane pattern, every valid base/code/byte position and signed
  boundary cases. Existing 128-coroutine in-place and ring tests remain active.
- Dynamic tests use the actual helper and producer AST for both FC1/FC2 halves,
  every lane and all 64 codes, including unsigned/signed loads.
- Production-image CPU compile: all five selected stages passed in
  `compiler/report.json` at source `b19c9dd51075a413ababcf4bd8b8002624050200`.
  This includes decode M6, ordinary M16, compatibility u/v/t/q, expansion
  sizes 1024/2048/4096 and direct dynamic prefill TM128. The runner correctly
  reports full transport-suite coverage incomplete.

The compile run used `fleet.sh run --cpu`, immutable image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
runc, no visible GPUs/network, two CPU cores, bounded memory and fresh caches.
It did not restart serving or acquire the GPU queue.

## Isolated assembly comparison

`probes/sf6_unpack_compile.py` compiles the previous scalar arithmetic and the
actual production helper with identical nonconstant loads/stores for 1, 4 and
8 output words per thread. It preserves all six PTX/CUBIN/SASS artifacts and
their hashes; the CPU runner requires every arm and validates the files.

The initial comparison probe failed before producing assembly because its
decorated kernel needed explicit constexpr loops when building Python lists.
`codegen-v1-failed/` retains that original failure. Commit `50940d44` makes the
loops explicitly constexpr, matching the production plain Python method's
trace-time unrolling. No production kernel was changed by that probe fix.

The next attempt was stopped before container creation by the existing 12 GiB
host-memory gate (9,499,783,168 bytes available). No instruction-count or speed
claim is made from this unfinished comparison.

GPU numerics, graph replay and serving step speed for this change have not
been measured. The prior SF6 adoption timings do not measure this optimization.
