# L9 M64 rejection and repair

The frozen candidate `34b5852e537129f8ab2a586a434921e7afcf6d53`
failed the actual-weight component numerical gate on 2026-09-13 at
19:19 KST under the granted ordinary FIFO session `st-prefill-m64-L9r2`.
The queue released its lease after 15.2 seconds of payload time. The first
2672-row balanced case failed all 2672 rows: maximum relative L2 2.5130472
and relative peak 4.9704642. Three stock controls agreed exactly. No timing
comparison, long-scale gate, engine boot or consumer measurement ran.
The fleet log's self-SSH warning was not the numerical failure's cause.

`m64-prefill.json`, `m64-prefill-gate.log` and `fleet-run.log` retain the
failure. `runtime-layouts.log` is a subsequent CPU-only CuTe compile in the
same immutable ST image (`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`),
with no CUDA device nodes. It inspects the actual shared storage pointer
swizzle and MMA thread partitions; it is not GPU numerical proof.

The repair addresses three concrete contracts:

1. Q1 immediate and deferred stores inherited M128 flattened coordinates.
   M64 has 64 M coordinates before advancing K, so the inherited doubled
   K coordinate aliases writes and runs past the 4096-byte stage. Both
   generated stores now use M64 coordinates. CPU tests execute the actual
   old/new AST assignments and require every byte to match the physical
   M128 reference location, with 4096 unique in-range M64 addresses.
2. M64 scatter assigned rows by `warp >> 1`, crossing the existing
   `warp & 2` barrier groups. The compiler's M64 MMA partition assigns
   M16 strips by `warp & 3`. Scatter now follows that ownership, with
   columns selected by `warp >> 2`. Tests check group membership as well
   as all live output elements, routed destinations and boundary tails.
3. Each logical M64 tile owns a full physical 128-row activation-scale
   atom. The declared scale view and zeroing span now cover that entire
   allocation, not only the logical row count rounded to 128.

The gate now retains Q0 routes and sampled packed-input/scale hashes before
comparing output, and attempts weight-identity auditing even after failure.
The L9 rejection has no Q0 equality evidence because its old gate compared
output first. Tolerances are unchanged. The repaired candidate still needs
fresh full CPU compilation and GPU numerics before any performance claim.
