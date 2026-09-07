# GLM53 MoE register scatter experiment

**No production change selected.** Both direct-register variants regress;
warp-owned shared staging is effectively unchanged at the representative
M6/U40 shape. All three pass the bounded numerical/replay gate.

The existing `t` MoE lane spends 22.87 ms in 42 calls in the retained
September 7 rank-0 trace. This is profiled attribution, not a fresh timing
baseline. This experiment targets FC2's shared-memory epilogue: each of its
32 output tiles stores BF16 values to shared memory, synchronizes, reloads
them and scatters the routed output, then synchronizes again.

Two private kernel variants preserve the existing FC1/FC2 MMA operations,
weight layouts and rounding sequence (FP32 down scaling, BF16 conversion,
FP32 route weighting, satfinite BF16 atomic reduction):

- `pair`: scatter each accumulator's two adjacent columns directly.
- `vector`: collect four lane pairs with warp shuffles, retaining the stock
  16-byte BF16 vector reduction. This avoids quadrupling atomic instructions.

One final barrier per item preserves ownership of the token/route-weight
cache when warps advance to the next item. The 64 per-output-tile epilogue
barriers become one item barrier; pipeline barriers remain unchanged.
The shared-memory allocation remains 101,376 bytes. There is no claim of
higher CTA occupancy, different weight bandwidth, or whole-model speedup.

Both variants compile for SM121 without a CUDA device. The compiler checks
all 4,096 output coordinates for complete unique coverage, adjacent pair
alignment and correct four-lane vector membership. Raw compilation logs are
in [cpu/](cpu/). Generated source SHA-256:

- pair: `704e3baa87d0b6c6352e6dfe589499931c708b121934267c32acc4192b74028a`
- vector: `7e5d5f4a44cf0bab6201b891b06d9c477a2ea65184c83ebf95b9a9207835cf57`

The generator at `81012d5` is unchanged in the GPU snapshot `857b6b0`.
CPU compilation used one fresh device-free `runc` container per arm on srv1,
the immutable serving image, CPU14–15, a 2 GiB memory limit and >=12 GiB
available before admission. No production overlay was edited or deployed.
The full CPU logic gate passed 6,689 checks, 30 megakernel regressions and
107 fleet regressions. Two focused memory-guard behavior tests also pass.

## First GPU result: both direct-register variants regress

Fleet job `moedirect0908` held the GPU from 05:33:52 KST and completed the
approved-main restoration at 05:45:19, exiting 0. Both variants passed all
130 numerical differential rows (13 routing cases, five replays and two
compared arms). The memory guard recorded no issue, with minimum available
memory 93.03 GiB. Each cell used 32 balanced pairs per shape/cache condition.

| Variant / M6 condition | Baseline us | Candidate us | Latency change | Candidate faster |
|---|---:|---:|---:|---:|
| pair, U40 cold | 641.376 | 655.136 | +2.15% | 1/32 |
| pair, U8 warm | 104.272 | 108.320 | +3.88% | 4/32 |
| vector, U40 cold | 635.888 | 657.424 | +3.39% | 1/32 |
| vector, U8 warm | 105.392 | 116.480 | +10.52% | 0/32 |

All twelve shape/cache medians in each variant regress. Both run orders
retain the regression in these representative cells. The pair's extra atomic
instructions and the vector's warp gather are plausible costs; they have not
been isolated by a new instruction-level profile. Neither variant is selected
for serving or default promotion. Sanitizers and a serving bracket were not
run for these slower implementations. Full raw results are in [gpu/](gpu/),
with order-specific comparisons in [summary.json](summary.json).

## Follow-up: keep shared staging, synchronize only the owning warp

The third `warp` candidate keeps the stock BF16 staging and vector reduction.
For M<=8 it remaps the scatter to the actual MMA ownership: warp 0 owns
columns 0–15, 32–47, 64–79 and 96–111; warp 2 owns the complementary 16-column
segments. The initial assumption of contiguous 64-column ownership was
rejected by the CPU checker before GPU execution. Exact coordinate diagnostics
are retained in [cpu/ownership-diagnostics/](cpu/ownership-diagnostics/).

With the remapped scatter, each valid output value is written and read within
one warp, allowing the two tile barriers to become warp barriers. One CTA
barrier per item still protects metadata ownership. M>8 retains the original
mapping and barriers. The fixed source passes the full 4,096-coordinate check
and device-free compilation (source SHA-256
`85c348c0458ed5ef4b26d912f3847eb00f7d926b64231eb94a2cf22380bcdf95`).
The GPU snapshot is `2e4ea40`, run as `moewarp0908` starting 05:49:25 KST.
It also passes 130 differential rows and the changed-input/routing graphs,
including the M16/M32 fallback. There are no resource-guard issues; minimum
available memory is 93.06 GiB.

| Warp / condition | Baseline us | Candidate us | Latency reduction | Faster pairs |
|---|---:|---:|---:|---:|
| M6/U40 cold | 637.456 | 637.136 | +0.05% | 19/32 |
| M6/U40 warm | 635.648 | 637.072 | -0.22% | 12/32 |
| M6/U8 cold | 137.168 | 136.224 | +0.69% | 15/32 |
| M6/U8 warm | 98.368 | 98.048 | +0.33% | 17/32 |
| M8/U40 cold | 639.232 | 640.752 | -0.24% | 9/32 |
| M8/U40 warm | 640.784 | 641.536 | -0.12% | 14/32 |

The tiny U40 cold change reverses by run order (+0.14% / -0.10%). The U8
warm median has only 17/32 faster pairs. These results do not establish a
stable kernel improvement. The precommitted escalation rule (>=1.5% U40
cold or >=3% U8 warm) is false, so no sanitizer/serving escalation was run.
This leaves the warp route unpromoted, rather than proving every use case
slower. Full raw results are in [warp/](warp/), with order-specific comparisons
in [warp-summary.json](warp-summary.json). Recompute with
`python3 probes/analyze_moe_direct.py <warp-evidence> --variants warp`.

No full-model step/s, output tok/s, acceptance or serving-quality comparison
was collected for these variants. Approved-main restoration completed at
05:58:53 KST and the runner exited 0. The read-only
[all-rank restoration proof](warp/restore-live.json) confirms HTTP 200,
all four containers running the immutable image, matching approved-main
CUDA/MoE source hashes, and CTA=2/input-reuse=1/MoE-static=`t` defaults.
Restored main: `4b0f1d15a1608848faa3c1a957be8f4933ae951d`.

## Reproduction protocol

For the initial direct-register comparison, the immutable
checkout is `/home/choiceoh/stkernel-moe-direct-gpu-20260908` at `857b6b0`.
The runner always restores approved `origin/main` through its independent
restore checkout before releasing the fleet.

Each variant runs in its own fresh container, with the unchanged `t` kernel
and independent stock kernel in the same process. Different outer JIT names
and source dependencies prevent accidental executable reuse; captured CuTe
executables are retained after dispatcher cache switches.

The synthetic fixture uses E288/H4096/I512/top8 and M1/2/6/8/16/32, testing
13 token/routing cases, five changed-input and changed-routing replays,
finite results and exact-zero routing. The numerical threshold is the
preexisting stock differential gate (max of four times stock-repeat noise
and 1% of reference magnitude), without relaxing it for this candidate.
Performance is measured only after those gates pass, using 32 balanced
forward/reverse pairs in each of six M6/M8 routing shapes, separately with
warm weights and read-drained L2 eviction while keeping inputs hot.

Recompute measured medians and order-specific reductions with
`python3 probes/analyze_moe_direct.py <evidence-directory>`.
No profile/default changes are included.
