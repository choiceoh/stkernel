# GLM53 MoE register scatter experiment

**No production change selected.** Both direct-register variants regress;
warp-owned shared staging is effectively unchanged at the representative
M6/U40 shape. The subsequent actual onepass B-A-B also did not establish a
step benefit: 21.947 -> 21.836 -> 21.848 step/s. All three variants pass the
bounded numerical/replay gate, and all onepass arms pass 18/18 quality with
0/8 corrupt responses.

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

At the end of this microbenchmark round, no full-model step/s, output tok/s,
acceptance or serving-quality comparison had been collected. The subsequent
onepass follow-up below addresses that omission. Approved-main restoration completed at
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


## Actual onepass follow-up

**No measured step improvement.** The candidate is 0.28% below the two-baseline
mean; the baseline boot spread is 0.45%. Output tok/s is 0.91% below that mean,
within an 8.00% baseline spread. One candidate boot does not establish a stable
speedup or a stable regression. The default remains `t`, with CTA=2.

| Arm | Pooled step/s | Window median step/s | Fixed output tok/s | All-request acceptance | Fixed windows |
|---|---:|---:|---:|---:|---:|
| Baseline 1 (`t`) | 21.947 | 21.849 | 68.62 | 43.55% | 83 |
| Candidate (`t,ws`) | 21.836 | 21.827 | 70.83 | 46.62% | 81 |
| Baseline 2 (`t`) | 21.848 | 21.822 | 74.34 | 48.70% | 76 |

Every arm passed 18/18 factual checks and had 0/8 corrupt combined-channel
responses. All request hashes, source hashes and rank/boot identities match
across the intended comparisons, and no traffic-exclusivity issue was recorded.

Prefill is the onepass input-token count divided by TTFT, not an isolated GPU
prefill-kernel duration. 2K warm uses the lowest TTFT of the repeated requests;
32K and 128K each have one combined-question request per boot.

| Context | Baseline 1 tok/s | Candidate tok/s | Baseline 2 tok/s | Candidate TTFT | Change vs baseline mean |
|---|---:|---:|---:|---:|---:|
| 2K warm | 2,410 | 2,410 | 2,500 | 0.883 s | -1.83% |
| 32K | 2,972 | 3,023 | 2,993 | 10.765 s | +1.35% |
| 128K | 3,083 | 3,085 | 3,089 | 41.669 s | -0.02% |

The first 2K requests took 2.409 / 1.942 / 1.921 seconds, so their apparent
first-baseline improvement is affected by startup/JIT warmup. The `ws` kernel
is M<=8 only; these prefill differences do not establish a prefill-kernel gain.

The chain completed at 07:15:12 KST. The unconditional restore deployed
approved main `4b0f1d15a1608848faa3c1a957be8f4933ae951d`, verified all four overlay
copies, booted the public API, and completed at **07:23:12 KST with exit 0**.
The restore receipt records HTTP 200. Subsequent fleet work belongs to the
next holder and is outside this bracket.

The user's explicit onepass request escalated the inconclusive warp candidate
to actual serving, without requiring a microbenchmark speed threshold. Commit
`27c4a1a` integrates the tested warp mapping behind `VLLM_GLM53_B12X_STATIC_V2=t,ws`.
The profile still defaults to `t`. The parser admits `ws` only with the plain
`t` configuration; exact GLM TP geometry retains the existing dispatcher gate,
and M>8 retains the original barriers and scatter mapping. The cache key and
M<=8 serving marker distinguish the two compiled kernels.

The same-build B-A-B run is `moewsonepass0908`, using the immutable checkout
`/home/choiceoh/stkernel-moe-onepass-20260908` and the canonical fleet boot lane.
`probes/run_moe_warp_onepass.sh` runs the integrated numerical fixture, deploys
the same source once, and calls the unchanged `bench/onepass.py` through the
SSE channel recorder. Each arm uses C=1, TP=4, SPEC_K=5, CTA=2, contexts
2K/32K/128K and three fixed 2,048-token outputs. The API binds to loopback
port 18000, and the harness requires exclusive request counters.

The primary step metric pools engine steps and elapsed seconds from intervals
wholly inside those fixed outputs, excluding 0.5 seconds at each response edge.
Output tok/s pools the three fixed responses' post-first-token tokens and decode
time. Acceptance covers the entire onepass workload, so it is reported separately.
Individual windows are correlated; independent boots are the replication unit.

The integrated GPU fixture passed all 130 stock-differential and mutated-graph
rows, including M1/2/6/8 and M16/32 fallback, with unchanged tolerances. CPU
baseline/candidate compilation passed, as did 6,691 logic checks, 30 megakernel
regressions and 107 fleet regressions on Linux. The initial macOS attempt and
its stale audit / platform-specific mock failure remain in the evidence; the
logic audit was refreshed after verifying only the MoE control test changed.

Raw records, SSE channels, all-rank source/capture receipts and the unconditional
approved-main restore receipt are under [onepass/](onepass/). Recompute the final
comparison with:

```sh
python3 probes/analyze_moe_onepass.py measurements/glm53_moe_direct_scatter_20260908/onepass
```
