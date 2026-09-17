# C1 CTA · sixteen-row · joined-query kernels: every k-block load of a warp in flight — 2026-09-17

Operator: item 2 of the lever list ("밀집 GEMM 메가커널"), then "너는 c=1 개선이라도 하던가", then "ㄱㄱ".

## Where the C=1 dense lane stood before this

The 09-15 profile's per-kernel times (KDA input CTA 112 µs, input pack 20 µs) include PDL waits. Same-build
eight-row cells on srv4 (the `dense_cells` lane of #1067's run, real rank-3 weights, served route, chains of
distinct layers with L2 evicted) read:

| Cell (8 rows, served) | chain evicted µs/layer | GB/s | 273 GB/s floor µs | calls/forward |
|---|---:|---:|---:|---:|
| kda.in_proj 6416×4096 | 72.5 | 204 | 54 | 34 |
| kda.o_proj 4096×2048 (TX) | 32.2 | 146 | 17 | 34 |
| mla.o_proj 4096×4096 (TX) | 63.5 | 148 | 34 | 11 |
| mla.query pair 2×4096×1536 | 49 | 145 | 26 | 11 |
| mlp.gate_up 6144×4096 | 85 | 167 | 52 | 3 |
| mlp.down 4096×3072 (TX) | 56 | 127 | 26 | 3 |
| drafter gate_up / down | 76 / 45 | — | 52 / 26 | 5 |

About 7.6 ms of dense time per C=1 forward against a 4.4 ms byte floor. The 0.7 ms shared-expert share cannot
hide beside the routed kernel at eight rows either (st_c2_shared_overlap_20260915: S − R = 30 µs/layer) and is
MoE-lane work. The eight-row cells match the sixteen-row cells on the same bytes, so "eight rows through the
wide pack" (named in st_decode_profile_c2_20260915) is not a candidate.

## Hypothesis

The served kernels give each warp 2–4 k-blocks and a two-stage ring (`NB=2, DIST=1`): the next load is issued
only after the previous one landed, so a CTA serializes 2–4 DRAM round trips before its epilogue. The v2
kernel's L2-prefetch cell (#1067) did not touch these kernels (its knob lives in `mk_gemm2_kernel`), so their
latency question is unmeasured.

Candidate: a per-instantiation ring depth that issues a warp's k-blocks together, with the same arithmetic in
the same order (bench knob `set_c1_deep`, `probes/engine_dense_cells.py` arms `bound_deep` / `pair_deep`):

| Kernel | k-blocks per warp | served ring | deep ring | dynamic smem | blocks/SM |
|---|---:|---|---|---:|---:|
| ordered<16,3> (K 2048 outputs) | 2 | 2 stages, 1 in flight | 2 stages, both in flight | 27,648 (same) | 2 |
| ordered<24,3> (K 3072) | 3 | 2 / 1 | 3 / 3 | 40,960 | 2 |
| ordered<32,2>, <32,3> (K 4096) | 4 | 2 / 1 | 3 / 2 | 45,056 | 2 |
| input_cta<1> (KDA input, MODE 1) | 4 | 2 / 1 | 3 / 2 | 32,768 | 3 |
| joined query (cta3, K 1536) | 4 | 2 / 1 | 4 / 3 | 31,744 | 3 |
| rows16 K 4096 | 4 | 2 / 1 | 3 / 2 | +4,608 | 3 |
| rows16 K 3072 | 3 | 2 / 1 | 3 / 3 | +4,608 | 3 |
| rows16 K 2048 | 2 | 2 / 1 | 2 / 2 | same | 3 |

A four-stage ring for the K 4096 cells would drop them to one block per SM; two of four in flight keeps two.

## Method

`probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes dense_cells --seqs 1,2 --samples 2`: every
served route against its `_deep` arm at 8 and 16 rows, single layer and chains of distinct layers, warm and
evicted, zero-tolerance byte gate first, then two B/A/A/B brackets per comparison.

## Result — rejected. Production is unchanged; the arm is not merged.

Ticket `c1deep-0917`, srv4 GB10 single lane (GO 00:30:27 KST, no serving container on srv4 during the run),
frozen source commit `3899b693` (kernels.cu SHA-256 `c58fbd6396f5c832…`, compile receipt `compile-deep.json`:
40 kernels, every deep instantiation 63–80 registers, 0 local bytes, the served ones unchanged). Rank-3 weights of
`/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors` and the DFlash2 drafter, identical RTN
W4 packs in every arm. Raw events: `dense-deep-3899b693.jsonl`; tables: `python3 summarize.py dense-deep-3899b693.jsonl`.

**Numerical gate: PASS, 34 of 34 groups** (every cell × 8/16 rows × single/chain, forward and reverse replay, six
input magnitudes, NaN-poisoned outputs, rebound TX descriptors with guard rows): the deep arm reproduces the served
route's BF16 bytes at zero tolerance. The change is what it claims — the same arithmetic in the same order.

**Timing: neutral to slower in every serving-regime cell.** Mean change of two B/A/A/B brackets, `_deep` against the
served route; chain = distinct real layers, evicted = 128 MiB flush before every replay.

| Cell | rows | single warm | single evicted | chain warm | chain evicted | chain evicted µs/layer served → deep | ms/forward Δ (evicted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| kda.in_proj | 8 | +6.9% | −0.7% | +0.5% | +0.4% | 72.6 → 73.0 | +0.011 |
| kda.o_proj | 8 | −0.8% | +1.2% | +4.7% | **+2.9%** | 32.3 → 33.3 | +0.032 |
| mla.o_proj | 8 | +4.9% | +1.3% | +1.9% | +0.3% | 62.1 → 62.4 | +0.002 |
| mla.query pair | 8 | +13.3% | +2.0% | −1.4% | −0.5% | 48.9 → 48.6 | −0.003 |
| mlp.gate_up | 8 | +5.8% | −0.5% | +1.5% | −0.5% | 85.3 → 84.9 | −0.001 |
| mlp.down | 8 | **+22.0%** | +5.7% | **+16.2%** | **+5.9%** | 56.8 → 60.2 | +0.010 |
| drafter.gate_up | 8 | +4.9% | +1.0% | +1.0% | +0.5% | 76.1 → 76.5 | +0.002 |
| drafter.down | 8 | +13.2% | +3.0% | +5.6% | +2.9% | 45.1 → 46.4 | +0.007 |
| kda.in_proj | 16 | +8.0% | +0.4% | +2.4% | +1.9% | 74.7 → 76.1 | +0.047 |
| kda.o_proj | 16 | −0.8% | +8.7% | +2.1% | **+4.7%** | 35.6 → 37.3 | +0.057 |
| mla.o_proj | 16 | +4.8% | +1.8% | +2.6% | −1.7% | 65.7 → 64.6 | −0.012 |
| mla.qkv_a | 16 | +0.1% | +4.9% | +1.0% | +2.5% | 38.4 → 39.4 | +0.010 |
| mlp.gate_up | 16 | +3.7% | +1.3% | +3.9% | +0.3% | 87.4 → 87.7 | +0.001 |
| mlp.down | 16 | +1.1% | +5.6% | +3.2% | **+6.4%** | 58.8 → 62.6 | +0.011 |
| drafter.gate_up | 16 | +1.1% | +1.1% | +2.2% | −0.1% | 79.2 → 79.1 | −0.000 |
| drafter.down | 16 | +3.4% | +5.1% | +8.0% | **+6.9%** | 47.6 → 50.8 | +0.016 |

Summed over a forward (chain evicted, calls per forward as in the baseline table): **+0.06 ms at 8 rows, +0.13 ms
at 16 rows** — slower, and nowhere near the −0.3 ms this record set as its floor for keeping the lever open.

## Reading

- **The serialized round trips were not the time.** The K 2048 output cells (`ordered<16,3>`, `rows16<true,16,3>`)
  change nothing but the issue order — both loads before the first wait, same two stages, same shared memory,
  same residency — and lose 3–5% in chains. The K 3072 cells (a third stage, still two or three blocks per SM)
  lose 6–7%. More outstanding requests per SM add queueing on a path that the 96–144 resident CTAs already fill;
  they do not add throughput. This is the same shape as #1067's `lf<n>` MoE prefetch and its v2 `_l2` cells.
- **Warm single-layer replays lose the most** (+5% to +22%): with the weights L2-resident, bulk-issued loads
  only queue at L2. The served ring's one-ahead issue is the better fit for this memory path.
- With #1067 (prefetching a v2 CTA's whole slice into L2 is neutral) this closes the two latency hypotheses for
  the small W4 GEMMs: neither first-touch latency nor in-flight depth is the lever, at 8 or 16 rows. What remains
  of the 3.2 ms gap to the byte floor sits in the request pattern itself (a warp's 1 KB pieces of interleaved
  128-row tile records, #1067's "128 B row-segment access pattern") and in the per-CTA epilogue chain. A pack
  layout that gives each CTA one contiguous run is layout work with the same numerics — it is the next question
  if anyone reopens this lane; it is not a plan.
- **The dense item closes here for both C=1 and C=2.** C=1's dense lane is within about 2.5 ms of its floor with
  every measured lever spent (v2 lane, PDL, C1 CTA kernels #939/#946, producer packs #968/#978, sixteen-row CTAs,
  input reuse, L2 prefetch, ring depth). The C=1 step's remaining levers are elsewhere: mHC at 50 µs per launch
  (4.4 ms), the shared expert inside the routed kernel (MoE lane, W4A4 numerics), the torch DSA top-k (0.85 ms),
  the drafter tail.

## What this did not measure

One rank, kernel component intervals, synthetic rows over real weights. No boot, no consumer step/s, no
acceptance, no quality — none is owed for a component loss. The measured arm lives in `deep-ring.patch` (against
main `da917a4d`; the run's commit `3899b693` carries it) and in nothing merged: production keeps the served rings,
and `kernels.cu` and `probes/engine_dense_cells.py` on this branch are byte-identical to main.

## Reproduction

```sh
# srv2 seed image, CUDA hidden: compile receipt (compile_deep.py is the scratch script whose output is compile-deep.json)
docker run --rm --init --runtime runc --network none --cpus 2 --memory 8g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/repo \
  --mount type=bind,src=<worktree with deep-ring.patch applied>,dst=/repo,readonly --mount type=bind,src=<out>,dst=/out \
  -w /repo --entrypoint python3 a9b53fd066bb /out/compile_deep.py
# srv4 single lane, from that worktree on srv2
ST_PROBE_GIB=24 bash bench/fleet.sh run --gpu --detach <s> 20 "<note>" -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes dense_cells --seqs 1,2 --samples 2
```
