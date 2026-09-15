# C=2 mHC consumers read the BF16 coefficient pack — 2026-09-15

**Default on.** K=7 verify steps at C=2 (16 rows) now read the lossless BF16 coefficient pack that C=1 (8 rows)
already reads. This covers both consumer families: the 84 packet consumers per step and the ordinary
(all-reduce) consumers. The FP32 form stays reachable as the same-build control `MHC(..., packed_rows=8)`, which
only the probe passes.

- Bitwise exact against FP32 coefficients of the same build: 23 zero-tolerance groups on the shipped source `cd067569`.
- Single-GPU component time at 16 rows, served dispatch versus the control:
  - packet consumers: **−13.9% warm / −14.3% evicted**;
  - ordinary consumers: **−10.0% / −24.9%**.
- C=1 is unchanged: 8-row controls are within ±0.6%.
- The 2026-09-13 M14 failure was a layout bug in that probe's adapter. Its numbers replay exactly on the GPU from the
  lane's own seeds.

This is kernel evidence from one GB10 beside production, with no transport and no model boot. It is not an engine
speed claim (D17). Consumer tok/s, acceptance and TP4 PDL overlap were not measured.

## Results — shipped build `cd067569`

Ticket `st-c2-mhc-final0915` ran on the single-GPU lane (srv4 GB10, 09:03:46–09:05:19 KST). Image:
`sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5`, Torch 2.13.0+cu132, CUDA 13.2.

Inputs and timing method:

- **Coefficients.** The real hc coefficients of `/home/choiceoh/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors`:
  all 90 `hc.*_fn` tensors (sha256 over all 90 `68d566be…`), with their scales, bases and norms. All 90 are
  BF16-origin.
- **Chains.** One captured graph holds 84 packet calls on 84 distinct packs, or 5 ordinary calls.
- **Timing.** Per chain replay. Each comparison ran two B/A/A/B brackets, 64 replays per sample warm and 32 per sample
  after a 128 MiB flush ("evicted"). This table gives the mean of four samples per arm; the minima change none of its
  percentages by more than 0.2 points.

| Consumer | Rows | Cache | Control (FP32) µs/call | Serving µs/call | Change | Per step (84 or 5 calls) |
|---|---:|---|---:|---:|---:|---:|
| packet | 16 | warm | 24.25 | 20.87 | **−13.94%** | −284 µs |
| packet | 16 | evicted | 24.59 | 21.07 | **−14.31%** | −296 µs |
| ordinary | 16 | warm | 23.30 | 20.96 | **−10.01%** | −12 µs |
| ordinary | 16 | evicted | 33.80 | 25.40 | **−24.88%** | −42 µs |
| packet | 8 | warm | 15.32 | 15.31 | −0.10% (same kernel) | — |
| packet | 8 | evicted | 15.58 | 15.51 | −0.44% (same kernel) | — |
| ordinary | 8 | warm | 15.92 | 15.88 | −0.35% (same kernel) | — |
| ordinary | 8 | evicted | 20.22 | 20.32 | +0.53% (same kernel) | — |

Same-graph decompositions from the same run (means; see the raw report for minima):

| Comparison (B → A) | Rows | Warm | Evicted |
|---|---:|---:|---:|
| packet FP32 → packed | 16 | −13.95% | −14.46% |
| packet FP32 → packed (the existing C=1 gain) | 8 | −15.46% | −16.86% |
| ordinary persistent FP32 grid → packed consumer | 16 | −8.98% (min −10.07%) | −23.56% |
| ordinary persistent FP32 grid → FP32 consumer | 16 | +1.17% | −6.14% |
| ordinary FP32 consumer → packed consumer | 16 | −5.29% (min −11.16%) | −18.61% |
| ordinary persistent FP32 grid → packed consumer | 8 | −14.11% | −30.35% |

The two rows marked with a minimum contain slow samples. The consumer comparison has two, at 121.5 and 122.1 µs; the
grid comparison has one, at 110.2 µs. Their steady state is about 104.7 µs, and the minima show it.

The 16-row component saving is about 0.30 ms warm (0.34 ms evicted) per C=2 step over 84 packet and 5 ordinary
calls. The 2026-09-15 profile (`../st_decode_profile_c2_20260915`) priced the C=2 packet lane at 5.61 ms per step,
about 82 µs a call against this probe's 24 µs. The two are not the same quantity:

- Under the serving PDL chain a consumer launches before its upstream exchange finishes and waits inside the kernel,
  so a profiler's kernel duration can include that wait.
- This probe has no upstream collective. It prices compute, not the served lane.

Raw: `gpu-final-cd067569.json`, `gpu-final-cd067569.log`. Tables: `python3 summarize.py gpu-final-cd067569.json --markdown`.

## Why the pack is exact, and why it stopped at 8 rows

- `MHC` keeps a BF16 pack only when `torch.equal(fn, fn.bfloat16().float())`: every coefficient already is a BF16
  value.
- BF16 is FP32's sign and 8-bit exponent with a 7-bit mantissa. The FP32 value is the same bit pattern with sixteen
  zero mantissa bits appended, subnormals included. `mk_mhc_unpack_bf16_late` does exactly that: `shl 16` for the
  low half of a pair, `and 0xffff0000` for the high half.
- The projection accumulates `v = 0; v += w0*r0; v += w1*r1; v += w2*r2; v += w3*r3`, the FP32 kernel's unrolled
  `for j` order. Partials, reductions, Sinkhorn, pre and norm are shared code.
- None of that depends on the row count. The 8-row bound was dispatch, not arithmetic:
  - `mhc.py` selected the pack only when `n <= 8`.
  - The native AR consumer admitted 1..8 tokens.
  - The packet entry already admitted BF16 coefficients up to 64 rows.
- Only the consumer instantiations read the pack's `[output, hidden, stream]` layout: `mk_mhc_ar_kernel<true>` and
  `mk_mhc_packets_kernel<true>`. The persistent BF16 kernel `mk_mhc_bf16_kernel` reads `[output, stream, hidden]`.
  The engine never packs that layout, and no serving call reaches that kernel.

## Why M14 failed

The 2026-09-13 `mhc_batch` lane (`probes/engine_decode_capacity.py`, added in `a6881a3f`, run from `2b3a74b9`;
`../st_decode_capacity_20260913`) wrapped the extension in `PackedMhc`. The wrapper swapped the FP32 pointer for the
owner's `[output, hidden, stream]` pack and called `run_mhc(ptrs, scalars, ints, True, False)`. With
`ar_consumer=False` that call reaches `mk_mhc_bf16_kernel`, which indexes `fn[m*4*4096 + j*4096 + h]` as
`[output, stream, hidden]`. Every coefficient came from a permuted position.

**It was a layout (pack geometry) bug in the probe adapter.** It was not BF16 precision, and it was not a
reduction-order difference.

**The recorded signature fits a layout permutation.**
- The residual matched: it does not use the coefficients.
- Post, the first output that does, differed in all 56 elements (14 rows × 4).
- The replay at input scale 0 passed, because every coefficient multiplies zero. The replay at scale 0.001 failed.

**Neither alternative can produce it.**
- Precision: the coefficients are exactly representable, so exact storage moves nothing.
- Reduction order: a different FP32 order moves an element by about one ulp, a relative 1e-7. The recorded failure
  is a relative 3.5e-4.

**The GPU replays it exactly** (`m14_diagnosis`, both tickets). The lane's own seeds were used: 91715 for its first
pack, and 91613 plus two `normal_` passes for its inputs.

| Form at 14 rows, input scale 0.001 | post mismatched | post max abs | comb | layer input |
|---|---:|---:|---:|---:|
| recorded failure (2026-09-13) | 56/56 | 0.000341951847076416 | — | — |
| the lane's vector pack → `mk_mhc_bf16_kernel` | 56/56 | **0.000341951847076416** | 224/224 | 1798/57344 |
| a `[output, stream, hidden]` pack → the same kernel | 0/56 | — | 0/224 | 0/57344 |

The maximum absolute difference is the recorded value to every printed digit. The relative difference agrees to seven
significant digits: 3.536289e-4 here in float64, against 3.536289e-4 recorded. The persistent BF16 kernel with the
layout it reads is bitwise exact at 14, 16 and 28 rows and input scales 0.001 and 1 (`m14_scalar_layout`).

A CPU estimate agrees independently. `m14_sim.py` draws the lane's distributions and reads the pack as that kernel
does, using the torch equations. Over 200 trials the largest post difference has median 2.9e-4 (p10 2.4e-4,
p90 3.8e-4), and 3.42e-4 sits at the 79th percentile (`m14-sim.log`).

An earlier and different failure is still unexplained. On 2026-09-09 the vLLM overlay's v4 boot logged
`AR consumer MHC mismatch T=16 fp32=False` for its own scalar-layout BF16 path (`probes/decode_transport_sf_README.md`).
Its details were never logged, and that kernel build is gone. The current `mk_mhc_bf16_kernel` is exact with the layout
it reads (above), and serving does not use it.

## Rejected form: expanding a block's coefficients once

A 16-row block runs twice the tokens of an 8-row block against the same coefficients. So the first candidate also
compiled `mk_mhc_packets_kernel<true, true>`: after the 24 vector loads it expanded the coefficients once, into the
96 FP32 registers the FP32 kernel holds.

- It was bitwise exact.
- It was slower than the C=1 kernel's per-multiply unpack at both widths.
- It compiled to 168 registers per thread, against 128 for the per-multiply form and 167 for FP32.

Ticket `st-c2-mhc-packed0915`, source `180d85c2`, 08:53:15–08:54:41, same image, same 84 packs:

| Comparison (B → A) | Rows | Warm | Evicted |
|---|---:|---:|---:|
| FP32 → per-multiply unpack | 16 | −13.85% | −14.53% |
| FP32 → once-per-block expansion | 16 | −11.64% | −12.24% |
| per-multiply → once-per-block | 16 | +2.05% (min +3.07%) | +2.85% |
| per-multiply → once-per-block | 8 | +5.56% | +5.65% |

The expansion, its template parameter and its native argument were removed. The shipped kernels compile to main's
resources. The same run also measured the ordinary consumers: at 16 rows, persistent FP32 grid → packed consumer was
−9.26% warm / −23.84% evicted. Raw: `gpu-forms-180d85c2.json`, `gpu-forms-180d85c2.log`,
`compile-forms-180d85c2.json`.

## What changed

- `engine/kernels/dense/mhc.py`
  - Consumers take the pack up to `MHC.PACKED_ROWS = 16` rows (was 8), at packet and ordinary boundaries alike.
  - Coefficients that are not BF16-origin keep FP32 storage. Their ordinary launches of 9–16 rows now use the FP32
    consumer kernel, which is exact in the gate. The served GLM-5.3 coefficients are all BF16-origin.
  - `packed_rows=8` is the same-build control. It is a constructor argument, not a serving knob (D11).
- `engine/kernels/dense/kernels.cu`: the AR consumer admits 1..16 tokens (was 1..8). No device code changed.
  - At 16 rows the ordinary one-shot sum is not a PDL consumer (`cells.ONESHOT_CONSUMER_MAX_ELEMENTS` is 8 rows on
    this main), so it releases nothing early. The mHC consumer then launches behind it and computes the same bytes:
    "a serialized launch remains correct".
  - Branch `ostcode/c2-oneshot-consumer` (unmerged) raises that one-shot bound to 16 rows. With both merged, these
    launches can also overlap the sum; that combination is not measured here.
- `probes/engine_mhc_c2_packed.py`, lane `mhc_c2_packed` of `probes/engine_kernel_check.py`: the gate and timing.
- `tests/test_engine_mhc_packed_rows.py`: the weight and native form for rows 1..64, both families, both storages,
  the control, and the probe's adapters.
- `tests/test_engine_direct_mhc_cuda.py`: the served cross-family gate (packets against the ordinary consumer on the
  rank-ordered sum) also covers 8 and 16 rows.
- `probes/engine_decode_native_compile.py`: the resource report lists every mHC kernel, not only the consumers.

## Native compile and CPU checks

The production-flag dense extension was compiled and loaded with CUDA hidden (`CUDA_VISIBLE_DEVICES=`,
`NVIDIA_VISIBLE_DEVICES=void`) in the same image:

- main `9c45086a`: 78 s;
- the two-form candidate `180d85c2`: 72 s;
- the shipped merge `cd067569`: 65 s.

| Kernel | Registers/thread | Stack | Shared bytes |
|---|---:|---:|---:|
| `mk_mhc_packets_kernel<true>` (C=1 and now C=2 packets) | 128 | 16 | 28,736 |
| `mk_mhc_packets_kernel<false>` (FP32 control) | 167 | 16 | 28,736 |
| `mk_mhc_ar_kernel<true, 4096/5120>` | 127 | 16 | 28,736 |
| `mk_mhc_ar_kernel<false, 4096/5120>` | 167 | 16 | 28,736 |
| `mk_mhc_kernel`, `mk_mhc_bf16_kernel`, `mk_mhc_v41_kernel` (4096/5120) | 167 | 16 | 28,736 |
| rejected `mk_mhc_packets_kernel<true, true>` (only in `180d85c2`) | 168 | 16 | 28,736 |

All 12 mHC kernels of `cd067569` match main `9c45086a` exactly (`compile-main-9c45086a.json`, `compile-final.json`).
`cpu-tests.log`: 103 tests on `cd067569`, 94 passed and 9 GPU-only skipped. The CUDA tests ran in the GPU gate.

## The numerical gate

The shipped build's ticket ran every check below. A failure stops the probe; both reports end in `complete PASS`. The
two-form ticket ran the same checks with ordinary widths 1/8/9/16/17 only and no refusal case.

- **Served dispatch across families.** `tests.test_engine_direct_mhc_cuda` compares packet consumers with the
  ordinary consumer on the rank-ordered BF16 sum. It uses both coefficient storages, rows 1/7/8/16/28/64, rebound
  descriptors and 12 changing replays. The comparison is exact, with no skips.
- **Refusals.** An AR consumer launch at 17 rows is refused before launch; 16 rows are admitted (`mhc_packed_refusals`).
- **Breadth: 19 groups.** Six real packs per graph. Packets at 1/2/7/8/9/12/15/16/17/32 rows; ordinary at
  1/2/7/8/9/12/15/16/17 rows.
  - Packet arms: FP32, packed, the control owner, the serving owner.
  - Ordinary arms: persistent FP32 grid, FP32 consumer, packed consumer, control owner, serving owner.
  - Input scales 0/0.001/1/32/0.5, alternating rank banks, rebound descriptor, the direct-MHC canary columns.
  - NaN-poisoned outputs before every replay, forward and reverse replay order.
- **Depth: 4 groups.** 84-pack packet chains and 5-pack ordinary chains at 8 and 16 rows, before timing.
- **Comparisons.** Every arm is compared bitwise with its family's FP32 reference, as integer views, so ±0 counts.
- **Dispatch recording.** Each owner's own dispatch is recorded and must equal the independently stated rule
  (`served_dispatch`).

## Reproduce

CPU tests, in the pinned image with CUDA hidden, from the repository root:

```sh
docker run --rm -e CUDA_VISIBLE_DEVICES= -e CUTE_DSL_ARCH=sm_121a -e PYTHONPATH=/repo \
  -v "$PWD":/repo:ro -w /repo --entrypoint python3 st-engine:bracket-9c45086a0622 -m unittest -v \
  tests.test_engine_mhc_packed_rows tests.test_engine_decode_native_compile tests.test_prefill_oracle_candidates \
  tests.test_engine_kernel_glue tests.test_engine_kernel_shape tests.test_engine_direct_mhc \
  tests.test_engine_direct_mhc_cuda tests.test_engine_mk_mhc tests.test_engine_kda_ring_bench tests.test_fleet_onepass
```

Native compile and load with every mHC kernel's resources:

```sh
docker run --rm -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e CUTE_DSL_ARCH=sm_121a \
  -e PYTHONPATH=/repo -e MAX_JOBS=2 --memory 24g -v "$PWD":/repo:ro -v /tmp/out:/out -w /repo \
  --entrypoint python3 st-engine:bracket-9c45086a0622 -m probes.engine_decode_native_compile \
  --build-root /out/build --output /out/compile.json
```

GPU gate, from a frozen checkout on srv2 through the single-GPU lane:

```sh
git -C ~/stkernel fetch -q origin ostcode/c2-mhc-packed
git -C ~/stkernel worktree add --detach ~/st-worktrees/c2mhc-cd067569 cd067569
cd ~/st-worktrees/c2mhc-cd067569
ST_IMAGE=sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5 ST_PROBE_TREE=c2mhc-cd067569 \
bash bench/fleet.sh run --gpu --detach st-c2-mhc-final0915 10 'C2 packed mHC final' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes mhc_c2_packed \
    --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
    --output /cache/st-c2-mhc-final-cd067569.json
```

For the M14 CPU estimate, run `python3 measurements/st_c2_mhc_packed_20260915/m14_sim.py` in the same image with
CUDA hidden and `PYTHONPATH=/repo`. The rejected two-form run is reproduced from `180d85c2` with the same lane.

## Not measured

- TP4 serving: consumer tok/s, step/s, acceptance, output quality, NIC transport or PDL overlap. The gate has no
  upstream collective.
- Whether the ordinary 16-row consumer overlaps the one-shot sum, which requires `ostcode/c2-oneshot-consumer`.
- A wider consumer grid at 16 rows. The packed kernel's 128 registers would fit two CTAs per SM; the host keeps one
  CTA per SM for the overlapping kernels. That trade is a TP4 question and was not tried.
