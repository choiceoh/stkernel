# C=2 (16-row) dense W4 cells: sixteen-row CTAs for the bound cells — 2026-09-15

Operator order "st커널 c=2 최적화 개선", then "c=2 저거도 도입". Since #950 production decode runs C=1
(8 verification rows) and C=2 (16 rows). The C1 cells of #939/#946 had no 16-row counterpart. This
directory measures, on real rank weights against same-build controls, what each dense W4 family should
run at 16 rows. Six sixteen-row CTAs are default; one is rejected.

**Adopted first (#969):** KDA input (6416×4096), KDA output (4096×2048, TX slot) and MLA output
(4096×4096, TX slot) run `mk_gemm_rows16_kernel`. `forward_pipeline=False` keeps the wide pack route as
the same-build control.

**Adopted second, same day (criterion change):** MLP gate/up (6144×4096), MLA qkv_a (2048×4096) and MLP
down (4096×3072, TX slot). #969 rejected them because they regress in the L2-warm replays. The operator
then ruled that the warm regime is not the target ("캐시따뜻한건 어차피 충분히 빠르니 느려져도 되지않아?",
"그냥 너가 도입해서 기본값 pr 머지해"). The deciding scope is now a chain of distinct layers with L2 evicted,
which is how a decode step reads them. All three win there (−1.7%, −8.2%, −4.4%), and their measured warm
regressions are accepted. They were restored from the measured source and were not re-measured before
the merge.

**Rejected and removed:** the joined 16-row DSA query grid, which is neutral when evicted (−0.4%). The
1536-wide query keeps the QueryPair shared pack.

The claim is component time only. The first three save about −0.3 ms (warm) / −0.4 ms (evicted) of dense
GEMM time per C=2 forward. The second three add +0.011 ms warm and save −0.039 ms evicted (sum of the
chain per-forward columns below). Together that is under 1% of a 55 ms step. No consumer step/s, onepass, acceptance or quality result is claimed;
that verdict comes from onepass with #964's C=2 arm.

## Method

`probes/engine_dense_cells.py` (lane `dense_cells` of `probes/engine_kernel_check.py`) replays every route
through the captured graph it serves in, input pack and output inside the interval, over the SAME RTN W4
packs of one build:

| Route | Kernel launches at 16 rows |
|---|---|
| `bound` (serving at measurement time) | KDA/MLA output, MLP down (TX slot), gate/up, qkv_a: `mk_wide_input_pack_kernel` + `mk_gemm2_kernel<2,…,PACKED_INPUT>` |
| `generic` | `mk_gemm2_kernel<2>` with in-kernel quantization (serving KDA input at measurement time) |
| `wide` | wide pack + `mk_gemm2_kernel<2,PACKED_INPUT>` |
| `pair` / `pair_generic` | QueryPair: wide pack + two packed `mk_gemm2_kernel<2>` / two `mk_gemm2_kernel<2>` |
| `rows16` / `pair16` (prototypes) | wide pack + `mk_gemm_rows16_kernel` / `mk_query_pair16_kernel` |

- **Numerical gate.** Every arm of a cell must reproduce the first arm's BF16 bytes at zero tolerance.
  Inputs: six magnitudes (0, 0.001, 0.1, 1, 50, 0) from a strided parent. Outputs are NaN-poisoned, both
  replay orders are run, and TX descriptors are rebound between steps with guard rows checked.
- **Timing.** Two B/A/A/B brackets per comparison. Warm samples replay 64 times; evicted samples flush
  128 MiB before each of 32 replays, outside the interval. `single` is one layer. `chain` calls distinct
  real layers back to back (KDA 0,1,2,4,5,6,8,9; DSA 3,7,11,15; dense MLP 0–2), which no L2 holds at once.
  The tables report the mean of the four samples per arm, and the minimum.

## The sixteen-row CTA

`mk_gemm_rows16_body` (`engine/kernels/dense/kernels.cu`):

- **Geometry.** One CTA per eight output columns. Each warp runs the ordinary 16-row lane's MMA,
  X[16] @ W[8], with `mk_gemm2_kernel<2>`'s operands in its order, reading the wide pack in its natural
  layout. Eight warps take independent K blocks.
- **Reduction.** The epilogue replays each column's per-slice FMA chain and slice sum (#946). With eight
  slices (KDA input) each warp owns one slice and accumulates it, like the C1 MODE kernels.
- **K slices.** Every shape keeps the ordinary lane's slices at 16 rows (`mk_choose_ksr2`): KDA input 8,
  outputs 3. The host refuses a changed plan.
- **Why eight columns.** Eight W rows per CTA keep the ring (9,216 B) and partials (≤16,384 B) within
  26,624 B, so three CTAs fit per SM. A sixteen-column CTA with 32 K blocks of 16×16 partials would need
  52,224 B and get one.

Resources:

| Kernel | Registers (cuobjdump) | Stack / local | Dynamic smem | CTAs/SM (device) |
|---|---:|---:|---:|---:|
| `rows16<false,32,8>` KDA input | 80 | 0 / 0 | 14,336 | 3 |
| `rows16<true,16,3>` KDA output | 74 | 0 / 0 | 18,432 | 3 |
| `rows16<true,32,3>` MLA output | 74 | 0 / 0 | 26,624 | 3 |
| `rows16<false,32,2>` gate/up (second) | 72 | 0 / 0 | 26,624 | 3 |
| `rows16<false,32,6>` qkv_a (second) | 72 | 0 / 0 | 26,624 | 3 |
| `rows16<true,24,3>` MLP down (second) | 70 | 0 / 0 | 22,528 | 3 |
| rejected `query_pair16<12,3>` | 72 | 0 / 0 | 16,384 | 3 |

The 14 existing C1 cta3/ordered/joined specializations keep identical resources in all three compile
records (`compile-main-9c45086a.json`, `compile-prototypes-01f5420c.json`, `compile-final.json`).

## GPU result — `c2dense-cells0915d`

Single-GPU lane, srv4 NVIDIA GB10, no model boot. GO 08:51:41, release 08:53:02 (81 s including build).
The fleet boot `selcal-trace0915` had released at 08:51:40.

- **Source.** Frozen checkout of `3c128c9b` (prototype tree, kernels.cu SHA-256
  `d7161de35a827ed2ecfeece62ab2635c5e3308dd145cb94a8015983389028cff`).
- **Image.** `sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5`, Torch 2.13.0+cu132, CUDA 13.2.
- **Weights.** Production rank `/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors`,
  38 BF16 tensors with their SHA-256 in the log; identical RTN packs in every arm, not the consumer GPTQ packs.
- **Artifacts.** Events: `gpu-c2dense-cells0915d.jsonl`; queue log: `gpu-c2dense-cells0915d.log`;
  tables: `python3 summarize.py gpu-c2dense-cells0915d.jsonl`.

**Numerical gate: PASS.** All 28 exact groups: 7 cells × single/chain × baseline/prototype. Every arm
matched the first bit for bit: generic = wide, bound = generic, pair = pair_generic, and rows16 or pair16 =
the serving route for all seven shapes. The adopted instantiations compile to the same register counts in
the final tree as in the measured tree.

### Adopted: sixteen-row CTAs against the serving route at measurement time

Chain = distinct layers; ms/forward = chain µs per layer × calls per forward (34 KDA, 11 MLA).

| Cell | Control | Scope | Warm mean / min µs (control → rows16) | Warm Δ | Evicted mean / min µs | Evicted Δ | ms/forward warm / evicted |
|---|---|---|---:|---:|---:|---:|---:|
| KDA input | generic | chain ×8 | 578.35 / 575.03 → 541.93 / 537.11 | −6.3% | 660.46 / 648.95 → 608.18 / 595.97 | −7.9% | −0.155 / −0.222 |
| KDA input | generic | single | 51.51 / 51.08 → 48.80 / 47.48 | −5.3% | 139.09 / 138.02 → 125.20 / 124.23 | −10.0% | |
| KDA output | wide | chain ×8 | 245.32 / 245.14 → 230.72 / 230.43 | −6.0% | 305.03 / 301.40 → 282.06 / 277.02 | −7.5% | −0.062 / −0.098 |
| KDA output | wide | single | 30.16 / 28.54 → 24.75 / 24.30 | −17.9% | 83.56 / 82.30 → 54.03 / 52.24 | −35.3% | |
| MLA output | wide | chain ×4 | 214.64 / 210.95 → 192.51 / 191.99 | −10.3% | 278.95 / 275.52 → 256.82 / 256.44 | −7.9% | −0.061 / −0.061 |
| MLA output | wide | single | 35.78 / 35.60 → 39.93 / 39.79 | **+11.6%** | 116.68 / 116.40 → 90.44 / 90.09 | −22.5% | |

MLA output is slower only in the L2-warm single-layer replay. A decode step reads eleven MLA outputs
between KDA and expert traffic, so its weights are not L2-resident. Both chain modes and the evicted
single layer favour the CTA, and it is adopted on that basis. The warm single-layer regression is recorded.

### Adopted second (gate/up, MLP down, qkv_a) and rejected (pair16): measured in the same run

| Cell | Control | Scope | Warm mean / min µs (control → candidate) | Warm Δ | Evicted mean / min µs | Evicted Δ | ms/forward warm / evicted |
|---|---|---|---:|---:|---:|---:|---:|
| MLP gate/up | wide | chain ×3 | 204.05 / 203.73 → 197.21 / 197.04 | −3.4% | 266.21 / 263.51 → 261.58 / 258.15 | −1.7% | −0.007 / −0.005 |
| MLP gate/up | wide | single | 33.27 / 33.19 → 49.78 / 49.66 | **+49.6%** | 128.45 / 127.60 → 122.90 / 121.41 | −4.3% | |
| MLP down | wide | chain ×3 | 94.47 / 94.21 → 97.16 / 97.11 | **+2.8%** | 188.71 / 188.37 → 173.24 / 172.84 | −8.2% | +0.003 / −0.015 |
| MLP down | wide | single | 34.27 / 32.34 → 34.08 / 32.62 | −0.6% | 101.07 / 100.36 → 72.49 / 71.55 | −28.3% | |
| MLA qkv_a | wide | chain ×4 | 68.78 / 68.47 → 74.17 / 74.05 | **+7.8%** | 159.75 / 158.06 → 152.70 / 152.28 | −4.4% | +0.015 / −0.019 |
| MLA qkv_a | wide | single | 21.39 / 20.30 → 21.29 / 20.62 | −0.4% | 60.21 / 58.88 → 50.86 / 50.46 | −15.5% | |
| DSA query pair (pair16) | QueryPair | chain ×4 | 131.03 / 127.26 → 139.94 / 139.52 | **+6.8%** | 203.88 / 201.70 → 203.03 / 202.17 | −0.4% | +0.025 / −0.002 |
| DSA query pair (pair16) | QueryPair | single | 29.12 / 28.79 → 30.66 / 28.84 | +5.3% | 78.61 / 78.00 → 67.81 / 67.36 | −13.7% | |

#969 rejected all four because each regresses in at least one chain mode or badly in a single mode, and
because the best per-forward effect of each is at most 0.02 ms. Under the evicted-chain criterion,
gate/up, MLP down and qkv_a win and are default. pair16 stays rejected: −0.4% evicted is within noise,
against +6.8% warm. The frozen source of all four is commit `46dafe93` / tree `3c128c9b`. The adopted
kernels.cu restores the three instantiations and `mk_run_gemm_rows16`'s checks and launch branches from
that source. `run_gemm_bound_input` widens #969's 16-row condition to the three shapes; the measurement
called the probe entry `run_gemm_rows16` directly. The six-kernel compile
record is `compile-rows16-six.json`.

### Kept: the existing 16-row routes against generic `mk_gemm2_kernel<2>`

| Cell | Serving route | Chain warm Δ of generic | Chain evicted Δ | Single warm Δ | Single evicted Δ |
|---|---|---:|---:|---:|---:|
| KDA output | wide | +0.2% | +0.5% | +6.4% | −0.7% |
| MLA output | wide | −1.3% | −1.2% | +8.8% | +0.9% |
| MLA qkv_a | wide | +2.9% | +0.3% | +2.0% | −1.2% |
| MLP gate/up | wide | −1.4% | −1.0% | +10.8% | −0.3% |
| MLP down | wide | +1.8% | −1.1% | +6.9% | −0.5% |
| DSA query pair | QueryPair (shared wide pack) | +1.5% | +1.6% | +8.4% | +2.4% |
| KDA input | generic → wide as candidate | wide −1.4% | wide −1.6% | wide −16.7% (noisy control) | wide −2.9% |

The wide cells are within ±3% of generic in chains and 6–11% faster in warm single layers. They stay.
The 1536-wide query keeps its shared pack through QueryPair: `bound_input_cell(16, 4096, 1536)` stays
False, as M14's (K=6) evicted regression left it (`measurements/st_decode_batch_20260913`). The M14 KDA input
warm regression (+1.9%) was a wide-pack result. At K=7 the KDA input takes the sixteen-row CTA, which beats
both generic and wide here.

## CPU evidence

- **Native compile/load.** Production flags, CUDA hidden, pinned image: `compile-main-9c45086a.json` (control
  source), `compile-prototypes-01f5420c.json` (measured prototypes) and `compile-final.json` (adopted tree,
  kernels.cu `d2cd19b7d4061b7899bfd730901871e7178b72555fc10d58f8d42dae47ac1575`). All PASS.
  `compile-rows16-six.json` is the tree that adds gate/up, MLP down and qkv_a (kernels.cu
  `c612d4453cb131956cf4798ef0eb70f6fd736f9a2a2e5263983b274fda542d6e`). It has six rows16 specializations.
  The new three have 72 / 72 / 70 registers and 0 stack/local, the same as the measured prototypes. The
  first three rows16 kernels and the 14 C1 cta3/ordered/joined specializations report resources
  identical to `compile-final.json`. Its focused CPU run added `tests.test_engine_decode_seven`,
  `drafter_decode_cells`, `dense` and `linear_family`: 74 tests, 46 passed, 28 GPU/Triton-only skipped.
- **Emulation.** `emulate_rows16.py` follows every address the kernel computes: tile-major W pack, swizzled
  ring rows, halfword exponents, natural X pack, partial layout and epilogue. Integer stand-ins are compared
  with the ordinary lane's arithmetic: 0 mismatches at eight shapes, including tile boundaries and the last
  CTA of 6416 (`emulate-rows16.txt`). `mutate_emulator.py` shows seven plausible offset bugs detected
  (`mutate-emulator.txt`). This checks index arithmetic, not device bytes; the GPU gate checks bytes.
- **Unit tests.** `cpu-tests.log`: 52 focused tests (dense routing, forward reduction/pipeline, DSA
  inputs, direct MHC and producer packs, kernel glue, W4A8 dataflow), 40 passed and 12 GPU/Triton-only skipped.

## Not measured

- Consumer step/s, tokens/s, acceptance or quality: pending onepass with #964's C=2 arm.
- NIC transport of the TX outputs.
- GPTQ consumer packs (RTN here).
- The adopted default through `run_gemm_bound_input`: the GPU run called the prototype entry directly.
  The post-merge confirmation ticket covers it at 8 and 16 rows. For gate/up, MLP down and qkv_a, it
  must be re-queued on the tree that adopts them.

## Commands

```sh
# CPU, pinned image, CUDA hidden (NVIDIA_VISIBLE_DEVICES=void)
python3 probes/engine_forward_reduce_compile.py --build-dir /out/build --output /out/compile.json --forward-pipeline --rows16
python3 -m unittest -v tests.test_engine_decode_fastpaths tests.test_engine_forward_reduce tests.test_engine_forward_pipeline \
  tests.test_engine_decode_dsa_inputs tests.test_engine_direct_mhc tests.test_engine_kernel_glue tests.test_engine_w4a8_dataflow
python3 emulate_rows16.py; python3 mutate_emulator.py

# GPU, single lane beside production, from a frozen checkout on srv2
ST_IMAGE=sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5 ST_PROBE_TREE=dense-cells-3c128c9b \
ST_DENSE_BUILD_ROOT=/cache/st-dense bash bench/fleet.sh run --gpu --detach c2dense-cells0915d 25 '<note>' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes dense_cells --seqs 2 \
  --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors --output /cache/c2dense-cells0915d.jsonl
python3 summarize.py gpu-c2dense-cells0915d.jsonl
```

## Post-merge confirmation

Pending: the same lane with `--seqs 1,2` against the merged main SHA.
