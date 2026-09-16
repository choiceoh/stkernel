# RTX 5050 head and FC cuBLAS comparison — 2026-09-17

Status of this initial comparison: **head wins, direct cuBLAS FC loses on RTX 5050.**
The subsequent [five-way batched FC implementation](../st_cublaslt_reform_20260917/README.md)
beats direct cuBLAS by 10.5–11.2% and DeepGEMM by 8.1–8.2% on the same device.
The original full-K results below remain historical evidence.
The user approved the comparison and explicitly offered the RTX 5050. This is a
device-specific exception to the older SM120 numerics-only rule, not GB10
admission. Production owns the four GB10s. No fleet ticket or service restart is
involved. `run_5050.sh` holds the direct host's exclusive GPU-probe lock and checks
for other ST probe containers; it is not a central queue reservation.

The comparison consumes FP8 PackStore bytes directly with `--packed-weight`;
it does not rebuild the weights from BF16. Inputs are seeded synthetic BF16,
shared between the two arms. This measures the warm captured quantization-to-output
component pipeline, not serving throughput, live activation numerics or acceptance.

## Result

| Component | M | DeepGEMM ms | cuBLASLt ms | Latency change | Decision on this GPU |
|---|---:|---:|---:|---:|---|
| Head | 7 | 0.679124 | 0.593184 | −12.65% | cuBLAS candidate |
| Head | 8 | 0.679513 | 0.594176 | −12.56% | cuBLAS candidate |
| Head | 14 | 0.682112 | 0.598208 | −12.30% | cuBLAS candidate |
| Head | 16 | 0.682914 | 0.598944 | −12.30% | cuBLAS candidate |
| FC | 8 | 0.344780 | 0.357848 | +3.79% | retain DeepGEMM |
| FC | 16 | 0.349719 | 0.361448 | +3.35% | retain DeepGEMM |

Each row uses its own B/A/A/B bracket. Head rows report the selected configuration;
FC rows report the lowest mean cuBLAS finalist, which still loses. Times include
activation quantization, scale production and BF16 GEMM output. Each sample is
4 replays of a 16-operation CUDA graph, with a separate warm replay. Compilation
and algorithm search are outside timing. No serving cells were replaced.

All six shapes enumerate 447 configurations, admit 4 and pass the initial
`allclose(rtol=.01, atol=.001)` numerical screen for all 4. The top 3 are paired
with 1/2/4-warp producers. All producers preserve the original FP8 and scale bytes.
Head chooses algorithm 70, tile 20, stages 36, zero workspace: 2 warps for M=7/8/16,
1 warp for M=14. Both paired head samples beat baseline by more than 2%.
The four selected cuBLAS head bindings pass changed-input graph replay twice and
allocate zero additional Torch GPU storage. FC's selected binding remains
DeepGEMM; its changed-input replay result therefore does **not** validate cuBLAS FC
replay. Its cuBLAS numerical evidence comes from the screening/brackets.

Evidence: [head](rtx5050/cublas-head.json), [FC](rtx5050/cublas-fc.json),
[derived summary](rtx5050/summary.json). This is a single 5050 window, with
synthetic activations and real packed weights. It establishes neither GB10 speed
nor end-to-end decode/acceptance. The direct WSL lock excludes other cooperating
ST probes, but cannot exclude Windows desktop GPU activity; utilization was 0%
and free VRAM 2613 MiB both before and after. No other GPU containers were running.

## Runtime and fixes exposed by execution

RTX 5050, SM120, 20 SMs; driver 595.79; Torch 2.13.0+cu132; CUDA 13.2;
cuBLASLt 130400. The immutable image and dependency hashes are in
[dependencies.json](rtx5050/dependencies.json). The x86 check image had no
DeepGEMM. Built the pinned official `nv_dev` revision
`572557e7ae9ad5331b81a1c250f141fba2c57962` in the owned scratch directory using
`DG_FORCE_BUILD=1 MAX_JOBS=2 OMP_NUM_THREADS=1 python3 setup.py build`, inside a
CPU-only container capped at 2 CPUs / 4 GiB. No installed image was modified.
The `sm120_fp8_fp4_gemm_1d1d.cuh` and `sm120_split_k_reduce.cuh` hashes match the
ARM production library exactly. The complete Python package and host extension
are not byte-identical to production.

Two existing preparation bugs prevented the original candidate from running:

1. MXFP8 heuristic queries require non-null scale pointers. The old plan only
   set them when executing. Plans now borrow validated real scale tensors during
   enumeration and clear those borrowed addresses afterward; execution bindings
   keep their own scale tensors. [cuBLAS diagnostic](rtx5050/heuristic.log).
2. Reusing a graph-pool token after resetting its last graph hit Torch 2.13's
   allocator live-pool assertion. Independent tuning captures now use independent
   pools. [initial failure](rtx5050/head-graph-pool-error.log).

Validation after these fixes: six GPU cells passed, a dedicated GPU regression
test passed (query scale lifetime, independent binding scales, invalid shape),
and 16 focused CPU tests passed. [GPU test](rtx5050/native-regression.log),
[CPU tests](rtx5050/focused-final.log).
Earlier producer validation is retained: 34 passes / 6 skips in
[producer-cpu.log](producer-cpu.log), and 76 SM121 variants in
[compile-sm121.json](compile-sm121.json). The producer source hashes still match;
that earlier host-binding compile predates the two preparation fixes and is not
proof of the final C++ binding. The final binding was built and executed on x86.

## Pinned inputs

Read from srv2's `/home/choiceoh/glm53-cache`, mounted read-only at `/cache`.
CPU-only PyTorch loaded the metadata and hashed the current rank-0 Hessians;
each selected pack's calibration digest matched. CUDA remained uninitialized.
These are current cache matches, not an attestation of the live process's pointers.

| Reader | Rows M | Padded N,K | Pack basename under `st-dense-packs/` |
|---|---|---|---|
| Shared target/draft head | 7,8,14,16 | 38784,4096 | `7e3c089b56122e41d05b7fcfc7163749d4f5a6eaa8f4186e2199fc71762d394e.pt` |
| Committed-decode FC | 8,16 | 4096,20480 | `e29b30a8de75e0e3e6fc6895abd857dac5a74910db589ab28f5306e98fc50009.pt` |

Head identity: `Glm5NextForCausalLM/lm_head`, source weight digest
`9af96ae356614f8b72ca969b16e72e5c2e1809eaf66bd4c9b47b8e84e47b0e65`,
calibration digest `a0d731b64b51ab03e70a1d27e9b1cb497275acffc4d7f89e4fea0959467d53dd`
(137244 calibration tokens).

FC identity: `DFlash2Qwen3ForCausalLM/outputs-5-14-24-33-42/model.fc.committed-decode-v1`,
source weight digest `dbf7c69cc7315f41ea8ebab6dedd907e68502464b1d63551bca2a02a27310f4b`,
calibration digest `1ac6cb2a21d748e013b66fa939ddcfe49a4022088285080596712cb2dadaee4d`
(33034 committed-decode calibration tokens).

The GPU receipt records the actual quantized tensor byte digest, original pack
identity, source digests, GPU identity and cuBLAS/runtime versions. Weight and MX
scale buffers are shared across the row shapes for each pack. CPU validation
checks byte preservation, digest identity and rejection of incompatible input packs.

Validation: both focused CPU tests passed, and the loader read both real packs
without initializing CUDA. [cpu-inputs.json](cpu-inputs.json) retains their actual
quantized tensor digests. These are input checks, not cuBLAS execution checks.

## Reproduction

`run_5050.sh` uses the explicit `--timing-target sm120-probe` and the same six
shapes below, with the two packs copied into the owned scratch `packs/` folder.
It expects the pinned DeepGEMM checkout's build under
`DeepGEMM/build/lib.linux-x86_64-cpython-312`. The production default timing target
remains GB10, and no SM120 result can set `gb10_admission`.

### Reserved GB10 payloads (not run)

These are payloads for an authorized, reserved GB10 window, not direct launch
instructions beside an occupied service. The normal runner's reservation and
memory checks still apply.

```sh
python3 probes/engine_cublaslt_check.py --gpu \
  --packed-weight /cache/st-dense-packs/7e3c089b56122e41d05b7fcfc7163749d4f5a6eaa8f4186e2199fc71762d394e.pt \
  --shape 7x38784x4096 --shape 8x38784x4096 \
  --shape 14x38784x4096 --shape 16x38784x4096 \
  --output /out/cublas-head.json
python3 probes/engine_cublaslt_check.py --gpu \
  --packed-weight /cache/st-dense-packs/e29b30a8de75e0e3e6fc6895abd857dac5a74910db589ab28f5306e98fc50009.pt \
  --shape 8x4096x20480 --shape 16x4096x20480 \
  --output /out/cublas-fc.json
```

Compare both paired B/A/A/B samples, workspace, FP8 byte preservation,
matrix output tolerance and changed-input graph replay. A fast component does
not establish engine speed or acceptance. Production traffic overlapping the
measurement invalidates an isolated timing claim.
