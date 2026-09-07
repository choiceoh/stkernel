# GLM large-prefill MoE streamed FC2 candidate — 2026-09-07

Default-off candidate `VLLM_GLM53_B12X_PREFILL_STREAM_FC2=1`, separate from
MLA PR #439. No serving speedup or cumulative 40% improvement is claimed.

The previous attribution capture on source `6f797df28c29e7e4cb606724e2416dbc1c5dcfcc`
put MoE at 32.08% / 28.15% of per-rank prefill spans at 32K / 128K.
That is the whole MoE category, not all removable time and not the exact
eligible-chunk budget. Source evidence is preserved in PR #439 under
`measurements/glm53_prefill_profile_20260907`; the new candidate uses a
separate checkout based on `619cfec`.

## Change

The earlier FC1 N128 candidate retained four Q1 A/SFA register-fragment pairs
through all 32 FC2 output tiles. The new subclass retains N128 FC1 and its
Q0/route preparation, but loads one Q1 slice at a time in a dynamic FC2 loop.
The increasing-slice MMA order, quantization, BF16 conversions, barriers and
atomic scatter are unchanged. Shared Q1 slots remain A=[3,4,2,1] and
SFA=[3,1,2,0]; none uses A0, which aliases FC2's third B stage.

The existing tile-major adapter supplies production `STATIC_V2=t` weights
without a per-request conversion or a second resident weight layout. The
new lane is exact-gated to SM121, E288/K4096/I512/top8, NVFP4, SwiGLU-OAI
(1,0,10), M128/N128 tiles and **executed chunks 4096–8192**. Short chunks
and other contracts fall back. Old reuse flags cannot be combined with it.
A distinct cache suffix prevents reuse of a stock or old candidate artifact.
Private helper/source drift declines the candidate. `COMPILED` and
post-call `LAUNCHED` log markers are separate; serving proof uses only the latter.

N128 FC1's requested TMA payload per unsplit M128/I512 task remains
5.875 → 4.5 MiB (23.4% fewer bytes than stock). This is source accounting,
not measured DRAM traffic or a predicted prefill gain. Streaming repeats
shared loads; GPU timing must establish whether the tradeoff wins.

## Completed CPU evidence

Pinned image: `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Real CuTe compilation at M6912 / sm_121a on srv4, through fleet's explicit
CPU lane. Containers used runc, no GPU devices, network disabled, 2 CPUs,
4 GiB RAM cap; the check verified CUDA was never initialized. File/dump
hashes and the compiled binary's resource records are in `cpu-resources.json`.

| Arm | Layout | Reported registers | Stack bytes | Reported local bytes |
| --- | --- | ---: | ---: | ---: |
| Stock | tiled | 168 | 1680 | 0 |
| Prior N128 + four cached FC2 slices | row-major | 40 | 3504 | 0 |
| New N128 + streamed FC2 | row-major | 168 | 1744 | 0 |
| New N128 + streamed FC2 | tiled | 168 | 1744 | 0 |

These are `cuobjdump --dump-resource-usage` records of the DSL's own cubins.
**Stack footprint fell 50.2% versus the earlier N128 candidate, but remains
3.8% above stock.** This does not measure spill loads/stores, occupancy,
register use per warp role, or latency. In particular `LOCAL:0` does not
establish spill-free execution. A CPU-only unroll=1 experiment reported
1776 stack bytes and provides no established advantage; the candidate keeps
the existing unroll=4 scheduling for the GPU comparison. That experiment
is preserved in the JSON, and is not a runtime rejection.

The first auxiliary reassembly attempt failed because image CUDA 13.0 ptxas
accepts PTX 9.0 while the DSL emits PTX 9.3. The CuTe compile itself had
succeeded. Resource inspection was corrected to read the actual compiled
cubin; no altered PTX or downgraded target was used.

Local validation: 6680 logic checks, including 30 megakernel and 51 fleet
regressions, passed. Torch-dependent checks were explicitly skipped on the
Mac. Five new unittest cases execute actual dispatch/cache/proof logic.
Composed snapshot parity, Python syntax, shell syntax and diff whitespace
checks passed. GPU arithmetic and serving quality are separate pending gates.

## GPU and serving gates

`probes/run_b12x_prefill_stream_check.sh` runs through fleet's GPU probe
queue and selects a worker with the unchanged UMA guard if the head has
insufficient room. It pins the image, mounts composed source files, verifies
file hashes and uses isolated caches. It does not deploy or restart serving.

The same-process stock and candidate use identical tile-major weights and
all 288 experts. Cases cover 4096, 6912 and 8192 tokens, balanced routing,
eight-expert skew (multi-slice tasks), every-row eager comparisons, poisoned
output buffers and graph replay after changing activations and expert IDs.
2593 tokens confirms short-chunk decline. Numerical gates compare each row's
relative L2 and maximum absolute error with both stock and stock-repeat
atomic-scatter noise (floors 2% / 4%, or 3x measured row noise). Balanced
AB/BA graph timings retain every sample; memcheck and racecheck run separately.

This does not substitute for direct serving evidence. After numerical and
sanitizer success, run the planned matched prefill workload at 2K, 32K and
128K with all quality checks, exclusive traffic, verified candidate launches,
identical image/build/settings and fresh prefix-cache inputs. The prepared
`serving-plan.json` records all three TTFT objectives. It must be bound to the
then-verified deployment, model and hardware before submission; no deployment
or serving bracket is claimed by the CPU/GPU-probe artifacts.

## Queue preparation correction

The initial `moestreamprobe0907` request passed fleet preflight and queued,
but was cancelled **before GPU admission** after the separate MLA probe
revealed that the pinned runtime image has no `compute-sanitizer` executable.
No MoE GPU result was produced by that request.

The corrected runner mounts `/usr/local/cuda/compute-sanitizer` from the
selected host read-only, including its injection libraries. All four hosts
reported version 2025.3.1.0 and executable SHA-256
`7a7fcdefb67042731daf021478176f4919e1843d0b10cb697af28a7d8a3d108b`.
The mounted executable's `--version` also succeeded inside the pinned image
in a CPU-only runc container. The GPU runner repeats that check before
launching the numerical tests and logs its tool hash.
