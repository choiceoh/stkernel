# Fused router tail alignment

The prior fused router changed three numerical details: the FP32 projection
association, selected-column tie order, and weight denominator association.
This change aligns the last two while retaining the one-launch projection.
It does not establish that the earlier combined consumer quality difference
was caused by the router, or that aligning the tail improves answer quality.

## Implementation

- The served runtime is PyTorch 2.13.0+cu132, source
  `cf30153c4c131c8164ee7798e5022d810682e2cb`, and Triton 3.7.1.
- The CUDA 288-expert top-8 path gathers scores above the boundary in expert-id
  order, then boundary ties, then sorts in a 32-entry bitonic network with
  padding. The aligned tail reproduces this network only when selected scores
  tie. Ordinary strictly ordered rows avoid the extra permutation.
- Compiling `glm_pointwise._weights` for SM121 confirms FP32 XOR shuffle steps
  4, 2, 1. The aligned tail uses those same additions instead of eight sequential
  additions. Accurate exp and round-to-nearest division remain unchanged.
- Arrival counters are separated by device and execution/capture stream.
  Graphs captured on one stream still share the counter and require ordered
  replay, as the engine's shared graph pools already do.
- `route_skip` capture experiments fall back to the common routing path so the
  fused early return cannot skip the requested mask/renormalization.

## Validation

Commit `538ad639`, session `router-align-rank2-0917`, GB10 on srv3 beside
production, 2026-09-17 12:09:36–12:10:59 KST. The initial srv4 ticket had no
memory room; its replacements retained its age. All 42 rank2 gate/bias hashes
match the prior rank3 probe. The measured source hashes are in `gpu.jsonl`.

- **24,768 row checks** (12 seeds, forward/reverse replay, C1/C2 single/42-layer
  chain): exact selection order and weight bits on the fused projection;
  unchanged projection bits between prior/aligned fused; control self-diff 0.
- Full served-versus-aligned selection **set and order mismatch 0** in these
  synthetic activation checks. Projection error remains up to 5.72e-6 and
  aligned weights can still differ from the full served path by 8.94e-8.
  Identical-logit equivalence is not whole-router bitwise equivalence.
- **40 finite edge cases**: rows 1/7/8/9/16, BF16/FP32, all ties, saturation,
  boundary ties, interleaved repeated scores. All exact against served CUDA.
- **64 independent-stream graph replays**, all bitwise stable.
- CPU: 24 tests, 4 GPU-only skips. CUDA compilation passed with no device
  initialization. BF16 C1/C2: prior/aligned both 128 registers, stack/local 0,
  shared 1,796/2,564 bytes. See `cpu-tests.txt` and `cpu-compile.txt`.
- CI run 35176444980 attempt 2: 260 files / 2,356 engine tests, 469 skipped,
  zero failures or unavailable modules; additional 137 onepass, 77 oracle and
  21 fleet tests passed. Attempt 1 failed the unrelated TP4 prefill test with
  `rank 3 failed`; its standalone 8-test file passed (`ci-prefill-recheck.txt`)
  and unchanged CI rerun passed. The initial failure is not silently waived.

## Cost: 42-layer chain, three B/A/A/B brackets per cache condition

| Rows | Cache | Prior fused us | Aligned fused us | Added us | Added % |
|---|---|---:|---:|---:|---:|
| C1 / 8 | warm | 1238.150 | 1250.717 | 12.567 | 1.015 |
| C1 / 8 | evicted | 1403.311 | 1415.686 | 12.375 | 0.882 |
| C2 / 16 | warm | 1621.493 | 1650.805 | 29.311 | 1.808 |
| C2 / 16 | evicted | 1790.374 | 1817.900 | 27.526 | 1.537 |

In the separate matched served/aligned bracket, evicted C1 is
3,196.533 -> 1,414.897 us (**-55.737%**), C2 3,255.079 -> 1,817.641 us
(**-44.160%**). Corresponding served self-control floors are +0.066%/-0.025%.
The aligned tail keeps almost all of the launch-fusion gain, at about
**0.0124 ms / 0.0275 ms added per 42 routed layers**. These are component costs,
not measured whole-engine step latency, throughput or answer quality.

Decision: keep alignment as the fused implementation's default. The projection
accumulation remains unchanged; restoring cuBLAS would give up the main fused
projection/launch saving and is not part of this change.

The consumer selector remains at its existing value during this implementation
comparison (`fused_decode_router=False`). `_align=False` is a private component
control, not a serving knob. Full-engine answer quality has not been rerun.

## Default adoption follow-up

After the alignment merged in #1112, the user requested enabling the serving
default and deploying it to production. The selector now defaults on for the
qualified GLM53 geometry (hidden 4096, experts 288, expert top-k 8, speculative
K7). Dispatch still requires prepared FP32 gates/bias, bound rows 8 or 16, and
no route-slot skip. Other widths and geometries use the common path; an explicit
`fused_decode_router=False` remains available for comparisons.

The measured CUDA kernel and wrapper are unchanged. This adoption reuses the
component evidence above; it is not a new whole-engine speed, acceptance or
answer-quality result. The earlier combined consumer quality difference is
still not attributed causally to the router.

Source references: pinned PyTorch
[gather](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/TensorTopK.cu),
[sort dispatch](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/Sort.cu),
[sorting network](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/SortUtils.cuh).
`served-weights.ptx` is the CPU-compiled SM121 Triton reduction used for alignment.

Default-selection validation: 54 CPU tests passed in the pinned x86 runtime
(`default-cpu-tests.txt`), covering geometry admission, bias arena accounting,
bound-row dispatch, explicit disable, route-slot skip, startup execution proof,
and execution plans. The first staging attempts omitted recipe evidence files;
after copying the missing repository fixtures, the same tests passed. No CUDA
source or wrapper changed for this selector adoption.
