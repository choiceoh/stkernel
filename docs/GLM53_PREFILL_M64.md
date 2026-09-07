# GLM-5.3 full-chunk MoE M64 experiment

The current dynamic workspace chooses M128 from its maximum capacity and reuses
that tile for smaller calls. This experiment retains the original workspace and
adds an independent M64 workspace only when `VLLM_GLM53_B12X_PREFILL_M64=1`.
The default is 0. No measured speed or cumulative 40% improvement is claimed yet.

The candidate admits actual 6,144–8,192-token eager wrapper calls with E288,
H4096, I512, top-8, NVFP4, BF16 output, SwiGLU-OAI alpha=1/beta=0/limit=10,
and SM121. Short calls, capture, other geometry, forced static backend and the
functional API retain their original workspace. Failure to query capture state
also uses the original workspace. MoE is called once on the complete chunk;
transport, per-expert quantization scales, FC1/activation/FC2 and scatter remain
stock. Existing M128-only prefill reuse kernels are not eligible for M64.

The separately allocated workspace is bounded at 8,192 tokens even if the wrapper
has a larger capacity. It adds approximately 185 MiB of device storage at that
capacity, shared across layers by the existing wrapper cache. Allocation uses
M64 geometry for packed rows and tasks and still aligns the scale plane to
128-row atoms. Launch/compiled-cache selection consumes the stored workspace's
`tile_m`; the functional workspace cache and its selector are unchanged.

## Evidence budget and its limits

`measurements/glm53_moe_m64_20260908/padding-budget.json` retains the exact expert
counts and source hash from PR #453's completed diagnostic. These were synthetic
inputs, not a production routing trace. For balanced 6,912/8,192-token inputs,
M64 reduces padded MMA rows by 14.95%/11.08% with BF16 gather and 7.43%/6.92% with
FP8-v3 gather. Concentrated routing saves no rows. Smaller tiles increase the
number of tasks and weight loads, so this is an available arithmetic budget,
not a predicted speedup. The current source profile uses FP8 v3 for chunks of
at least 4,096 tokens; both transport modes must be tested.

## Validation

The CPU contracts execute the wrapper's allocation/routing code with fakes to
verify boundary rows, capture, error fallback, other geometry, bounded storage
and independent stock/M64 allocation. The inherited offline lifecycle tests
exercise exact container identity checks and recovery after partial failures.
The distributed API binding is checked against the frozen source before CUDA.

`probes/glm53_moe_m64_check.py` runs real TP4 balanced/concentrated inputs at
4096, 6143, 6144, 6912 and 8192 tokens in both BF16 and FP8-v3 transport processes.
It records every rank's candidate and independent stock-control per-row errors,
changed-input/route reuse, retained-output lifetime, local MoE before transport,
and changed-input capture/replay of the original M128 path. Original per-row
L2 .02 / peak .04 floors and three-times same-row stock repeat error are retained.
A stock-control failure is inconclusive and blocks serving; it never approves
the candidate. Six alternating timing samples use the slowest rank. These
component times are not full-model TTFT.

GPU work uses `glm53_offline_checks.py` inside a normal `fleet.sh run --gpu` boot
hold. It pins all source copies, preserves the 128 GiB disk and 16 GiB probe
memory guards, and restores the exact incoming containers in `finally`.
Only a passing GPU gate can proceed to a matched same-build fresh-cache
2K/32K/128K B1/A/B2 TTFT bracket and public recovery.

Status: PR #455 opened. GPU check1 (source `2e2ede9719c319f039cd95ec63ca0268b0d620d8`)
started under `moem64check10908` at 2026-09-08 01:14:58 KST; probe started 01:15:50.
The independent serving collector is prepared while that frozen check runs.
Its gate requires every rank of both transport reports, stock-control success,
changed-input reuse and M128 capture replay, unchanged composed source/profile,
and completed recovery before deployment. Its 10 failure-path tests and the
7 comparator / 4 fresh-cache / 4 memory tests pass. GPU and TTFT results remain
pending; no serving run has been submitted yet.
