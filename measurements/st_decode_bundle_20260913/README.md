# Five decode candidates in one GPU reservation

The previous consumer completed both C=1 repeats and one full C=4 run on
`8c8b031b`. C=1 window medians were 19.896 / 19.891 step/s and acceptance
44.260% / 44.549%. It has no baseline on this build and remains below 22 step/s.
`prior-concurrency-throughput.json` recomputes the C=4/C=1 end-to-end ratios from
that boot's retained consumer record: 1.66–1.71x at 2K, 1.50x at 32K, 1.41x at
128K. These include prefill and are not pure-decode scaling or candidate gains.

The next component reservation contains five independent, opt-in lanes:

| Lane | Candidate | Required comparison |
|---|---|---|
| `mhc_single` | One token per CTA group; fewer live registers | Exact changed-input graph replay, 89 distinct coefficient packs |
| `moe_waves` | Existing resident-wave scheduling for M7 | Real TP4 L3 packs, 8–56 active experts, FP32 scatter tolerance |
| `input_pack` | Two/four rows per input packing CTA | Exact M6/M7 outputs at three N shapes, strided inputs, complete projection timing |
| `short_gemm` | CTA-local K reduction for N4096/K2048 | Original M6/M7 K slices and exact output, complete projection timing |
| `shared_direct` | Read the published shared-MLP activation directly | Exact M1/M6/M7/M8 output and scratch rearming, complete gate-up/activation/down timing |

`probes/engine_kernel_check.py --lanes decode_bundle` launches each lane in a
separate bounded subprocess under one canonical probe container and reservation.
A failed candidate preserves its failure and does not discard later lanes.
Each lane finishes numerical qualification before balanced B/A/A/B timing.
The dense/shared timings use distinct packs beyond L2; MoE retains both warm and
64-MiB-evicted samples. These are component comparisons, not engine throughput.

`compile.json` and `controller.json` retain the actual full CUDA/Torch extension
build and load from `a606f65d`, with no GPU initialized or opened. The integrated
CUDA source has the same SHA-256. All new emitted kernels have zero local bytes;
single-token mHC uses 80 registers versus 128 for the packed baseline. Eleven
focused CPU tests and Python syntax/whitespace checks passed after integration.
GPU numerical results and timings remain pending; all five serving defaults
remain unchanged until those results select a candidate.

The cancelled two-lane ticket never reached a GPU: its controller added
`ST_LEASE_OWNER` and `ST_LEASE_PATH` after preparation. `queue-environment.json`
reproduces the resulting receipt mismatch through the actual controller class,
then verifies identical environments when the same canonical owner and path are
bound before preparation. No authentication or actual lease check is disabled.

The improved consumer will run C=1 twice and C=4 once on one boot. The operator
excluded answer grading from this performance decision; original grading remains
recorded. No fresh baseline boot or separate state-rounding experiment is planned.
