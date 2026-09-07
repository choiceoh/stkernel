# MoE overlap retry: BF16 passed, FP8 fallback failed

Session `moeoverlap20907` received the normal four-node GPU hold on
2026-09-07 at 23:57:28 KST. Probe source was
`ce71af658a7c95ab2b23e04612fb7de39f05a64c` at
`/home/choiceoh/stkernel-moe-overlap-check2-0907` on every node.
The probe ran 23:58:21–23:59:59 and exited 1. Exact incoming containers,
image/config/mounts/source and public health were restored at 00:02:58 on
2026-09-08; supervisor exit 1 followed at 00:02:59. No candidate serving
source was deployed and no full-model MoE TTFT was measured.

All four source/API/memory preflights passed. The BF16 process completed all
eight size/routing cases, including changed inputs, retained outputs and
allocator churn, with `MOE_OVERLAP_GPU_PASS`. FP8-v3 passed the two 4096
fallback cases and 4143 balanced, then failed the 4143 skew eager comparison.
The 6144 minimum means **overlap was not admitted at the failing case**:
all three compared calls use the same stock MoE/transport fallback. FP8
active-overlap cases were never reached. This is not yet a candidate-specific
numerical defect, nor is it evidence that the FP8 candidate is correct.

The synchronized failure list reports one bad row on rank 0 and two on
rank 3; ranks 1 and 2 reported none. Rank 0 maximum row-relative L2 was
0.006479, maximum peak-normalized absolute error 0.068376 and baseline-repeat
L2 0.008243. Rank 3 values were 0.007127, 0.065574 and 0.007207. The gate
uses the maximum of fixed per-row tolerances (0.02 L2 / 0.04 peak) and
three times the corresponding observed repeat difference. A nonzero
baseline-repeat difference is observed, but it does not establish the
cause or justify widening the tolerance. Output atomics and subsequent
FP8 quantization are hypotheses requiring component-level evidence.

## Completed BF16 kernel/transport timings

These are six alternating-order samples per arm, aggregated by slowest
rank. They are synthetic MoE plus shared-expert/transport timings, not
full-model prefill TTFT. Speed depends strongly on routing distribution.

| Tokens | Routing | Stock median ms | Overlap median ms | Reciprocal time gain |
| --- | --- | ---: | ---: | ---: |
| 6912 | balanced | 18.537408 | 25.772336 | -28.07% |
| 6912 | skew | 15.131856 | 13.587072 | +11.37% |
| 8192 | balanced | 20.065392 | 28.111199 | -28.62% |
| 8192 | skew | 17.478976 | 16.548128 | +5.63% |

The substantial balanced-route regression must be explained; the skew
wins alone are insufficient to promote the option. Additional tile-padding,
stock-versus-serial-stripe and communication/compute span evidence should
separate extra MoE work from overlap benefit. The current failed GPU gate
correctly prevents serving deployment.

The next diagnostic must preserve the failing fixed seed, shapes and
thresholds, distinguish stock-repeat from candidate error, record per-rank
failure context/rows, and isolate repeated MoE partials from FP8 transport
replay. Do not rerun the unchanged full gate hoping for a different result,
relax tolerances, or claim a serving gain from these timings. A changed
probe/runtime requires a new frozen revision and normal fleet admission.

The eight original rank logs, combined probe log, parsed BF16 result,
all-rank API/lifecycle/resource evidence, failure summary, supervisor and
exact recovery completion are retained here. `VLLM_GLM53_PREFILL_MOE_OVERLAP`
remains 0 by default. `summary.json` is the machine-readable current verdict.
