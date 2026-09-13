# Decode projection, indexer and wide-row input bundle

The serving candidate combines KDA low-rank projection pairs, joined indexer
projections and a fused indexer boundary, plus qualified wide-row W4 input
reuse. The indexer copy costs 22 MiB/rank, explicitly carved after input
smoothing and included in admission and runtime budgets. FP32 indexer head
projection, BF16 materialization boundaries and recurrent-state arithmetic
are preserved.

## Component proof

Canonical single-GPU ticket `st-decode-batch0913v3`, source `c07f7641`, ran on
srv4 with all real rank0 projection families. It waited 1.1 seconds and used
36.6 seconds of GPU payload time. All 66 numerical cells passed, including
changed-input replay and poisoned outputs. The 48 wide-row cells matched the
ordinary W4 output bit for bit. Indexer queries/effective weights matched
exactly; projection comparisons use the established relative-error ceiling
of 0.0005. The native CUDA source matches the GPU-hidden full compile report.
29 CPU tests passed, 40 GPU-only tests were skipped, and 15 Triton/PTXAS
specializations compiled without initializing CUDA.

These are component timings, not engine speed or speculative acceptance:

| Component chain | Rows | Reference ms | Candidate ms | Change |
|---|---:|---:|---:|---:|
| KDA pair, 34 layers, warm | 7 | 0.38334 | 0.23974 | -37.46% |
| KDA pair, 34 layers, warm | 28 | 0.36231 | 0.27473 | -24.17% |
| Indexer pair + boundary, 11 layers, warm | 7 | 0.87298 | 0.75296 | -13.75% |
| Indexer pair + boundary, 11 layers, evicted | 7 | 0.91721 | 0.80403 | -12.34% |
| Indexer pair + boundary, 11 layers, warm | 28 | 0.46352 | 0.40333 | -12.99% |
| Indexer pair + boundary, 11 layers, evicted | 28 | 0.55195 | 0.46048 | -16.57% |

Wide input reuse is selected only at M14/M21/M28 for N,K cells
(4096,2048), (2048,4096), (4096,4096), (6144,4096), (4096,3072), and
at M21/M28 for (6416,4096), (4096,1536). M14's KDA input warm regression
and 1536-wide input evicted regression remain on the ordinary route.
Private-workspace shared-expert overlap retains its existing entry point.
All raw B/A/A/B samples and selected-cell context are in
`component-timings.json`; they include packing and output materialization.
Cold intervals use captured external CUDA events after a 128 MiB eviction,
so the eviction's bandwidth and host enqueue time are excluded.

## Earlier attempts retained

`st-decode-batch0913` used 29.2 seconds. Paired projections passed, but the
new timing wrapper attempted graph replay inside capture, which CUDA refused.
`st-decode-batch0913v2` fixed that and passed 54 boundary/wide-input cells in
33.4 seconds. Its cold timing included eviction cost; it is superseded by v3
for timing selection. Neither failure was a consumer reboot.

The final v3 incorporates main's shape-bound projection widths and fused
head-gate reference. Subsequent serving selection changes only the Python
dispatch to the same qualified native entry point.

## Consumer measurement

Candidate-only canonical onepass is pending. Intended coverage is C1 twice
and C4 once on one boot, 32K/128K, with answer grading retained in raw records
but excluded from the user's performance decision. A 3072-token completion
ceiling with 2048 reasoning tokens bounds this speed sample. It does not
establish natural completion length or matched quality against older runs.

`prior-onepass-lengths.json` records the previous two C1 passes requested by
the user: source 8c8b031b, FP16 KDA storage. Reasoning reached approximately
4096 tokens for individual questions and 12288 for combined questions.
Their final answers were 152-617 and 1575-1775 tokens respectively. Channel
lengths were re-tokenized from saved SSE content using the served tokenizer;
total completion counts are server-reported and include control tokens.
