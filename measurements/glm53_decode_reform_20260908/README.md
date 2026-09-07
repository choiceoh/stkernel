# GLM-5.3 integrated decode tile reform — 2026-09-08

The complete candidate is `VLLM_GLM53_B12X_STATIC_V2=t,r`. The default remains
`t`. The user requested several expensive paths changed together and one
combined final test; there is no per-feature speed selection.

| Geometry / per-item work | t | t,r for M<=8 |
|---|---:|---:|
| Padded M rows | 32 | 16 |
| FC1 N/K | 64/512 | 128/256 |
| FC1 halves per 128-column intermediate | 2 | 1 |
| FC1 gate/up pipeline stages consumed | 32 | 32 |
| FC1 TMA A/B/SFA/SFB bytes per stage | 8/16/4/4 KiB | 2/16/2/2 KiB |
| FC2 N/K | 128/128 | 256/128 |
| FC2 output tiles per H=4096 item | 32 | 16 |
| Compiled shared memory | 101,376 B | 98,304 B |

These are geometry and requested-transfer counts, not measured DRAM savings or
latency predictions. Weight payload bytes and the per-128-column BF16 rounding
boundary remain the same. Larger M uses the original t executable geometry.

The one-M-warp arrangement requires restoring separate N/K axes in the FC1
scale fragment. FP4 intermediate byte stores use the actual swizzled layout.
Compile-time checks cover every MMA output coordinate and packed byte address.

## Completed CPU gate

`cpu/m6.log` and `cpu/m16.log`: exact serving image, device-free sm_121a,
2 GiB memory cap, no GPU exposure, max_rows=640. Both t and t,r compile; M16
t,r specializes to the original t geometry. `cpu/logic.log`: 6,701 checks,
30 megakernel regressions and 107 fleet regressions pass. CPU compilation is
not a numerical or performance verdict.

## One integrated GPU campaign

Fleet maintenance session `moereform0908` owns:

1. Stock/baseline/candidate numerical differential, five graph replays with
   changed inputs/routes and exact-zero route weights, M=1/2/6/8/16/32.
2. Focused M6/U40 memcheck and racecheck in the same maintenance campaign.
3. One A-B serving comparison, same source/image, TP4, SPEC_K=5, C=1,
   three fixed 2K decode requests per arm and the 2K/32K/128K prefill ladder.
4. All-rank source and active M6 lane proof before/after each arm, private
   endpoint isolation, separate reasoning/content channels, exact request hashes.
5. Unconditional approved-main restore, public health, default knobs and
   all-rank source verification before releasing the reservation.

The two serving arms are `MOERFA1` (t,r), then `MOERFB1` (t). This is one
candidate boot and one baseline boot: within-boot windows are correlated and
boot drift is not independently measured. Report pooled step/s and output
tok/s separately from speculative acceptance.

Pending GPU results; no speedup claimed.
