# Fixed K7 follow-up: scoped mHC, MoE and MLA defaults

The candidate is `3c3c3260ba8a26929d0f466a90db9e9589c2f278`. The matched
control is `0e949d11d0421784f2681aa21f5ba7465662a776`; it changes only
three Python booleans. Native sources, packing, weights and build inputs
are identical between arms. Whole-engine validation is pending; component
improvements below are not a tok/s claim.

## Selected execution

- mHC: retain the eight-row KDA input-pack fusion. The original BF16
  rounding, FP8 scale and consumer bytes are unchanged. Thirty eligible
  boundaries per rank were proven in the earlier TP4 campaign.
- MoE: at eight/sixteen decode rows, load each BF16x16 activation scale
  group with two aligned 128-bit loads. Keep all sixteen FP32 conversions,
  the maximum reduction, quantizer, expert scale and output accumulation.
  This uses the existing vector-load helper; there is no new weight copy.
- MLA: sixteen rows and selection width at least 128 use 32-slot tiles
  with retained BF16 query fragments and the existing split count (six on
  GB10). Eight-row, narrow-width, tree and cluster cells keep their old
  dispatch. Tile32 changes FP32 grouping; independent FP32 error checks
  pass, but this is not bitwise equality.

## Bounded measurements

All GPU tickets used the canonical fleet reservation on srv2, the ST
image (Torch 2.13.0+cu132, CUDA 13.2), and GB10. Compile-only containers
have no GPU exposure and assert no initialized CUDA context. Hashes and
native binaries are recorded in each JSONL. The rank file is
`/home/choiceoh/models/st-glm53-9391-up-gate-full/rank0of4.safetensors`.

| Candidate | Evidence | Finding |
|---|---|---|
| MLA register queries, tile16 | `mla-v8.jsonl`, commit `9a1890f0` | 42 fixtures bitwise; no useful latency gain; removed |
| MoE byte/nibble/plain bulk and linear shared stages | `moe-v8.jsonl`, `9a1890f0` | Corrected a double-tiling error in the probe. Byte-swizzled z and linear zl pass; zn/zp fail. z is near-neutral; zl is slower at C1. Remove zl. |
| MLA tile32 | `mla-mhc-v10.jsonl`, `705083ea` | 42 independent-reference fixtures pass. C2 normal full selections improve 3.6–6.7%; C1/short rows do not. |
| MLA split plans | `mla-mhc-v12.jsonl`, `ed11409d` | Six splits retain the C2 win (60.03 → 55.67 µs, −7.26%). Other full-selection plans lose. Keep C1 unchanged. Plans requested beyond workspace capacity were clamped by `mla_splits`; they are not larger allocations. |
| MoE vector input | `moe-v11.jsonl`, `36e8f03d` | Three real layers; independent/shared/calibrated routes, poisoned outputs, zero route, forward/reverse replay pass. Frontend falls 13.488 → 12.696 µs at C1 and 18.840 → 18.048 µs at C2. C2 evicted three-layer chain −0.84%; C1 chain +0.06%, no engine claim. |
| mHC C1 fusion | `mla-mhc-v12.jsonl` | 90 real coefficients, four magnitudes, packet/ordinary forms and pack bytes pass bitwise; packet coefficient chain −19.19%. Full mHC + three KDA projections is only −0.36%. |
| mHC C2 extension | `mla-mhc-v12.jsonl` | Same numerical coverage passes; coefficient chain −5.08%, but mHC + KDA projection chain **+3.94% slower**. Removed rather than adopting the isolated mHC win. |

The v9 tile prototype stopped with illegal memory access: its local K-quarter
formula doubled the number of score planes. v10 fixes the geometry and adds a
compile-time warp-coverage assertion. The v10 record has no completion marker:
its final sixteen-row projection correctly refused an unextended inner guard.
v12 extends that guard and completes all numerics/timings; its later measured
C2 regression is why that extension is absent from the final code.

## Consumer gate

The previous mHC-only campaign is documented in
`../st_fixed_k_cost_20260917/README.md`: both arms failed some quality checks,
so its raw request rates are not evidence for adopting this three-way candidate.
The new matched full onepass must retain 2K/32K/128K, C1 twice, the serving
capacity once, quality, acceptance, actual request tok/s, TTFT and build identity.
If capacity remains two, C4 is unmeasured and must be stated explicitly.
