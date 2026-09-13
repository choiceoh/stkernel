# K=7 decode fastpath qualification

K=7 produces 8/16/24/32 target verification rows. Several optimizations
qualified at K=6 still select only 7/14/21/28 rows, so their benefit is
absent at the new draft depth. This is remaining optimization work, not a
measured explanation of the whole engine's speed.

| Candidate | Current serving rows | K=7 cells to qualify |
|---|---|---|
| C1 W4 input pack reuse and CTA reduction | 6, 7 | 8 |
| Wide W4 input reuse | selected 14, 21, 28 cells | 16, 24, 32 |
| Paired KDA low-rank projections | 1, 6, 7, 14, 21, 28 | 8, 16, 24, 32 |
| Joined indexer projection and fused boundary | same | same |

Serving defaults remain unchanged. Native input mode 2 exposes the existing
pack/MMA at M8 only to the probe; mode 1 keeps the qualified rows. Projection
allowlists are expanded inside a scoped probe override and restored even
on failure. The short campaign expires on 2026-09-16. Winning cells still
need explicit promotion and the pending consumer measurement.

The actual rank's layout matters. The current ModelOpt rank stores its first
three dense MLPs as NVFP4, so those are excluded from W4 savings. Real C1 W4
families here are KDA in-projection (6416 x 4096) and MLA output projection
(4096 x 4096). The 6144 x 4096 native specialization retains synthetic
numerical coverage and receives real-weight timing only for a BF16-MLP rank.
Missing or mixed MLP declarations fail before measurement.

## Validation

- Eight CPU checks cover K=7 scope, weight-family selection, override restoration,
  failed-arm continuation, native resource reporting and ring-lane admission/accounting.
- `native-compile.json`: complete CUDA/Torch extension build on the pinned
  ST image, with CUDA hidden. Source SHA256 is bound in the report.
- `k7-triton-compile.json`: 11 SM121 compilations, including both input strides
  at each new KDA row count. CUDA was not initialized.
- GPU numerics and timings are pending. CPU compilation is not their proof.

The GPU bundle runs KDA commit and projection checks in separate processes,
bounded to 300 seconds each. A failure is retained and the other arm still
runs. The projection gate checks exact ordinary/input-reuse output, changed
graph inputs, strided parent inputs, poisoned outputs, and direct output
address changes with shared and private workspaces. Real-weight paired
KDA/indexer comparisons retain the established 0.0005 relative bound and
exact indexer query/effective-weight outputs. Timing uses captured B/A/A/B,
including pack/output costs; warm and 128 MiB-evicted W4/indexer intervals
exclude the eviction itself.

Canonical payload, using the exact consumer rank directory:

```sh
bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes k7_commit_bundle \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391
```

No full-model boot is part of this component bundle. The later consumer
remains one candidate boot, C1 twice at 32K/128K and C4 once at 32K, 5 GiB KV,
3072 total / 2048 reasoning tokens, and no fixed-decode repeats. Answer
grades are observations; decode step/s, generated tok/s, tokens/step and
acceptance are the requested metrics.

## PR #875 Oracle

Tool head: `95c27e6340159413c44a38eef9e54a84b13f9fc5`.
`oracle875.json` and its unfilled paired-profile template retain the exact
source/config fingerprints at 32K/128K and C1/C4. `--acc 0` is an explicit
timing-only scenario, not an observed acceptance estimate; its token-rate
outputs are not used. No new acceptance value is available.

The source comparison reports zero cache/state allocation change and an
unresolved total decode delta in all four cells. It sees the native source
edit but does not activate probe-only overrides or price their GPU effect.
It therefore cannot establish the benefit of these candidates, or a decode
performance ceiling. Component microseconds must not be pasted into its
whole-step timing fields or presented as an engine step/s improvement.
