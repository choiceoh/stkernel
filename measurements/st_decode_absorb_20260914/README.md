# Operator defaults and K=7 MLA decode contractions, 2026-09-14

The operator requested "항상 기본적용하고 추가개선". Experimental and production
defaults now enable `decode_dsa_inputs`, `decode_indexer_gate`, and the new
`decode_absorb_tiles`. CHARTER D11 records this selection policy. Experimental
rollback values remain zero; the bare `ExecutionPlan()` stays neutral for
reference callers. Enabling these paths is not a claim of measured speed or
acceptance improvement.

Base: `b02bff42dfa788fa742e6f4d6132cb957bd7a4d8` (merged PR914).

## Additional implementation

The two MLA contractions used head-major einsum outputs. The query consumer
then called `.contiguous()`, and the output projection's `.reshape()` needed
another layout copy. The new owner reuses the existing BF16-input/FP32-accumulating
absorb kernel with small decode tiles, directly producing token-major BF16.

- K=7 rows 8/16 use M tiles of 16; rows 24/32 use tiles of 32. N/K tiles are 64.
- Retain both original `kv_b` views, including the output slice's storage offset;
  no persistent weight repack or additional weight allocation.
- Bind the owner before capture and require both contractions at every DSA
  layer and every declared width in the native boot proof.
- This removes two layout copies per DSA layer: **22 copy launches per target
  forward** over 11 layers. Together with PR914, source counts are 44 fewer
  launches at C=1 and 66 at C=4. These are operation counts, not GPU timing.

The BF16 operands and output boundaries are unchanged. The new tile schedule
can change floating-point reduction order, so GPU numerical and acceptance
evidence remain pending. KV capacity and recurrent state precision are unchanged.

## Completed checks

- `cpu-tests.log`: 70 checks, 66 passed and four require a reserved GPU. Includes
  production/experimental defaults, immutable weight ownership, graph-width
  routing, both-side boot proof, existing DSA/head-gate integration, kernel-package
  CI, and the actual absorb kernel body evaluated against independent einsums.
  The latter covers every K=7 width, both weight offsets, padded/tail tiles and
  guard regions, including the new M=16/K=64 tile.
- `compile.json` / `compile.log`: cached full native extension plus SM121/PTXAS
  DSA specializations in the pinned runtime with GPUs hidden. All four new MLA
  decode cells passed; shared memory is 10,240 bytes for M=16 and 12,288 for M=32.
  Image: `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`.
  Torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0; CPU containers used 1–2 CPUs,
  2–4 GiB, runc, no network, `CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`.
- `oracle.json`: PR875 tool `95c27e6340159413c44a38eef9e54a84b13f9fc5`, no knob
  overrides. The source's defaults select all three paths. Covers 32K/128K and
  C=1/C=4. Layout deltas are zero; changed costs remain unpriced and every total
  decode delta is null. `paired-profile-template.json` is unmeasured.

## Batched GPU gate

The existing canonical `engine_kernel_check.py --lanes dsa_inputs --ranks ...`
now includes MLA absorption after the query-pack, latent-write and head-gate
checks. Its new component loads all 11 real `kv_b` weights, checks both
contractions at every K=7 width, poisoned outputs, changed signed inputs, both
graph replay orders and deterministic repeat replay. Numerical comparison uses
relative L2 <= 0.0005; graph timing includes the old layout copies in the base
arm and the same consumers' views in the candidate arm. Warm and evicted
B/A/A/B measurements remain separate, with eviction outside event intervals.

Replace only the previous owned waiting ticket through `fleet.sh --replaces`,
preserving its enqueue time and using one combined short reservation. No model
boot is part of this component command. Final consumer evidence still needs
step/s, tok/s, tokens/step and acceptance in the batched 32K/128K campaign,
with C=1 twice and C=4 once; answer grading is not the operator's adoption gate.
