# Causal-prefix MLA and prefill output ownership

This follows merged PR #881 (`9f0b72f1`). It contains two changes:

**Operator enablement, 2026-09-13:** after PR #887 merged, the operator requested
the new kernel be turned on. Production and experimental boots now default to
`prefill_dense_prefix=1`; experimental rollback is `STK_prefill_dense_prefix=0`.
The CPU/compiler/Oracle records below remain pinned to the stated implementation
revisions. Default enablement does not constitute GPU performance/quality proof.

1. **Remove a redundant TP4 prefill output copy.** The served MLA wrapper receives one fresh output
   tensor at 16 heads/rank, then formerly concatenated that single tensor. Eager prefill now returns
   it directly. Decode/capture retains its previous path. The kernel, values and output lifetime are
   unchanged: consecutive calls still own distinct output storage.
2. **Dense-prefix MLA, enabled by default in serving.** The new Triton kernel handles the
   causal prefix where every complete kpool pool fits the selection width. In this checkpoint that
   is through length 2,051, including partial tails. Two adjacent queries (32 query/head rows) reuse
   each FP8 KV tile. The kernel follows each rank's page table directly and masks future positions.
   It needs no union discovery, membership buffer, new collective or resident cache representation.

The dense prefix and remaining sparse queries write into disjoint views of one final output buffer.
There is no full-output concatenation or additional full-chunk output allocation for the split. The
indexer still completes every key pool and tail update, so this composes with #881's query partitioning.
Steps under 128 rows, prefixes under 128 rows, multi-segment steps, probes and captured decode retain
their existing attention path. The option is declared on in production and production overrides
are refused. Runtime proof requires its execution on every DSA layer when enabled.

## CPU and compiler evidence

`9325fa9a`: **55 tests passed, no skips**, 17.842 seconds in the existing ST image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781` (`cpu-tests.log`).
The source was mounted read-only in a network-disabled runc container, CPU=2, memory/swap=4 GiB,
pids=256, `CUDA_VISIBLE_DEVICES=''`, `NVIDIA_VISIBLE_DEVICES=void`, with no GPU devices.

```text
tests.test_prefill_dense_prefix
tests.test_prefill_indexer_shards
tests.test_engine_knobs
tests.test_engine_execution_plans
tests.test_engine_prefill_tiles
tests.test_engine_prefill_outputs
tests.test_engine_native_execution
tests.test_engine_mla_hardware
```

The tests execute the actual Triton body through checked CPU loads/stores and compare it with an
independent full-softmax attention calculation. This covers 768-token/non-power-of-two page geometry,
permuted pages, nonzero layer offsets, nonunit KV scale, page boundaries, causal masks and ragged query
groups. Relative error stayed below 0.006 in those cases. This is CPU arithmetic evidence; it does not
validate GPU MMA lowering. Separate TP4 model tests use the semantic reference lane and compare hidden
states, auxiliary outputs, KDA/KV state and the following seven-token decode exactly, including 131/132
rows and layer-major 259-row execution with both new prefill options enabled. Served-wrapper tests
execute its real function, verify output identity/ownership and preserve decode/capture/multi-head behavior.

`compile-r5/result.json` compiles the actual kernel with the checkpoint's **block=768, stride=9,240
latent rows, scale=0.0625**, and every one of its 11 DSA layer offsets. `model-config.json` is the config
read from `/home/choiceoh/models/st-glm53-nvidia-tp4-9391/config.json`; its SHA-256 is in the report.
All **15 variants compiled**, CUDA remained uninitialized, elapsed 12.529 seconds.

| Geometry at layer 43 | Registers | Stack bytes | Dynamic shared bytes | Disposition |
|---|---:|---:|---:|---|
| 32 rows, 32 keys, 8 warps | 178 | 0 | 65,536 | Selected candidate |
| 32 rows, 32 keys, 4 warps | 255 | 192 | 65,536 | Rejected: spill |
| 64 rows, 32 keys, 8 warps | 255 | 24 | 98,304 | Rejected: spill |
| 32 rows, 64 keys, 8 warps | 252 | 0 | 98,304 | Unselected, larger footprint |
| 32 rows, 32 keys, 16 warps | 128 | 0 | 65,536 | Unselected; no timing basis to switch |

The selected geometry uses 178–180 registers across all layers and zero stack bytes. Static shared
memory adds 1,024 bytes, so the selected total is 66,560 bytes/CTA. It permits only one shared-memory
resident CTA per GB10 SM; the served tile32 kernel permits two. Saved KV loads can be offset by this
occupancy difference, additional MMA work for masked query positions and changed scheduling.

Merge `0e7fe03e` incorporates main `27fe4fff`, retaining its new arena admission reporting. Kernels,
attention modules, lanes, net, execution plan and prefill tests are unchanged from the qualified source.
The only subsequent change in the compiler's hashed files is main's two boot admission gauges.
`cpu-main-merge.log` records **39 passing focused post-merge checks, no skips**, 7.011 seconds.

## Source work counts, not a speed forecast

`work_budget.py` counts requested FP8 latent-row loads per rank and DSA layer at chunk=32,256.
It does not model cache hits, DRAM transactions, latency or TTFT.

| Input tokens | Dense-prefix queries | Reduction in all-query KV row loads |
|---:|---:|---:|
| 2,000 | 2,000 | 49.975% |
| 2,672 (historical 2K prompt) | 2,051 | 31.111% |
| 32,000 | 2,051 | 1.655% |
| 128,000 | 2,051 | 0.404% |

The separate single-output-copy removal applies throughout long prefill. At 128,000 tokens it removes
3.90625 GiB of explicit copy reads plus writes per rank per DSA layer, or **42.96875 GiB across 11 DSA
layers per rank**. This is tensor-copy payload arithmetic, not measured DRAM traffic or time saved.
Both transformations preserve resident KV layout and add no communication.

## PR #875 Oracle

The upgraded Oracle at `e2bfbb9a` was run against **main `27fe4fff` → candidate `0e7fe03e`**, with the
actual checkpoint config and `prefill_dense_prefix=1`. #881's indexer option is off on both comparison
arms, isolating this candidate; the CPU model test separately exercises their combination.

```sh
python3 /path/to/pr875/bench/storacle.py compare --tree /path/to/this/tree \
  --base 27fe4fff --candidate 0e7fe03e \
  --config measurements/st_prefill_dense_prefix_20260913/model-config.json \
  --ctx 2000,32000,128000 --width 1 --set prefill_dense_prefix=1 --json
```

`oracle-pr875.json` confirms the option, chunk=32,256 and zero resident cache/state layout deltas.
**Prefill timing delta remains null at every length** because this source has no matching GPU timing.
`paired-profile-template.json` binds the current source/settings/model identities; its blank durations
are deliberately not usable measurement evidence. The drafter geometry remains the Oracle's reference
assumption because no drafter checkpoint was loaded.

GPU numerical checks against the served tile32 lane, quality/acceptance, fixed decode and profiler-off,
cache-reuse-zero C=1 2K/128K consumer TTFT remain unmeasured. No GPU queue submission, baseline engine
build/boot or GPU launch occurred. Neither 3,300 nor 4,000 tok/s is claimed.
