# Compact KDA state: compilation evidence

**CPU compilation passed; GPU correctness, component timing and serving performance are pending.** The serving cache does not select the compact ABI.

The candidate stores a committed FP32 state and a separate prefix-boundary state in one slot-major arena. At GLM K=7, the recurrent payload is 68 MiB instead of 272 MiB per request per rank, excluding factor workspace, padding, the null slot and all other caches. This is layout arithmetic, not a measured engine memory or tok/s result.

## Reproduction and identity

- Controller: `srv2`; GPU devices were unavailable to the compile container.
- Image: `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`.
- Runtime: Torch `2.13.0+cu130`, Triton `3.7.1`; target `cuda`, SM121, warp size 32.
- Isolation: `docker run --runtime=runc --cpus=1 --memory=3g --network=none`, `NVIDIA_VISIBLE_DEVICES=void`; no `--gpus` flag.
- Entry: `python3 probes/engine_kda_deferred_compile.py --compact-only --output /out`.
- The script asserts `not torch.cuda.is_initialized()` and reports `gpu_used=false`.
- The two JSON files retain source fingerprints, cubin hashes, compiler shared-memory metadata and raw `cuobjdump` resource output.

[Initial source snapshot](compile_a1a84c8f.json) and [current source at c386ee21](compile_c386ee21.json) each compiled eight variants: verify T=1/T=8 × ordinary/compact addressing, and commit C=1/C=4 × ordinary/compact addressing. Source hashes, rather than branch names, identify exactly what was compiled.

## Observed compiler resources

| Kernel | Initial compact | Current compact | Current ordinary addressing |
| --- | ---: | ---: | ---: |
| Commit C=1/C=4 registers | 64 | 56 | 48 |
| Commit compiler `metadata.shared` bytes | 2,048 | 512 | 512 |
| Verify T=8 registers | 130 | 130 | 130 |
| Verify T=1 registers | 254 | 254 | 254 |

The current ABI uses one arena pointer and proves alignment for both record offsets. The first version passed a redundant boundary pointer and did not communicate the boundary offset's alignment. All recorded variants report `LOCAL:0` and `STACK:0`. The compact commit still uses more registers than ordinary addressing. These compiler properties do not establish runtime occupancy, exact numerical behavior or a speedup.

Local CPU regression command after rebasing onto `15bd6e3a`:

```bash
uv run --with torch --with numpy python -m unittest \
  tests.test_fleet_onepass tests.test_engine_deferred_state_contracts \
  tests.test_engine_kda_compact tests.test_engine_kda_deferred_batch -q
```

Result: **31 passed, 10 CUDA tests skipped** (41 discovered). Python compilation and `git diff --check` passed. GPU tests must run without skips before the numerical gate can pass.

## Remaining GPU gate

The admitted probe is `engine_kda_deferred_check.py --compact-only --samples 8`. It compares actual outputs and committed/boundary states with the full FP32 ring before timing 34-layer ordinary / deferred-ring / compact-state graph replays. It retains raw samples and distinguishes state-reset and 64 MiB eviction conditions. Allocation, compilation, graph capture and state resets are outside timed replay.

Two preparation attempts did not obtain reservations because the candidate needed recent main changes. After rebasing onto `71306bda`, session `st-kda-compact0913v3` was accepted into the queue with ticket `17893082511214606`. It uses output `/cache/kda-compact0913v3.json` and the image ID above. The submitted checkout is `6c4b47b4`; compiled source fingerprints still match. The [admission response](admission.json) is a queued-state snapshot, not a completed result. No other session's reservation is interrupted.

The current exclusive fleet owner is a separate requantization campaign. This file records preparation and compilation, not a completed GPU run. No onepass result exists for this ABI because the prefill/checkpoint/restore/stage serving adapter is a subsequent change; its requirements are in the [architecture design](../../bench/ST_GB10_ARCHITECTURE_20260913.md).
