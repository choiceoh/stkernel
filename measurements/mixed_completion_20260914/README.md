# Full mixed-expert component (M1b, PR #895)

M1b completes the routed and shared FFN for one eager rank. It is not connected to serving and adds no execution knob. Earlier S/P/M1a GPU reservations keep their original source revisions.

## Implementation

- `plan_cold` compacts **only** the remaining top-8 routes into M128 expert tiles. Every descriptor keeps the original expert, token and slot. Zero-weight routes remain work. One task retains all four intermediate slices. Every task appears in exactly one window, with a default quota of 48 tasks and an allowed bound of 1–128.
- A symbolic-row CuTe producer writes group-16 FP4, SFA and scatter metadata. `PreparedPrefillKernel` inherits the ordinary long-prefill compute body and only replaces Q0 initialization. The owner publishes prepared descriptors and a bounded `[head, tail)` before each launch. The accumulator is cleared once per invocation, never between windows.
- `begin` publishes a decode event after routed reduction and decode shared output. `advance` issues one bounded cold window. Its first call also packs all cold routes; the quota bounds MMA tasks, not packing time or preemption latency. `finish` waits in stream order for every window, adds all hot routes and the prefill shared expert, then records the final event. Early access, duplicate completion, stale identities, changed sources/weights/metadata and foreign streams are refused. Failed partial dispatch cannot be retried into the same sum.
- The explicit shared reference is BF16 linear → clamped SwiGLU → BF16 linear, identical in every probe arm. This does not qualify the serving dense reader selection. The M16/M32 hot partials and M128 cold accumulator have different rounding boundaries; full output comparison is required.
- Rebased on main `0d3d6d59` (#913). The ordinary static kernel AST matches that reviewed baseline exactly after removing the private frontend conditional. The new C1 SF6 staging and scatter-cache changes are retained.

## Reproduce and scope

```sh
python3 -m unittest tests.test_engine_mixed_experts tests.test_engine_mixed_completion tests.test_engine_moe_sf6_staging
CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a python3 probes/engine_mixed_completion_compile.py --output /out/compile.json
```

The compiler gate executes eight real CuTe/PTXAS/TVM-FFI builds: two producers and ordinary/prepared M16, M32 and M128. It also requires the 9240- and 32768-row dynamic requests to return the **same** compiled handle. CUDA must remain uninitialized. Offline build times are not fleet startup or first-build performance measurements.

Canonical 8 GiB GPU qualification, through an admitted fleet reservation:

```sh
bash probes/run_engine_probe.sh probes/engine_mixed_completion_check.py \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --ckpt-meta /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --samples 8 --output /cache/mixed-completion.json
```

The probe uses real L3 weights and ModelOpt scales, source/config hashes, 8/32 decode rows, 9216/9240/32768 prefill rows, controlled tail boundaries and the real router on synthetic activations. It checks every cold route's packed FP4/SFA/weights against the ordinary frontend, exact decode/hot partials, zero-weight overwrite, seven-task continuation, full routed/shared completion and input ownership. The decode routed aggregate keeps the M1a relative-max gate of 0.001. Full BF16 atomic output uses fixed relative-max 0.02 and relative-RMS 0.004 gates with baseline-repeat error reported. Chunked error reduction bounds validation scratch at 32K.

Native split, prepared split (hot quota zero) and mixed (quota 128) are measured in native/split/mixed/mixed/split/native brackets. Both **decode ready time** and **whole prefill completion time** include the explicit shared reference. CPU route transfer/planning, validation, metadata/workspace allocation and preparation are recorded separately; first-use compile is labeled. Repeated prepared measurements reuse identical inputs and do not hide the per-invocation preparation needed for changing routes.

GPU numerical and timing results are pending. No TTFT, token/s, acceptance, quality or serving speedup is claimed. M2/M3 still require the served shared reader, TP4 agreement/reduction, cancellation and reader lifetime, graph/scheduler integration, matched arrival traces and 32K/128K C=1/C=4 consumer measurements.

## Evidence

Pinned runtime: `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`, PyTorch `2.13.0+cu130`. CPU and compiler containers expose no GPU and use `--runtime=runc --network=none --cpus=1 --memory=3g`.

- `compile.json`: eight actual SM121 builds on `e2a5a9e9`, including ordinary/prepared M128; both dynamic handles are reused across 9240/32768 rows. No CUDA context was initialized.
- `cpu.json`: 35 isolated Linux modules on `b25b6ff1`, **305 discovered / 289 passed / 16 CUDA skips**, including the new task layout, ownership, completion ordering, main's SF6 staging and existing S/P integration.
- `cpu-followup.json`: all 10 completion tests pass on `43d5816b`, adding a chunked-metric numerical test. Replacing the earlier 9-case module yields **306 discovered / 290 passed / 16 CUDA skips**, without counting repeated cases twice.
- `source-continuity.json`: all compiled device sources are byte-identical at the final code revision. The host owner's executable AST is unchanged; its only edit after compilation is a quota-scope docstring. The output-comparison helper and its extra test are covered by the follow-up gate.
- `admission.json`: the canonical GPU reservation `st-mixed-completion0914v1` was accepted/queued with ticket `17893453911662551` on clean source **`43d5816b`** in its own checkout. That accepted checkout is frozen; later evidence commits do not alter it. GPU results remain pending.
