# Prepared decode-tail expert component (PR #895)

Implemented M0 admission and M1a hot-route computation for the fixed GLM TP4 SF6 pack. There is no serving selector or new execution knob.

This directory preserves the original M1a scope and frozen admission. The subsequent cold/shared completion implementation and its separate validation are recorded in [M1b evidence](../mixed_completion_20260914/README.md).

- Decode 1..8 rows retain M16; 9..32 retain M32. The planner admits only complete M128 tails fitting existing decode tiles, preferring fewer added routes and bounding the total at 128. Decode order and every remaining top-8 route are retained, including zero-weight routes.
- A separate CuTe producer reads explicit `(local expert, expert row, decode/prefill, original row, route slot)` descriptors. The two BF16 sources are never concatenated. Per-expert group-16 NVFP4 quantization and SFA addressing match the ordinary static frontend. Symbolic input/route row extents share one producer build.
- The prepared V5 body skips initialization/routing and writes four independently owned, BF16-rounded weighted FC2 partials per route. Its compute body is unchanged. Ordinary kernel AST comparison is canonical across Python versions; baseline kernel AST was taken from the rebased main body.
- One eager stream owns each invocation. Layer/epoch/slot/source generation, tensor versions and source layout are validated. Graph capture and foreign-stream invocation are refused. The owner records its last reader event and retains both sources, actual weight views and metadata.

The component does **not** finish a prefill FFN. Cold M128 compaction/execution, shared output, cancellation/slot reclamation, TP4 agreement and serving scheduling remain follow-ups. The reported removed tile count is conditional work accounting, not executed tile savings. No TTFT, output tok/s, acceptance, quality or speedup claim is made.

## Reproduce

CPU coverage is `tests.test_engine_mixed_experts`, plus fleet admission, existing S/P integration and changed main decode tests. CUDA tests remain explicitly skipped in CPU evidence.

Compile without exposing a GPU, with the pinned ST runtime and `CUDA_VISIBLE_DEVICES=`:

```sh
python3 probes/engine_mixed_experts_compile.py --output /out/compile.json
```

This builds five actual CuTe/PTXAS/TVM-FFI handles: one symbolic producer, ordinary and prepared M16, ordinary and prepared M32. No fake compiled handle or CUDA context is used.

The canonical 8 GiB GPU qualification is:

```sh
bash probes/run_engine_probe.sh probes/engine_mixed_experts_check.py \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --ckpt-meta /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --samples 8 --output /cache/mixed-experts.json
```

Run it through an admitted fleet reservation. It uses the local rank's actual L3 weights, original weight/config hashes, checkpoint routing scale, ModelOpt input/down scales and synthetic BF16 activations. Cases cover 8/32 decode rows, 9216/9240/32768 prefill rows, controlled M128 tail boundaries and actual router decisions on synthetic inputs. It checks ordinary versus prepared FP4/SFA (including unequal scales), exact decode/hot route-part contributions, zero-weight overwrite, input immutability and stale-generation/mutation guards. Native aggregate error and baseline repeat variance have a fixed relative-max gate of 0.001.

The timing arms are native decode, prepared decode only, and prepared decode plus hot routes, each including the probe's output reduction. Host preparation is separate and includes validation, CPU route transfer/planning, metadata/workspace allocation and cold-route enumeration; initial compilation is labeled. Cold/shared work is explicitly marked incomplete. These samples can expose decode interference but cannot establish total workload benefit. Future adoption requires cold completion and matched 32K/128K C=1/C=4 consumer measurements.

Existing compact KDA and packet FFN admissions keep their frozen source revisions. This component uses a separate checkout and reservation.

## Recorded validation

- `cpu.json`: 33 isolated Linux modules at `79a554e5`, 288 discovered / 271 passed / 17 CUDA skips. No GPU exposed and no CUDA context initialized.
- `cpu-followup.json`: the final host resource-freezing change at `f8151e61` passes all 9 mixed-expert tests. Replacing the prior 8-case module gives **289 discovered / 272 passed / 17 CUDA skips** without double-counting repeat tests.
- `compile.json`: five real SM121 builds at `79a554e5`. All compiled source files remain byte-identical at `f8151e61`; only the host owner and its tests changed, covered by the follow-up CPU gate. `source-continuity.json` records both sets of hashes.
- Pinned CPU/compiler image: `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`, PyTorch `2.13.0+cu130`.
- Exact code HEAD `f8151e61` passed [PR CI](https://github.com/choiceoh/stkernel/actions/runs/34789934124/job/103812159001). Python-version AST formatting and unguarded CUDA-only observe tests were repaired before this pass.
- GPU reservation `st-mixed-experts0914v1` was accepted/queued with ticket `1789342553296863` on source `f8151e61`. `admission.json` records the separate frozen checkout, image, command and log. No GPU result is recorded yet.
