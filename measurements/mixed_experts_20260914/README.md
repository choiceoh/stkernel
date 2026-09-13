# Prepared decode-tail expert component (PR #895)

Implemented M0 admission and M1a hot-route computation for the fixed GLM TP4 SF6 pack. There is no serving selector or new execution knob.

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
