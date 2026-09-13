# GB10 serving connections

The serving boot now selects a bounded greedy decode pipeline with
`decode_iterations=2` or `4` (default `1`). It reuses the existing target,
sampling, token-commit, observation and proposal chain. KDA state stays FP32.
Direct MHC and tiled prefill are already connected to the same profile;
`nvme_mapped_staging` connects the two NVMe tiers to shared mapped storage.
These options are experiments, off in the qualified production defaults.

| Option | Serving consumer |
| --- | --- |
| `direct_mhc=1` | Captured target decode consumes rank packets at immediate MHC boundaries |
| `prefill_project_tiles=1` | KDA prefill projects arrived tiles before the full gather finishes |
| `nvme_mapped_staging=1` | Conversation and prefix NVMe tiers share one CPU/GPU staging allocation |
| `decode_iterations=2/4` | Greedy requests use the boot-captured bounded pipeline |

The four options above can coexist. Direct MHC still excludes the separate
2+2 reduction-overlap experiment; they have different collective ownership.
The receiver optimization does not implement producer GEMM writes into a send ring.

## Bounded serving contract

- Capture the finite existing target shapes at boot, using retained child
  graphs and a private pool. No request causes a new capture. Every loop keeps
  the borrowed graph/pool and caller allocation owners until its last launch ends.
  Child graphs are appended to the active capture with explicit dependencies;
  CUDA rejects calling ordinary graph replay while a stream is capturing.
  Only the deterministic graphs used by this option retain their raw templates,
  and each is instantiated at boot for ordinary fallback replays too.
- Reserve the whole burst before publishing block mappings. The first
  iteration must fit; later ones stop before exceeding the reservation or
  captured context bucket. Replay uses the construction stream.
- Only one burst may be pending. Any row's EOS, generation limit or prefix
  crossing stops every rank after TP4 MAX agreement. The staged prefix state
  cannot be overwritten by another iteration in that burst.
- Copy every iteration's tokens, count, acceptance and original contexts into
  separate log rows; resolve them in order before admitting the next burst.
  Zero progress without completion fails instead of scheduling indefinitely.
  Streaming sees tokens when the burst retires, so inter-chunk delivery latency
  must be measured alongside throughput before promotion.
- Existing runner cancellation drains the finite burst before releasing its
  rows. Prefill, row changes and slot reuse follow the same drain/identity
  contracts. There is no concurrent raw CPU write to CUDA memory.
- Stochastic requests keep their existing draws and async pipeline. Rich
  sampling/min-token/reasoning constraints retain the existing scheduler gate,
  with the thinking-cap horizon enlarged for a possible burst. More than eight
  stop-token IDs use the ordinary path without truncating the set.
- String-stop requests use the ordinary pipeline so the host checks text at
  the existing cadence, without additional burst lookahead.

## Timing and records

The CUDA global timer records each iteration and its forward, sample, commit,
boundary, observation and proposal stages. The host stores these as separate
`gpu_iteration` latency rows (including positions, committed counts and accepted
counts), and feeds stage seconds to the existing metrics. Runner step/batch
counters count actual iterations. The death ring retains one launch-to-resolution
wall record per burst with its full token budget; it does not invent individual
host timings by dividing the burst time.

Native conditional-body validation rejects host callbacks, explicit event or
semaphore nodes, allocations and nested conditions. CUDA instantiation enforces
remaining body restrictions. Cross-stream captures encoded as dependencies are
not categorically rejected. All native and serving paths are still experimental.

## Validation

CPU tests cover C=1/C=4 equality with the existing pipeline, the full reservation,
prefix/EOS/bucket exits, row and slot reuse, reasoning caps, ordered result
retirement, zero-progress rejection, record counts and nanosecond/unit conversion.
The existing pipeline, runner, graph, boot, release, tier and package checks also run.
The combined ST-image suite ran 165 tests: 160 passed and 5 unrelated GPU tests
were skipped. A separate HTTP/sampling/burst suite ran 232 tests: 230 passed and
2 were skipped. `compile.json` records the original conditional executor build;
`compile-composed.json` records the child-graph composition build (41.71 seconds),
both without a CUDA context. After composition, 112 focused engine tests ran:
107 passed and 5 GPU-only checks were skipped.
The later host-stop/row-departure guards passed the focused 34-test
pipeline/burst suite. Mapped alignment and allocation accounting passed all
38 tier/boot tests. These are overlapping suites, not additive coverage totals.
After integrating main's DSA glue change (#819), the focused integration gate
passed 128 CPU tests and skipped 16 CUDA-only tests (the execution-plans module
was run separately after correcting its name in the invocation). The merge
changes the DSA helper portion of decode_graphs.py and net.py, so their whole-file
hashes differ from the earlier GPU reports. The native kernels, composition
bindings and burst protocol tested by the toy GPU gate are unchanged. This
integration does not substitute for a full real-weight TP4 qualification.

`probes/engine_bounded_loop_check.py` tests actual conditional graphs, native
commit, changing inputs and owner lifetime, plus the real serving adapter with
small deterministic GPU target/sampling child graphs. This single-GPU toy target
is not a real TP4 transport or model-quality qualification. All three GPU tests
passed on `st-bounded-decode0913r4`, source `2a7a18e9772c4bd45445fdab51eede224f74dbbe`
(49.52 seconds including preparation); `gpu-consumer.json` retains source hashes.
The prior attempt passed both native tests but rejected ordinary nested replay;
`capture-failure.log` records that failure before explicit child composition.
Full real-weight,
matched C=1/C=4 onepass acceptance, quality, length, tok/s and TTFT remain required
before changing production defaults.
Explicit seeds and an unmet min_tokens constraint retain the ordinary
scheduler path, including onepass's seeded fixed-length decode phase. A
bounded-decode performance verdict must additionally attest actual
`gpu_iteration` records on eligible greedy requests; an enabled option alone
does not prove that this pipeline executed.

The earlier execution-order bracket `st-gb10-orders0913v4` failed in baseline
C=1 preparation (stream timeout; last contexts repeated at 129775), before
candidate arms were measured. It supplies no performance comparison. The direct
MHC consumer's separate GPU gate passed 96 changing-input/address replays;
that is native arithmetic proof, not a full TP4 serving result.

## PR760 simulation

`simulate.py` uses C=1/C=4, 2K/32K/128K, 512 generated positions, seed 7 and
three repeats. Device cost is zero; the model is synchronous and generates no
language. This host measured 3 us Runner median at C=1 and 4-6 us in C=4
workloads. The 128K C=4 workload never reached width 4; observed widths are kept.

Hypothetically amortizing all modeled host work over four steps saves at most
2.25 us/step at C=1 and 3-4.5 us in these C=4 workloads. Torch submission,
existing async hiding, the new stop collective and real kernel time are absent;
result consumption still costs host work. This is neither a GPU win nor an
upper bound on the full serving opportunity. See simulation.json for the frozen
source hashes; the later serving connection is not simulated by those records.

CUDA handle/conditional constraints:
[Conditional graph nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes).
Graph-template retention follows
[PyTorch CUDAGraph](https://docs.pytorch.org/docs/main/generated/torch.cuda.CUDAGraph.html);
explicit capture dependencies use the
[CUDA stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).
