# Large-prefill MLA: register Q and 32-slot tiles

This candidate rewrites the sparse MLA computation for actual chunks of
**4,096–8,192 rows**. It preserves the selected KV slots, their multiplicity,
BF16 query inputs, FP8 latent storage and per-row attention semantics. It
changes the reduction and probability-rounding order, so it is not bit-exact.
`VLLM_GLM53_MK_MLA_PREFILL32=0` remains the default pending device and serving
evidence. No measured speedup or cumulative 40% result is claimed yet.

## Target and implementation

The completed production profile on source `6f797df` found the old transport
fusion scope occupied only 2.64% / 3.08% of 32K / 128K prefill. The new target
is the MLA attention kernel itself. After the short-chunk guard, its head-rank
occupied-time budget is **1,269.95 ms / 10,546.36 ms (12.04%)** at 32K and
**6,127.88 ms / 41,961.92 ms (14.60%)** at 128K. This is about 4.6–4.7 times
the earlier target, not an assumed removable latency.

The 32K guard admits four 6,912-row chunks, 44 MLA calls, and leaves the
2,304/2,593-row tails on the existing kernel. All 209 MLA calls in the 128K
capture qualify (18 chunks of 6,912 and one of 4,143). See
[target-budget.json](target-budget.json) and the [original profile](../glm53_prefill_profile_20260907/README.md).
That profile used actual dynamic NVFP4 scale `0`; it is not a baseline pair
for the repository default scale `16`.

The existing kernel processes 16 KV entries per softmax iteration, loads Q
fragments from shared memory repeatedly and assigns rows to a persistent
grid. The candidate:

- Keeps the 64 packed BF16 Q-fragment registers across the complete row.
- Computes 32 selected KV entries per iteration, halving softmax and CTA
  synchronization rounds. Two K partitions and four N groups cover the tile.
- Uses two asynchronous FP8 buffers and 39,296 bytes of shared memory; it
  does not materialize a gathered BF16 KV tensor or allocate split scratch.
- Launches independent row CTAs, letting CUDA schedule ragged rows without
  a grid barrier. Short requests, small tails and captured serving paths
  retain their existing dispatch. The pair experiment takes precedence if
  both options are supplied, so the experiments do not combine.

The implementation is in
[glm53_megakernel.cu](../../overlay/modules/glm53_megakernel/glm53_megakernel.cu)
and its [driver](../../overlay/modules/glm53_megakernel/glm53_megakernel.py).
Both GLM and shared DSV4 composed files are synchronized; the new GLM option
remains disabled in both unless explicitly requested through its driver gate.

## Validation status

- The verbatim device kernel compiled for `sm_121a` with host CUDA 13.0:
  **128 registers, zero stack, zero spill stores and zero spill loads**.
  This proves device compilation/resource allocation, not extension ABI,
  achieved occupancy, correctness or speed. See [cpu-compile.log](cpu-compile.log).
- Six executed dispatch regressions cover actual chunk sizes, short/tail
  boundaries, storage checks, capture, disabled mode and pair precedence.
  They are included in the normal megakernel deployment test discovery.
- Repository logic and composed-source checks pass locally; Torch-dependent
  macOS checks are explicitly skipped. Full current output is retained in
  [local-logic-current.log](local-logic-current.log).
- The GPU gate is queued through the fleet. It compiles the complete
  extension, compares full outputs against the same-build baseline and a
  sampled FP32 reference, checks every row rather than only a global norm,
  and covers empty/poisoned rows, repeated selections, tile tails, 6,912/8,192
  rows, repeated execution, and graph replay after changing device inputs.
  Memcheck and racecheck follow in the same isolated container. An existing
  memory guard selects an admitted node before any GPU process starts.
- Paired kernel event times are diagnostic. A positive result still needs
  a direct matched serving bracket at 2K, 32K and 128K, with identical source,
  image and settings, fresh cache salts, actual token counts, retrieval,
  Korean corruption, memory and path-engagement proof. No default promotion
  follows from the queued probe alone.

The final queued gate is `mla32probe30907`, source
`4be95a59ad9dab650a2fef22f065531338d70cef`, in the private detached checkout
`srv2:/home/choiceoh/stkernel-prefill32-check-0907`. Its durable runner log,
eventual `completion.json` and `exit_code` live under
`srv2:/tmp/glm53-mla32-check3-0907`. The submitted request and initial
preflight/queue receipt are [gpu-request.json](gpu-request.json) and
[gpu-fleet.log](gpu-fleet.log). Two earlier waiters were cancelled before
GPU execution while the admission and per-row checks were completed; there
has been no GPU result from those revisions. The final job has passed
preflight and is waiting behind other fleet work. The direct serving bracket
has not yet been submitted; it depends on this numerical gate.

Commands:

```bash
REPO=$PWD bash bench/fleet.sh run --cpu mla32compile 3 'MLA resource check' -- \
  python3 probes/mk_mla_prefill32_compile.py
REPO=$PWD IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211 \
  bash bench/fleet.sh run --gpu --probe mla32probe 10 'MLA device gate' -- \
  bash probes/run_mk_mla_prefill32_check.sh
```

MoE is the larger remaining compute target (28–32% of this profile). The
previous Q0/FC2-reuse and paired-N128 versions already regressed in direct
serving; their retained-register cost must be addressed in a new design.
Their old switches are not evidence of unused, immediately available gain.

## First GPU result and remaining sanitizer gate (17:36 KST)

The 11 eager/numerical and changed-input graph cases passed on srv2 with
source 4be95a59ad9dab650a2fef22f065531338d70cef and the pinned image.
At eligible executed chunks 4143 / 6912 / 8192 and W2048, the paired kernel
speedups were 1.0181x / 1.0204x / 1.0184x. These are kernel microbenchmarks,
not direct serving TTFT and not a 40% result. All sample timings and row
errors are preserved in gpu-numerics.json; raw output is gpu-fleet.log.

The runner then exited 127 because compute-sanitizer is absent from the
runtime image. Neither memcheck nor racecheck ran. The corrected runner
mounts the host sanitizer installation read-only and supports --sanitize-only
so the completed timing matrix need not be repeated. All four hosts have
version 2025.3.1.0, executable SHA-256
7a7fcdefb67042731daf021478176f4919e1843d0b10cb697af28a7d8a3d108b; its
--version succeeded inside the pinned image without exposing a GPU.
The candidate remains off pending sanitizer and direct-serving evidence.

Sanitizer-only retry `mla32san40907` passed preflight and queued at 17:56 KST
behind the new MoE probe. Frozen source: e74b15e5a846827e554835ca73eda71e7698e4ac;
checkout: srv2:/home/choiceoh/stkernel-prefill32-check4-0907; supervisor PID
2554596; logs/completion: srv2:/tmp/glm53-mla32-check4-0907. The GPU kernel
is unchanged from the 11-case numerical/timing run.
