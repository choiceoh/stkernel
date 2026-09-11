# Integration with PR #534

PR #534 merged while PR #535 was being validated. These results supersede the
initial runtime results in the parent directory for the integrated implementation.
The b12x expert lane, DFlash2 adapter, auxiliary hidden states, HTTP boot path,
and LocalTP main-thread dispatcher from #534 are retained.

The integration moves that adapter and drafter onto the block-major paged arena,
reserves absolute decode write horizons so rejected drafts reuse their space,
retains raw beta logits at the KDA lane boundary, and stops generation at the
first EOS or requested token limit. Rank order remains srv2, srv1, srv3, srv4.
The drafter's ring and persistent block table are included in the arena budget.

## Validation

- CUDA unit/regression suite: **55 passed, no skips**, `srv1/engine-tests.log`.
  Includes unequal absolute write horizons, drafter ring ownership, immediate
  completion and clipping accepted draft tokens, and the paired cache oracle.
- Without installed PyTorch: **40 passed, 15 skipped**, `cpu-tests.log`.
- Eight base self-checks passed, including LocalTP main-thread dispatch and
  object broadcast: `srv1/base-selfchecks.log`.
- Served raw-beta KDA passed six token-count/initial-state cases:
  `srv1/served-kda.log`.
- Four grouped scale shapes matched FlashInfer exactly, including a padded
  200-row case. Both physical bytes and the six-dimensional MMA view's values
  and strides match; the view aliases the original storage:
  `srv1/scale-mma-view.log`.
- Real layers 0 and 3, one rank per server, reference lanes: all four passed,
  `srv*/fleet-reference-rank*.log`. This run preceded the paired-oracle change;
  the reference expert's independently executed outputs matched exactly.
- Real layers 0 and 3, one rank per server, full served lanes including b12x:
  **all four passed**, `srv*/fleet-served-rank*.log`.

The model checks use 64 tokens, 32-token chunks, six-token verification,
rejection/continuation, and two generated results of lengths 3 and 1. Replicated
hidden states, logits and generated tokens must agree across ranks. All request
blocks and state slots must return. `engine-source-sha256.json` and the four
node manifests identify the final sources used by the CUDA suite and served run.
The reference run and KDA probe preceded only the expert-view/checker changes.

## Why the served check changed

The first b12x run exposed a missing kernel-module mount; `run_mk_probe.sh` now
mounts the full imported MoE family. After that, b12x rejected the flat saved
scale tensor because its dispatcher expects a six-dimensional MMA view. The
lane now keeps stable strided views over the presharded bytes; it does not
repack weights during a forward pass. `scale-mma-view.log` independently checks
that representation against the installed FlashInfer implementation.

The b12x static kernel uses unordered BF16 atomic scatter additions, so two
executions with identical inputs can differ in their final bits. A bitwise
whole-output comparison would confuse that arithmetic with a cache-address bug.
The paired cache check now executes the actual expert lane in both passes,
requires **bitwise-identical activations, selected experts and routing weights**
at every expert boundary, and forwards the first expert output to the following
layers. It then requires exact final hidden states. Actual expert repeat error
is reported separately. A regression test injects differences into each of the
three inputs and proves that the oracle rejects them.

Chunking, verification, rollback and generation use the unmodified served lane.
The attention error gates were not relaxed. Expert numerical differences are
reported; they are not independently qualified by this cache check.

## Reproduction and limits

The image, checkpoint and isolated directory are unchanged from the parent
README. New rank slices in `ranks-v534` were regenerated from the checkpoint to
include #534's folded/interleaved expert scales, retaining the aligned writer.
Production rank files and existing serving containers were not changed.

`run-reference.sh` and `run-served.sh` are the actual per-node commands. Dispatch
all four concurrently with srv2=0, srv1=1, srv3=2, srv4=3. Do not run two
`run_mk_probe.sh` invocations on the same checkout concurrently: overlay
composition rebuilds its shared `build/glm53` directory. The scale probe can run
in the image directly with the checkout mounted; it needs no composed overlay.

These tests cover a two-layer real-weight slice. DFlash2's ring/adapter contracts
are tested with a synthetic drafter; actual DFlash2 acceptance rates, full
45-layer onepass quality, graph serving, production throughput and ITL remain
unqualified. Log trailing whitespace is removed without changing diagnostics
or numerical output.
