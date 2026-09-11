# GLM decode adapter overhead — 2026-09-11

Baseline: PR #538 merge `3bae7c84d16dc4d308469952a21a4cf5fbe43b87`.
This change removes redundant work in the existing eager `Glm53Engine` adapter.
The model forward, attention/expert lanes, collective calls and KV layout are
unchanged. These results measure adapter components, not full-model throughput
or inter-token latency.

## Changes and compatibility

- Entirely greedy steps use argmax over a view of the decodable vocabulary.
  They avoid temperature/top-p uploads, float logits, softmax, multinomial,
  the discarded random result, and cloning logits to mask orphan token IDs.
- Decode builds one flat host token vector with the same sequence/slot/context
  segments, then uploads once. The prior path uploaded each sequence separately
  and concatenated the resulting CUDA tensors.
- Per-step generation limit checks use `len(tokens) - prompt_len`. The previous
  checks sliced the entire generated history twice per decoded sequence, making
  cumulative history bookkeeping quadratic in output length. `generated()`
  continues to return an independent result copy when collecting output.

An entirely greedy step now leaves the generator untouched. Stochastic and
mixed steps use the unchanged base sampler and preserve both tokens and ending
RNG state for the same starting RNG state. Cross-version replay of a greedy
step followed by stochastic sampling changes, because the previous adapter
consumed random draws even when it discarded all sampled tokens. The new
contract is tested explicitly, including greedy calls interleaved before
stochastic calls, tied logits, strided inputs and orphan masking.

## Measured results

srv1 NVIDIA GB10; torch `2.13.0+cu130`; BF16 logits, vocabulary 154,880,
temperature 0 and top-p 1. Medians below are wall microseconds per component
call. Sampling uses the full vocabulary with no padding, so it does not include
the additional clone savings available with orphan IDs.

| Component | Baseline µs | Optimized µs | Reduction |
|---|---:|---:|---:|
| Greedy sampling, 1 row | 605.03 | 65.39 | 89.2% |
| Greedy sampling, 6 rows | 601.05 | 65.27 | 89.1% |
| Greedy sampling, 24 rows | 880.51 | 66.15 | 92.5% |
| Decode input, 1 request, K=0 | 107.71 | 69.55 | 35.4% |
| Decode input, 1 request, K=5 | 108.98 | 70.67 | 35.2% |
| Decode input, 4 requests, K=0 | 213.63 | 75.85 | 64.5% |
| Decode input, 4 requests, K=5 | 221.17 | 78.32 | 64.6% |
| Count 128 generated tokens | 0.343 | 0.121 | 64.8% |
| Count 8,192 generated tokens | 15.599 | 0.145 | 99.1% |
| Count 65,536 generated tokens | 119.982 | 0.145 | 99.9% |

At 24 sampling rows, additional peak PyTorch CUDA allocation falls from
44,612,608 bytes (42.55 MiB) to 3,584 bytes (3.5 KiB), excluding the input logits
and other allocations already live before measurement. This is allocator
accounting for the isolated call, not total process or physical memory usage.
The operator trace contains one argmax and no softmax/multinomial on the new
greedy path. Four-request input assembly changes from four `aten::copy_` calls
and one `aten::cat` to one copy and no concatenation.

Each GPU component has 40 warmup calls and five rounds of 100 timed calls.
Baseline/optimized order alternates each round. Each sample waits for its end
CUDA event; wall measurements include that wait and event submission. CUDA
stream timings include host launch gaps and are not kernel-only times. The JSON
retains per-round wall summaries as well as aggregate medians/p95. Host count
measurements use five rounds of 2,000 calls; their statistics summarize round
averages, not individual-call tails. The prefix probe runs the actual decode
method and stops at `_forward`, before executing any model or cache preparation.

`q38-worker` was present in the initial host inspection and absent in the
post-measurement snapshot; this task did not change its service or settings.
Other work on the host was not controlled. The rounds show local repeatability,
not a fleet isolation/performance gate.
GPU load, clocks, and host scheduling can change absolute latency between runs.
The component reductions must not be added together or presented as a full
GLM model speedup. Full 45-layer quality, DFlash acceptance and fleet ITL were
not requalified by this optimization probe.

## Validation and evidence

- `engine-tests.log`: **86 passed, zero skips** on CUDA. This includes the
  previous lifecycle, paging, rollback, continuation, LocalTP and HTTP suite,
  plus six new tests for sampling and decode contracts.
- Ragged drafts with full acceptance, rejection, no drafts and limit clipping
  preserve flat token order, state slots, context positions, drafter auxiliary
  rows and completion flags. Long histories reset their generation count at
  conversation continuation, and collected results remain independent copies.
- Stochastic and mixed-temperature batches match the seeded base sampler for
  repeated calls at top-p 1 and 0.8, including a tiny positive temperature that
  rounds to zero in float32. Greedy tokens match the baseline in every measured
  vocabulary/batch shape.
- `sampler-selfcheck.log`: the existing greedy/seed/nucleus/multinomial checks
  pass unchanged.
- `overhead.json` and `overhead.log`: complete component measurements and
  operator counts. `source-sha256.json`: 82 tested Python source hashes, verified
  against the submitted source and baseline commit. `environment.log`: host
  and image metadata captured after the measurement.

## Reproduction

The baseline adapter is exported from Git, rather than keeping another runtime
implementation in the repository:

```bash
git show 3bae7c84:engine/profiles/glm53/adapter.py > baseline_adapter.py
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
python3 -m engine.base.sampler
python3 probes/engine_decode_overhead.py \
  --baseline baseline_adapter.py --output overhead.json
```

On srv1 these commands ran in an isolated copy at
`/home/choiceoh/st-engine-f4d7-optimize`, using image `glm53:v13-b12x-it`, ID
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The test and benchmark each used their own disposable container with
`--gpus all --memory=4g --memory-swap=4g --cpus=2 -e OMP_NUM_THREADS=2`,
the isolated directory bind-mounted at `/repo`, and the image's Python runtime.
