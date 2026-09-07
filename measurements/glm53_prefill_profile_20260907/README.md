# Current GLM prefill attribution — 2026-09-07

**The profile confirms a narrow optimization target, not a global performance
ceiling.** The unpack/MHC-post work targeted by #439 occupies only **2.64% at
32K and 3.08% at 128K** on FP8-eligible chunks. MoE accounts for 28–32%,
MLA/indexer 16–20%, dense GEMM/quantization about 13%, and communication
about 12–13%. Communication and compute are effectively serialized in these
captures. No new kernel speedup or cumulative 40% result is claimed.

This diagnostic profiles the production settings observed at 15:59 KST,
not the settings assumed by a stale profile file. The mounted source is
`6f797df28c29e7e4cb606724e2416dbc1c5dcfcc`, manifest `0aca81454720`, and
all four ranks use image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Actual NVFP4 static scale is `0`; the earlier #439 bracket and the current
repository default use `16`. This difference is preserved and disclosed.

The complete measurement contract is in `measurement-plan.json`.

## Measured results

All six canonical requests completed: retrieval **18/18**, Korean corruption
**0/6**, no traffic or cache-isolation issues. The observed model-token counts
match all three requests per size and every rank trace: 32,545 and 128,559.
Every request used a distinct cache salt and recorded zero cache-hit delta.
The actual wire hash differs only because of salt; the unsalted request hash
matches across each control/profile/control trio.

| Context | Control before | Profiled TTFT | Control after | Profile / after |
| --- | ---: | ---: | ---: | ---: |
| 32K | 11.880 s | 10.647 s | 10.553 s | +0.89% |
| 128K | 42.055 s | 42.177 s | 41.855 s | +0.77% |

The first 32K control was slower on the fresh process. That comparison alone
does not isolate JIT, cache warmup or other first-request effects. Against
the later warm control, profiled first-token times differ by under 1% at
both sizes; this is one capture and one later control per size, not a
confidence interval. Trace export inflated *full answer completion time*
(43 s at 32K and 118 s at 128K), so those totals and the profiled decode
windows must not be used as normal-serving latency estimates.

The following percentages are mean per-rank occupied time divided by that
rank's instrumented prefill span. Raw per-rank values are in `analysis/`.

| GPU work | 32K | 128K |
| --- | ---: | ---: |
| MoE expert kernel | 32.08% | 28.15% |
| MLA attention and indexer | 16.40% | 20.42% |
| Dense GEMM and its quantization | 12.94% | 13.12% |
| NCCL collectives | 12.67% | 12.11% |
| KDA | 10.77% | 10.73% |
| All MHC | 4.95% | 5.06% |
| FP8 transport pack/unpack | 3.60% | 4.19% |
| Elementwise and copies | 4.15% | 3.91% |

Head prefill spans were 10.546 s and 41.962 s, with 224 ms and 831 ms of
GPU idle respectively. All four ranks have consistent category ordering.
Communication overlapped other categories for at most 0.011 ms at 32K and
less than 0.001 ms at 128K. This is observed overlap, not a proof that all
communication can be hidden: dependencies and stream scheduling still matter.

The 32K trace has six actual chunks: 6912, 6912, 6912, 6912, 2304, 2593.
The 128K trace has 18 chunks of 6912 and one of 4143. Each chunk is counted
once across auxiliary-stream annotations, and decode graph events are
excluded. The earlier decoding-step census is not used as a denominator.

## What this permits us to conclude

The standalone microbenchmark's scope (FP8 RS unpack plus MHC post, only
on FP8-eligible chunks) is 278 ms / 10.546 s at 32K on the head and 1286 ms /
41.962 s at 128K. Across ranks the shares are 2.61–2.68% and 3.05–3.13%.
A hypothetical 12.7% reduction of just that work corresponds to roughly
0.34–0.39% of total time with every other cost held fixed. This is a budget
illustration, not a new gain prediction: that microbenchmark was exploratory,
and its measured speedup need not transfer unchanged into serving.

This explains why #439's matched +0.74% / +0.27% results can be small even
when the fused path really executes. Routing it only for long prompts can
avoid short-request regressions but cannot make this target occupy a larger
fraction of a long prefill. The current profile is scale=0 while that bracket
used scale=16; they must not be presented as one matched performance pair.

**Next investigation: overlap communication with the token-independent MLP
work.** The current path synchronously gathers, executes the existing TP
attention/MLP and reduce-scatters before MHC; the measured lack of overlap
matches that structure. A bounded prototype would pipeline coarse MLP token
blocks with ordered collectives, preserving rank order, tensor lifetimes and
routing semantics. More launches or smaller GEMMs can cancel the saving,
so this needs direct serving evidence before promotion. No gain is assumed.

MoE tiling/pipeline changes and MLA KV-load scheduling are the larger compute
candidates (together about 48–49% of this profile), but a large share does not
establish spare hardware efficiency. Existing generic MoE reuse was neutral
or slower and FC1 N128 strongly regressed; existing MLA pair/group variants
also regressed. Re-enabling those same implementations is not a justified
next step. A new design must address the specific prior failure.

For scale: with all other costs fixed, eliminating the entire 12–13%
communication slice would imply only about 14% more throughput. A 40%
throughput increase requires about 28.6% less total time. Thus one small
fusion or communication-only tuning is not a supported 40% plan; substantial
progress would need several large components. The original campaign's
cumulative gain needs a separate matched original-baseline comparison.

## Capture and accounting

The canonical Korean onepass runs one combined retrieval request at each
of 32K and 128K. For each size the order is clean control, profile, clean
control. Model inputs and sampling are identical, with separate `cache_salt`
values to force full prefill. Wire and unsalted body hashes are recorded
separately. Prompt-token counts must match and cache-hit deltas must be zero.

Profiling stops asynchronously on the first streamed content/reasoning
piece. The answer continues to completion for the existing retrieval and
Korean corruption checks. Extra decode captured while the profiler stops
is excluded using explicit `execute_context_N(T)_generation_0(0)` GPU ranges,
not by dividing a mixed trace by an estimated number of steps. Duplicate
range annotations on auxiliary streams count each chunk once.

The analyzer reports kernel work sums, occupied interval unions, time where
only one category is active, overlap and idle separately. Category occupancy
may overlap: it is not an additive critical-path decomposition. Instrumented
times are compared with clean TTFT controls; they are not substituted for
serving latency or used as measured speedup predictions. Four ranks are
parts of one run, not four independent repetitions.

## Resource scope

Model settings, image and all mounted file hashes are verified across four
nodes before and after measurement. The diagnostic uses the previously
validated smaller KV allocation: 524,288 KV-token target, 415 blocks,
262,144 maximum length, private loopback port 18000. The public service uses
2,000,000 KV-token target, 1,056 blocks and 1,048,576 maximum length. Therefore
this is not full-capacity 128K acceptance. Every request has an all-node
12 GiB MemAvailable guard. Minimum observed headroom was 17.40 GiB on the
head and 18.42–22.55 GiB on the workers; all guard checks passed. Before boot, every node must have 128 GiB of
free disk space. The owned runner restores the public service in `finally`.

## Initial failed attempt

Fleet `pattr0907` acquired 15:59:45 and completed its measurement boot at
16:07:25. Before any model request, `/reset_prefix_cache` returned HTTP 404;
that route is development-only in this image. This was a harness error,
not a kernel failure. Public health 200 was restored at 16:12:18. The complete
failed scripts and receipts are in `failed-reset-api/`. No timing from that
attempt is used.

The corrected `pattr20907` acquired 16:18:34. It checks the profiler routes
and `ChatCompletionRequest.cache_salt` in live OpenAPI before spending a boot.
The installed cache-key implementation also confirms that request salt is
included in the first block hash. The retry preserves the original effective
GMU without applying the launcher's graph-budget correction twice.

## Completion and production recovery

All measurement requests ended at 16:28:56 KST. The runner restored the
public server by **16:33:32**, observed health **200**, and exited **0**.
`restore-verification.json` checks that every node's complete serving command,
image, manifest and mounted file hashes match the original production
snapshot, with no changed original environment value. The four formerly
absent #439 variables are now explicit defaults (`0`, `0`, `-1`, `-1`).
Effective GMU is restored to 0.6229, full capacity to 1056 blocks / 1,048,576
maximum length, and port to public 8000. No new optimization is promoted.
Final srv1 disk headroom was 266.58 GiB; head memory was 9.77 GiB at recovery.
The public-capacity 128K case was not rerun after restoration.

The complete raw job is durably copied to
`srv2:/home/choiceoh/glm53-logs/prefill-attribution-20260907-pattr20907`.
`archive.json`, `completion.json`, `trace-manifest.json` and the final
four-node snapshots provide the completion and provenance evidence.

## Reproduction

`tools/trace_prefill_attribution.py` streams each gzip trace and writes per-rank
JSON. `summarize.py` reconciles trace token totals, all three canonical request
records per size, cache evidence, quality and memory guards. Six CPU tests
cover overlap accounting, explicit prefill selection, duplicate stream ranges,
missing-range rejection, the first-token capture trigger and exclusion of
BF16 tail chunks from the narrow FP8 target budget.

Raw four-rank traces are retained in the ignored local `traces/` directory and
on srv2; the completed trace manifest records their locations and SHA-256
values. Derived per-rank reports and request evidence are versioned.
