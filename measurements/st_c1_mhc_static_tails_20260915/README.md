# C=1 MHC static tail ownership: accepted (2026-09-15)

## Result and implementation

**Accepted by the user** after reviewing the measured step rate, acceptance
and prefill observations ("저정도면 그냥 후보 수용해"). The qualified static
eight-row path remains enabled by default. The queued additional candidate
boot was cancelled; adoption is not conditional on more benchmarking.

The native gate passed. The 89-boundary interval falls **8.11% warm / 8.43%
evicted** for ordinary AR inputs and **8.77% / 8.81%** for local rank packets.
Both independent captures improved in every measured case. All output fields
match bitwise, including mixed row counts and the FP32 fallback. These are
component results. The observed whole-serving C=1 step rate is essentially
unchanged; no whole-engine 8% speedup is claimed. The user accepted the
candidate with the evidence below before a complete paired TP4 repeat.

### C=1 serving observations reviewed for acceptance

| Metric | Dynamic baseline | Static candidate | Observed change |
| --- | ---: | ---: | ---: |
| Decode step/s, inside-answer window median | 19.896189 | 19.897425 | +0.0062% |
| Raw speculative acceptance | 51.0719% | 53.6997% | +2.6278 percentage points |
| Generated tokens/step | 4.0643 | 4.2220 | +3.88% |
| 2K TTFT, median of three fresh requests | 1.0892 s | 1.0796 s | -0.88% |
| 32K TTFT | 9.4927 s | 9.6728 s | +1.90% |
| 128K TTFT | 37.3635 s | 36.9422 s | -1.13% |
| Final-answer correctness | 9/9 | 9/9 | preserved in this sample |

Sources: `onepass-control-cold.json` (baseline `78c3f5ed`, run
`20260915T061115-e904f00ebe5d`) and `onepass-incomplete.json` (candidate
`b2e59673`, run `20260915T055558-6151c6353435`). Both C=1 intervals are prepared,
profiler-off, free of observed JIT/capture and exclusive, with no prefix-cache
reuse. Actual input lengths are 2,627 / 33,826 / 129,784 tokens. TTFT includes
first-token generation; it is not an isolated prefill-kernel timer.

The candidate record ends early during subsequent C=2 preparation, as detailed
below. These are one-boot C=1 observations, not a complete paired full-run
benchmark. The latency-recorder fix in baseline `78c3f5ed` and accepted source
`fcd32f13` changes accepted recording widths, not the measured C=1 kernel.
Fresh random prefix salts can change generated continuations; the higher
acceptance rate is an observation, not an established effect of this change.

`adoption.json` records the decision, source identities and exact arithmetic
behind this table. The baseline's already-running second pass also completed:
**9/9 final answers**, **19.897906 step/s**, **50.6767% acceptance**
(`onepass-control-warm.json`). There is no matched second candidate pass.
`full-control.log` records normal four-rank cleanup and fleet release;
`full-candidate-cancelled.log` records the queued candidate cancellation.

Raw records are in `gpu.jsonl` and `queue.log`; `components.md` is generated
by `python3 measurements/st_c1_mhc_static_tails_20260915/summarize.py FILE.jsonl`.

The packed C=1 consumer has eight rows and 48 resident CTAs: three groups of
16 hidden-dimension chunks. Two groups process three rows apiece and the third
processes two. The candidate gives the third group's first eight CTAs one
tail each. Phase-one projection, per-token chunk arrivals, Sinkhorn, mixing,
normalization, rounding and the PDL dependency are unchanged.

Each launch removes 56 shared tail-ticket and 48 exit-ticket atomic updates.
This is a count from the source, not a measured latency claim. The 128 chunk
arrivals and each token's wait/reset remain. The native host gate requires
hidden size 4096, eight rows, lossless BF16 coefficients, an AR or direct-packet
consumer, and exactly 48 resident CTAs. All other shapes keep dynamic tails.

The downstream PDL wait remains in place. NVIDIA documents that this wait
holds dependent work until the upstream kernels complete and flush their
global-memory results; removing the unused dynamic exit-ticket reset does
not remove that dependency. See the [CUDA PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Source `f772d319` on base `5871c559` qualified native `tail_mode=0/1/-1`
(dynamic/forced-static/shape-gated-auto). The full-serving candidate selects
the proven automatic gate by default. This changes only host defaults; the
device kernel bodies, inputs, flags and shapes are the ones tested here.

## Evidence

### Quality criterion for this task

The user explicitly selected **final-answer correctness** for this optimization
("최종정답이면 되지"). The serving comparison therefore uses the recorded
`quality.dimensions.result` counts for C=1. Proof-certificate, derivation,
counterfactual and witness scores remain in the raw records, but do not veto
this task's answer-correctness result. The canonical harness and its original
pass/fail fields have not been rewritten.

Fresh prefixes, profiler-off measurement, no observed JIT/capture, exclusive
traffic and execution integrity are still checked. The user subsequently
accepted this candidate on the available evidence and ended additional
benchmarking. That acceptance does not convert the incomplete candidate run
into a complete paired comparison or a pass of the stricter proof rubric.

### Recorded checks

- CPU native compile/load: **PASS**, `compile.json`. CUDA was hidden; both new
  native specializations were found, with cuobjdump reporting 128 registers,
  16 stack bytes, 28,720 shared bytes and `LOCAL:0`.
- CPU tests: **37 passed, 8 GPU-only skipped** (`cpu-tests.log`, 45 total).
  The read-only task checkout was mounted at `/repo`, with `--workdir /repo`
  and `PYTHONPATH=/repo`; CUDA was hidden.
- The final automatic-default candidate was compiled and loaded again:
  **PASS**, `compile-final.json`, cache key `20b59b6a4b9c5b6d444799ed`.
  Its two static kernels have the same resource usage. The rebased CPU suite
  also passed (37 passed, 8 skipped; `cpu-tests-final.log`).
- GB10 exactness and component timing: **PASS**, `c1mhc-tails-a55c`, ticket
  `17894505191390768`, revision 4. Payload 12.4 seconds after 459.7 seconds
  waiting in the canonical queue. Peak allocated memory: 378,778,624 bytes.
- First TP4 attempt: C=1 completed at 2K/32K/128K, with **9/9 final answers**;
  the full run is **incomplete**. Its C=1 observations are shown above, and
  the subsequent C=2 recorder failure is retained below.

`source-reuse.json` verifies that the final native source differs from the
GPU-qualified source only at three C++ and two pybind host defaults (0 to -1).
The native dense source of old base `5871c559` and new base `8b593fae` is
identical. `qualified-native.tar.gz` preserves the tested source overlay from
`f772d319`; apply it to base `5871c559` in an isolated checkout to reproduce.

The full TP4 reservation is `c1pack-full-v4-a55c`, ticket `1789447070414497`,
revision 4. Its command was replaced with the qualified MHC candidate:

```sh
ST_BRACKET_VALIDATION=full bash bench/st_bracket.sh pair \
  b2e596739566f0d0ce288c05fc35eff95b1c9a54 \
  --base 8b593fae2883eda0c819d5a4b9e11d436a45ad06
```

The live `/v1/models` response reports `max_concurrent_requests=2`, speculative
width 7 and maximum context 1,048,576. The current canonical harness therefore
runs **C=1 twice and C=2 once** per boot, with C=1 at 2K/32K/128K and C=2 at
2K/32K. Its historical `c4` record keys and this ticket's original C4 note do
not establish a four-request serving measurement. The 32-row C4 native
transition check is separate. The historical ticket name also does not select
the rejected input-pack prototype.

### First TP4 attempt: preserve the failure

`onepass-incomplete.json` and `onepass-incomplete.log` retain run
`20260915T055558-6151c6353435` on candidate `b2e59673`. All four ranks booted;
the measured C=1 requests had fresh prefixes, no observed JIT/capture and no
external traffic. Nevertheless, the quality gate passed only **5/9** cases
(50/57 rubric points): final results were 9/9, while derivation, counterfactual
and witness checks were incomplete. Korean corruption was 0/5.

The harness clears `decode.windows_med` under its original proof rubric and
retains `raw_windows_med`. This task's final-answer criterion was selected
after that run. C=2 preparation then received HTTP 409 and the
remaining run was not executed. The existing latency recorder accepted only
concurrencies 1 and 4, although the current canonical harness follows this
server's admission width of 2. That unconditional refusal explains the 409.

The recording fix accepts integer widths 1 through 4 and still refuses bools,
non-integers and out-of-range values. CPU server tests drive real scheduler
rows at every accepted width, verify token ownership, and check that refused
widths create no recording. **19 tests pass**, `cpu-tests-recording.log`.

Both repeat arms contain exactly this fix:

| Arm | Commit | Fleet session | Ticket |
| --- | --- | --- | --- |
| Dynamic baseline, main plus recording fix | `78c3f5ed32d607be6c8e600aabd4e1471dcbf84f` | `c1mhc-control-a55c` | `17894525171732173` |
| Static C1 candidate plus same recording fix | `fcd32f1358cbd34799c23bdf2fca367d542b1181` | `c1mhc-final-a55c` | `17894525771743125` |

The engine differences between these arms are only `kernels.cu` and its
`SOURCE.json` provenance. Both use the same workload, token budget and raw
harness grading. The task-specific interpretation follows the user's
final-answer criterion above. The candidate's native CUDA source is still
the one in `compile-final.json`.

The candidate's queued command was changed to a single-arm full chain at
revision 2, retaining ticket `17894525771743125`, because the matched baseline
already had its own full reservation. After the user accepted the candidate,
`bench/fleet.sh cancel c1mhc-final-a55c` stopped this waiter before admission.
No additional candidate boot or C=2 pass is claimed. Baseline run 1 completed
and reported C=2 final answers **11/12**; C=2 is additional coverage, not the
C=1 final-answer count used for this decision.

## GPU gate

The probe loads all 90 real BF16-origin MHC coefficient tensors and measures
the 89 carried boundaries. It tests ordinary AR inputs and local four-rank
packet descriptors. Four magnitudes, forward/reverse replay and poisoned
outputs compare all four output fields bitwise against the original dynamic
kernel. Rank-fold and rounding canaries and descriptor rebinding exercise the
packet path. A mixed 1/7/8/16/32/64-row schedule checks 32 iterations of counter
rearming, including C=4 and the wide FP32 consumers. Eight-row coefficients
that cannot be packed losslessly also retain bitwise FP32 behavior. Forced
static tails reject unsupported row counts and coefficient precision.

Timings use external CUDA events captured around native calls, two independent
captures in opposite allocation order, four B/A/A/B brackets, and 16 replays
per bracket arm. Warm single-layer intervals repeat 32 calls; a distinct-layer
chain visits all 89 coefficient packs. Evicted intervals follow a 128 MiB
flush outside the timed interval. Local packet timings do not include network
transport and cannot establish serving throughput.

```sh
bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes mhc_c1_tails --samples 4 \
  --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors \
  --output /cache/c1-mhc-tails-a55c.jsonl
```

Runtime: `sha256:848e493f37af252865deea2fe6169916f6bac727343b5ab592cd74fcf3639544`,
Torch 2.13.0+cu132 / CUDA 13.2. Native source SHA-256:
`b3cd1b099ecf5e984cddcf712dbc751962c755dafea391f5623a6deebf85fdb4`.
GPU work runs only through the canonical fleet queue. The full TP4 onepass
workflow followed `engine/CHARTER.md` D17; the user's explicit acceptance ended
further benchmarking before the complete candidate repeat. Accepted runtime
source is `fcd32f1358cbd34799c23bdf2fca367d542b1181` (static tails plus the
recording-width fix); later commits preserve documentation and evidence.
