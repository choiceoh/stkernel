# Rank-consistent one-shot sums

The operator's current criterion is decode step/s and speculative acceptance;
answer grading is excluded from this performance decision. Raw grading records
remain intact. `c1-two-pass-metrics.json` retains both completed C=1 measurements
on the same `8c8b031b` boot: window medians **19.896 / 19.891 step/s**, acceptance
**44.260% / 44.549%**, and **3.656 / 3.673 tokens per step** (six draft tokens).
This is a repeat on one candidate, **no baseline on this build**, and below 22 step/s.
The first full onepass is complete; pass 2 is continuing through C=4 and diagnostics.

Fresh-prefix C=1 prefill (input tokens / TTFT) is **2,795 / 2,844 tok/s** at 32K
and **2,672 / 2,688 tok/s** at 128K for passes 1 / 2. Pass 2 TTFT is 11.760 s
and 48.093 s respectively; all five C=1 requests report zero cached tokens.
Both runs retain FP16 KDA state. These repeats do not isolate FP16's effect on
acceptance; the separate state-rounding work is not part of this change.

The old local-first FP32 fold gives different BF16 outputs across ranks for
the same four operands. `[2^24, -2^24, 1, 1]` gives `[2, 2, 1, 1]` instead of
one replicated result. Fold ranks 0, 1, 2, 3 on every rank; rank 0's arithmetic
is unchanged. The vector lanes and scalar tail use the same CPU/CUDA helper.

The actual helper passes four cancellation fixtures and 65,536 random BF16
vectors, with bitwise agreement across all four rank orderings. The existing
signed MAX/publication oracle passes alongside it (two tests, 1.777 seconds).

`compile-240dd876.json` records full Torch/CUDA extension compile and load from
source `240dd876746cdae931ea3dad8e9b0b11b7fd5cc8`, with no GPU initialized or
opened. The image was `st-engine:main-ff728f43`, Torch 2.13.0+cu130, CUDA 13.0.
Reproduce with `CUDA_VISIBLE_DEVICES=` and a private `ST_ONESHOT_BUILD_ROOT`:

```
python3 -m unittest tests.test_engine_oneshot_sum tests.test_engine_oneshot_integer
python3 probes/engine_oneshot_cpu_check.py --output /out/compile.json
```

GPU constructor qualification passed on source `8c8b031b`. The constructor checks the
four cancellation columns at 1, 7, 24 and 64 rows, both ordinary and PDL entry
points where supported, plus seven-row graph replay at changed input scales.
Transport publication/fences and MAX packet arithmetic are unchanged. The door
opened at 06:29 KST after these checks, including changed-input graph replay.

The production sequence-5518 stall has not been reproduced with a traced first
divergence. This independently demonstrated numerical defect is not yet proof
of its cause or repair. The consumer's separate parked-row `KeyError` is fixed
in the base PR #792 and reproduced by a four-row CPU scheduler test. No engine
speedup or completed consumer-quality result is claimed here.

Both PR heads passed CI. Source `8c8b031b94bb80175cfccd49cfb6329d18cdecae`
combines the two fixes and is frozen remotely for `st-decode-ranksum0913`.
The canonical hold performs the boot; `functional_then_onepass.py` first checks
the exact sampled four-token production health request for another 35 seconds,
then four concurrent requests with staggered limits at temperatures 0 and 1.
All four immutable container identities must survive each wave. Only after
that qualification do two canonical onepasses run, with prefix resets on the
same boot. The first pass is explicitly labelled `cold=reset`: functional
traffic precedes it. The hold's own stop file releases it on completion or error.

`functional-8c8b031b.json` records PASS: the four-token sampled health response
completed in 1.945 seconds and all four ranks remained alive for another 35
seconds. Both C=4 waves completed their 128/256/384/512-token limits, with
running requests falling through 3, 2, 1, 0 and no container replacement.
`runtime-8c8b031b.json` retains immutable IDs and per-rank runtime manifests.
Canonical pass 1 began at 06:30:57 KST, run `20260912T213057-9725b41d901c`.
Both full consumer passes were pending at that functional checkpoint; this
functional gate is not consumer quality or engine speed proof.

At 06:49 KST, `finish_same_boot.py` took over the controller while retaining
the already-running first consumer process (PID 1065892) and all four ranks.
The original wrapper's `check=True` would have treated a fully recorded
onepass quality/evidence exit 2 as a crash and stopped before pass 2. The
replacement verifies fresh complete canonical records, retains their failed
checks, and continues pass 2 on the same boot. It also leaves the canonical
hold's 210-minute limit in charge rather than imposing a shorter per-pass
timeout. Only the exact original controller PID was terminated; the consumer,
containers, frozen checkout and hold were not signalled or restarted.

`prior-c8562a7c-stages.json` joins sampled device stages to each completed C=1
request in the previous, incomplete consumer. Rank 0 forward averages were
46.341 / 47.471 / 49.303 ms at 2K / 32K / 128K; drafter proposal averaged
3.031 / 3.036 / 3.095 ms. These are sampled stages, not a live kernel profile,
and overlapping host waits must not be added to them. They identify target
forward work as the main remaining budget for 22 step/s, not a measured win.

Private cache copies retain compiled files but exclude all calibration blobs;
`cache-preparation.json` records the source, independent copy and unchanged B12x
metadata. Root-owned immutable compiler source files required root-assisted
copying, without changing their originals. The served numerical packs therefore
start with the same RTN policy as the preceding candidate.

The separate production boot at 06:17:22 was refused because ranks 0/1 retain
conversation 0 while ranks 2/3 retain none. `production-061722/` preserves the
four startup refusals. This persisted disagreement explains why repeating that
production boot does not recover it; no production tier was edited here.

At 07:52 KST, `finish_bounded_same_boot.py` retained that first consumer and
all four ranks while selecting harness `905d8fe6f2c5a971c1325fd2179af080892fbcd8`
for pass 2. The engine still runs `8c8b031b`; its checkout and the 210-minute
hold deadline are unchanged. The new client changes only diagnostic output:
64 tokens with a 64-token floor and 32-token reasoning cap for spec-k=6.
All preparation and measured requests retain 8192/4096 and 24576/12288 limits.
The controller verifies the complete measured `_main` prefix and seven
request/grading helper modules match the first harness before taking over.
The two harness identities and their separate diagnostic envelopes are explicit
in `bounded-controller-start.json`; no same-harness diagnostic timing claim is made.
PR #797 (`9651be3a`) requires four distinct decode traces and a prefill trace with
CUDA activities per rank; its 23 CPU tests exercise the actual canonical main.
The earlier quality-exit continuation fix is merged in PR #795 (`71356aab`).

All five measured C=4 groups on the frozen candidate have now completed:
2K q0/q1/q2 aggregate output rates are 106.00/120.60/108.32 tok/s, 32K 106.03,
and 128K 78.97. These include prefill and **are invalid quality measurements**:
C=4 has 2/36 complete proof certificates (118/228 checks), with all requests
normally stopped. C=1 has 1/9 (28/57), with a raw window median of 19.896 step/s.
The first canonical pass was still in its separate diagnostic stage at that
checkpoint. It subsequently completed all six diagnostics on all four ranks;
the final record is retained under `consumer-pass1/`. Pass 2 began at 08:29:37 KST
as `20260912T232937-6373a6373feb`, on the same boot. The failed quality verdicts
remain failures, and 22 step/s has not been reached.

`consumer-in-progress/answer-failures.json` records the final-answer errors.
All five C=1 requests (nine cases) have overall `finish_reason=stop`. For example, 2K ledger
reversed the original/counterfactual decisions (296 versus the expected 287
available units), and 32K portfolio chose the right best combination but scored
it 40 instead of 56. This overall stop does **not** establish that reasoning ended
naturally: the engine forces the reasoning-end token at the thinking cap and can
then stop normally after its final answer.

`consumer-in-progress/reasoning-counts.json` re-tokenizes the complete retained
SSE channels on the CPU with the served tokenizer, explicitly disabling its
serialized 2048-token truncation. All five reasoning spans equal their configured
caps (4096 for each 2K request, 12288 for 32K and 128K) and end mid-sentence.
This is strong evidence of capped reasoning; it does not prove that more tokens
alone would make the certificates correct. The harness discarded the original
usage reasoning-token counter, so these are re-tokenized text counts rather than
original generated IDs. `summarize_reasoning.py` reproduces the counts without
engine traffic. The current two-pass workloads and failure verdicts stay intact.

`consumer-pass1/measured-reasoning-counts.json` extends the count to all 25
measured requests: all five C=1 spans and 19/20 C=4 spans equal the cap; the
remaining C=4 span re-tokenizes to 12,287 against a 12,288 cap. Re-tokenization
cannot establish whether that one-token difference reflects an original ID
boundary. No saved response contains Halvorsen or its Korean transliteration.

`consumer-pass1/problem-identity.json` compares this campaign with the stopped
01:42:11 KST preparation from `7b687301`. All five shared prompts and their
answer data are byte-identical; the reasoning caps changed from 800/2400 to
4096/12288. The request body template also stayed identical: temperature 0,
thinking on, no verbosity parameter, and both runs omit a sampling seed.
The question generator's seed remains 7. The stopped run is not a completed
consumer baseline. The reasoning tasks were introduced by #763 at 23:50:09 KST
on September 12; earlier retrieval 9/9 results use a different quality test.

`consumer-pass1/verbosity-comparison.json` separates reasoning from final content.
The five final answers became shorter by re-tokenized counts, while some early
reasoning progressed less far within the first 800 tokens. This does not identify
a causal verbosity change: **the old source stored KDA recurrence in FP32; the
current record reports FP16**. FP16 support landed at 03:31 (#788), and became
the default at 04:42 (#791). Both build and cap changed between these samples.
The user is investigating state rounding in another session; this work does not
duplicate that state change or conclude that token limits are the sole cause.
Merged #798 (`17b660f2`)
retains the server's reasoning-token usage for future runs; these frozen clients
remain unchanged.

`consumer-in-progress/profile-summary.json` retains four profiled decode steps
on each rank at each C=1 context. It groups exact kernel names from the actual
CUDA activities, whose operator attribution is unmapped. At 2K, rank 0's mean
per-step summed durations are 24.105 ms for MoEStaticKernelV5, 8.960 for mk_gemm2,
7.824 for k_oneshot_consumer and 5.442 for packed mk_mhc_ar. These durations can
overlap and include dependency waits; they are not additive unprofiled step
latency. `summarize_profile.py` also reports per-step interval unions separately
on every rank, without comparing timestamps across hosts.

Two further default-off kernel probes are prepared on separate sources:
packed mHC single-token grids (`4626491b`) and MoE resident-wave scheduling
(`c13f2802`). `compile_during_preparation.py` waits for pass-2 preparation and
compiles them only in private CPU containers (runc, no visible/requested GPU,
cores 16–19, two CPU equivalents, 12 GiB memory, 180 seconds per job). It stops
only its labelled CPU container if preparation ends. GPU tests/timings remain
pending and neither probe is present in the running engine.
