# C=1 MHC static tail ownership (2026-09-15)

## Result and implementation

The native gate passed. The 89-boundary interval falls **8.11% warm / 8.43%
evicted** for ordinary AR inputs and **8.77% / 8.81%** for local rank packets.
Both independent captures improved in every measured case. All output fields
match bitwise, including mixed row counts and the FP32 fallback. These are
component results; full TP4 consumer comparison is the remaining gate.

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

Source `f772d319` on base `5871c559` qualified native `tail_mode=0/1/-1`
(dynamic/forced-static/shape-gated-auto). The full-serving candidate selects
the proven automatic gate by default. This changes only host defaults; the
device kernel bodies, inputs, flags and shapes are the ones tested here.

## Evidence

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
- First TP4 attempt: C=1 completed at 2K/32K/128K, but the full run is
  **incomplete and invalid as speed evidence**; details below.

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

The record correctly clears `decode.windows_med` and retains the raw value
only as invalid evidence. Then C=2 preparation received HTTP 409 and the
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
`SOURCE.json` provenance. No quality check or workload budget was weakened.
The candidate's native CUDA source is still the one in `compile-final.json`.

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
GPU work runs only through the canonical fleet queue. Final adoption requires
the full TP4 onepass under `engine/CHARTER.md` D17.
