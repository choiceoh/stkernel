# Fixed K7 follow-up experiments, 2026-09-17

This campaign starts from the three scoped defaults merged in #1068. The
additional mHC, MLA and MoE input-reuse selectors are enabled as a combined
default by explicit user decision: small or inconclusive component gains do
not by themselves block adoption. The fused-router consumer comparison is
complete and has worse quality observations, so its default was restored to
the original router. No speed claim is made for the retained combined default.
K=7, FP32 KDA state, expert weight storage and selection width are retained.
These are component measurements, not engine throughput claims.

## Recorded comparisons

| Record | Candidate source | Result |
|---|---|---|
| `gpu-v1.jsonl` | `87f2a59e` | MLA tile-end barrier cleanup passes 42 fixtures bitwise, usually <1% faster; empty-tail C2 is 1% slower. MoE max tree passes but timings are close to noise. mHC explicit rounded reciprocal multiplication changes comb bits. Paired MoE register cache corrupts some route inputs. BF16 MLA tile hits the shared ticket-counter divisor bug. |
| `gpu-v2.jsonl` | `ce98dbdf` (candidate kernels from `ec10f4b8`, merged main `f24230e7`) | mHC contraction repair is bitwise over 90 coefficients, ordinary/packet inputs and real KDA consumers, but no latency win. Separate MLA ticket counters fix the launch failure; BF16 materialization passes the numerical gate but loses 15% at C1 and 28–31% at C2 on full selections. Balanced paired MoE still fails; raw route-byte diagnostics localize the error to paired routes. |
| `gpu-v3.jsonl` | `90d93fb8` (candidate kernels from `59cedb1a`) | All component gates pass. Direct MLA conversion is bitwise on all 42 fixtures and reduces ordinary full-selection latency by 3.3–4.8%. mHC exact expansion reduces the three-layer KDA consumer chain by 4.27% at C1, while the broad 90-boundary mHC chain regresses. Block-local paired MoE quantization fixes the corruption, but whole-layer gains stay at noise level. |
| `gpu-v4-final.jsonl` | `4de8c56f` | Trimmed final source passes all 45 numerical records. MLA is bitwise on 42 fixtures, with ordinary full-selection latency -3.3% to -5.1%; duplicate selections and empty tails are near-neutral. C1 mHC plus three real KDA projections is -2.07%. |

Do not compare absolute latency between records: main advanced and the GPU
state differs. Each timing row contains its own same-process B/A/A/B samples.
The original v1 MLA events have the inherited `mla_tile32` label; this run
actually compares synchronization cleanup against the current tile16/tile32
defaults. The summarizer corrects that label explicitly.


## Retained scope and rejected trials

- mHC shared reciprocals: rejected for no useful speed gain after restoring
  the compiler's original contraction. `rejected-mhc-rcp.patch` preserves it.
  Exact BF16 coefficient expansion is retained only at C1 boundaries that
  produce the following KDA input pack. C2 and other mHC boundaries keep the
  old path: the broad coefficient-only chain regresses despite the KDA gain.
- MLA BF16 tile: rejected for latency. The compiler's initial 184-register
  allocation was reduced to 128 without local-memory spills, but the extra
  conversion/publication/shared traffic still costs more than it saves.
- MLA graph counters: experimental kernels own distinct monotonic counters;
  switching resident grid divisors on the same counter can deadlock replay.
- MLA candidate: direct FP8-to-BF16 conversion only at 8/16 decode rows;
  long prefill retains the qualified half bridge. No precision change intended.
- MoE paired input reuse: block-local reuse fixes the corrupt routes, but
  C1/C2 evicted whole-layer changes remain -0.32% to +0.13%. This and the
  max-tree change are removed; `rejected-moe-frontend.patch` retains the trial.
- The fused router introduced by #1075 is connected behind an explicit,
  default-off consumer selector for the exact GLM53 K7 profile. Resident FP32 biases are budgeted in the arena;
  native compilation precedes collectives, and startup proof requires every
  bound layer/row width to execute. Its changed logit/tie order requires
  consumer quality and acceptance evidence, beyond component numerics.

## Reproduction and evidence

`probes/engine_kernel_check.py --lanes next_k_compile` compiles with CUDA
hidden. The final `next_k_cost:<section>` selects only `mhc` or `mla`.
Historical commits also contain rejected BF16 and MoE frontend sections.
Use the recorded source revision for historical flags and native APIs.

GPU trials use the canonical fleet queue with `--gpu --fleet`, an immutable
worktree and the real rank0 checkpoint. They do not run beside serving.
CPU compilation uses a GPU-unexposed runc container and asserts no CUDA
context is initialized. `native-resources*.jsonl` is static disassembly
metadata; fewer instructions are not a speed result.

```
python3 measurements/st_fixed_k_next_20260917/summarize.py
```

## Final source and consumer pair

The matched control is `0330d576a4c9749aaf6b855d81c3d8bd42a87c3b` and the
candidate is `9b3a192fd0d58a994cb5b30de0c1ef55d823e593`. They differ only in
three Python defaults: `MHC.EXPAND_FN`, `ENABLE_MLA_DIRECT_CVT`, and
`Glm53Net.fused_decode_router`. Native sources and build inputs are identical.
`consumer.patch` records this difference. Both include main `e4332734`.

`final-compile.jsonl` compiles the final dense, MLA and router CUDA sources
without GPU exposure. The measured source is `2d15d27c`; subsequent changes
before the pair also include the startup-proof repair and main integration
described below. The three candidate CUDA source files remain unchanged. The focused
CPU suite runs 50 tests with one GPU-only skip and no failures. The full
CPU engine suite runs 2,255 tests in 241 files with 368 skips, no failures
and no unimportable modules (`cpu-suite.txt`). The candidate-default focused
rerun passes all 23 tests (`cpu-candidate.txt`).

The full consumer tickets are `fixedk-next-consumer-b2-0917` and
`fixedk-next-consumer-a2-0917`, using the same controller from the control
worktree, separate admissions and the same `fixedk-next-consumer-0917.jsonl`
sink. They retain the canonical failed gates; separate admissions allow the
candidate to be observed even if baseline quality fails. Each boot runs the
full onepass twice, with 1,024 fixed decode tokens and three repetitions,
2K/32K/128K contexts, and C1/C2 coverage where capacity permits.

Control B completed both runs. Normal quality is 25/30 across both C1 passes
and the one C2 pass; Korean corruption is zero. The canonical quality gate
still fails, including the bounded 1,024-token requests. Fixed-decode pooled
step/s is 13.4306 cold and 13.3462 warm, with output 61.5209/59.7590 tok/s.
The cold fixed-concurrency measurement also reports changed preparation and
must not be used as a valid throughput result. These are control observations,
not an optimization verdict. `consumer.jsonl`, `consumer-summary.json`, mapped
runtime identity and diagnostic launch counts retain the source evidence.

Candidate A resumed its original immutable reservation after the MoE component
probes. Both cold and warm runs are complete and the fleet was released. The user
authorized combined adoption even where an individual gain is small or inconclusive.
The retained input-reuse mode 3 has separate same-build component evidence in
`st_moe_input_reuse_20260917`; this older consumer A/B does not include it.

### Cold consumer observations

`collect_step_costs.py` reads unprofiled `gpu_iteration` records from the
measurement phases. It counts rank 0 once and separates actual batch width
two from C2's single-request startup/tail. C1 admissions, initial prompt
positions and committed-token totals are checked against every completed
ordinary request. The raw device timer covers the decode body and TP4 stop
agreement; it excludes host/HTTP overhead. All four ranks report unchanged
preparation for these phases. `consumer-device-step-cost.json` retains source
hashes, iteration counts, acceptance and stage costs.

| Ordinary workload | Control device ms/step | Candidate device ms/step | Reduction |
|---|---:|---:|---:|
| C1 2K | 50.5388 | 47.7546 | 5.51% |
| C1 32K | 50.9764 | 48.0708 | 5.70% |
| C1 128K | 51.3432 | 48.8642 | 4.83% |
| C2 2K, width two only | 74.3296 | 72.6478 | 2.26% |
| C2 32K, width two only | 74.4890 | 72.5556 | 2.60% |

C2 width-two acceptance changes from 55.10% to 53.44% at 2K and from 50.28%
to 51.39% at 32K. Its forward stage accounts for approximately 1.68/1.93 ms
of the observed reduction. Changed output trajectories and the quality result
below still prevent treating this as an equivalent-quality performance verdict.

The earlier window-derived comparison remains available as a separate metric:

| C1 workload | Control ms/step | Candidate ms/step | Reduction |
|---|---:|---:|---:|
| Ordinary 2K, reciprocal of window mean | 53.6854 | 51.0364 | 4.93% |
| Ordinary 32K, reciprocal of window mean | 54.0519 | 51.3898 | 4.93% |
| Ordinary 128K, reciprocal of window mean | 54.3357 | 52.0922 | 4.13% |
| Fixed 1,024 tokens, pooled elapsed time / steps | 74.4570 | 73.7695 | 0.92% |

The ordinary window medians remain approximately 19.9 step/s; fewer low-rate
windows raise the mean. The legacy series excludes zero-step stalls. The
ordinary rows are equivalent costs, not direct
average step latencies. Fixed step/s changes from 13.4306 to 13.5557. C1 raw
acceptance across ordinary and fixed requests changes from 51.47% to 53.63%
(+2.16 percentage points). Ordinary C1 output changes from 85.24 to 91.13 tok/s
(+6.91%) and fixed C1 output from 61.52 to 72.81 tok/s (+18.36%).

Normal C1 quality remains 7/9 with the same failing cases and dimensions. C2
strict quality falls from 11/12 to 8/12 (rubric 75/76 to 70/76), despite the
final-answer dimension remaining 12/12. Additional failures concern derivation,
counterfactual and witness checks. This is a quality regression observation,
not evidence that quality is unchanged; one C2 run cannot establish its cause.
The canonical second run repeats C1 only, so it cannot resolve the C2 result.
All compared output hashes differ. Neither observed token-rate changes nor
ordinary step-window changes isolate a kernel speedup.

Normal C2 per-request pooled decode changes from 62.20 to 63.97 tok/s (+2.84%).
Fixed C2 changes from 49.39 to 49.89 tok/s (+1.00%), but its preparation state
changes on all four ranks in both arms, so the fixed-concurrency result is
invalid. Quality and preparation failures remain in the canonical evidence.

### Warm repeat and adoption decision

| Ordinary C1 workload | Control device ms/step | Candidate device ms/step | Reduction |
|---|---:|---:|---:|
| 2K | 50.3610 | 47.8617 | 4.96% |
| 32K | 50.7274 | 48.1810 | 5.02% |
| 128K | 51.5012 | 48.8533 | 5.14% |

Warm fixed-output cost is 74.9276 to 71.1497 ms/step (-5.04%). Overall C1 raw
acceptance is 51.11% to 54.18% (+3.06 percentage points). Ordinary pooled output
is 84.48 to 92.47 tok/s, and fixed output is 59.76 to 72.40 tok/s. These remain
observations of the measured three-selector candidate, not the final default.

Warm normal C1 quality falls from 7/9 to 6/9: the candidate's 32K ledger answer
fails the result, derivation and counterfactual checks. The final-answer
dimension is 8/9, versus control 9/9. C2 is measured only once and remains 8/12
versus 11/12. Korean corruption is zero throughout. `consumer-verdict.jsonl`
records the canonical **NO EVIDENCE** result; neither arm has a valid warm
sample. Bounded fixed-output quality failures were not removed from that gate.

The optional fused router's FP32 summation and tied-selection order differ from
the control. This trial does not prove that the router caused each quality
failure, but it also does not qualify that numerical change for a serving
default. `fused_decode_router` is therefore false again. Bitwise-qualified mHC
and MLA changes remain enabled, as does separately qualified MoE input-reuse
mode 3. The final combined default has no full-engine speed verdict; the
measured candidate included the fused router and excluded input reuse.

The first control ticket (`fixedk-next-consumer-b-0917`, source `4de8c56f`)
was interrupted before a complete onepass record. Fixing the candidate's
startup proof required renewed admission after main had changed the head
and drafter paths (#1082), so both arms were moved to the same new base.
The initial candidate ticket never ran. `aborted-consumer-runtime-B.json`
retains the old control identity and is excluded from the final comparison.

The startup proof now requires coefficient expansion at actual KDA input-pack
consumers, excluding the first layer and boundaries after auxiliary outputs.
The router build remains in the boot registry because this branch binds it;
main's probe-only registry fix (#1085) does not apply to the new consumer.
The integrated candidate passes 30 focused CPU tests (`cpu-integration.txt`).
The dense/mHC, MLA and fused-router CUDA sources did not change after the
final component gate; the independent main changes are shared by both arms.

The offline consumer reader preserves the existing metrics and adds explicit
ordinary step windows, pooled fixed step cost and C1 acceptance comparisons:

```sh
python3 measurements/st_fixed_k_next_20260917/consumer_summarize.py \
  measurements/st_fixed_k_next_20260917/consumer.jsonl
ONEPASS_VERDICTS=measurements/st_fixed_k_next_20260917/consumer-verdict.jsonl \
python3 bench/st_judge.py judge --cand 9b3a192fd0d58a994cb5b30de0c1ef55d823e593 \
  --base 0330d576a4c9749aaf6b855d81c3d8bd42a87c3b \
  --jsonl measurements/st_fixed_k_next_20260917/consumer.jsonl --write
# On the artifact host, read existing measured traces (no new GPU work):
python3 collect_step_costs.py /path/to/fixedk-next-consumer-0917.jsonl
```

## Default integration and ordinary step windows

The initial combined defaults passed CI on `aa6657e0`: 257 engine test files, 2,344
tests, zero failures or unimportable files, 467 skips; also 137 onepass contract,
77 oracle, and 21 fleet admission tests. The first default-on CI exposed that
the fused router also selected the tiny test profile. That was scoped to its
exact supported geometry before measurement completed. The subsequent quality
decision disables its default entirely, while retaining explicit probe access.
The rollback passed 26 focused Linux/ST-image tests with one GPU-only skip
(`../st_moe_input_reuse_20260917/cpu-defaults-router-control.txt`).

`consumer_summarize.py` wraps the existing consumer reader without changing the
quality, output-hash or preparation gates. It also reports the one-second step
windows for ordinary 2K, 32K and 128K requests. The fixed-output windows are
validated and removed from the tail of the 2K bucket before computing its mean
and median. Equivalent ms/step is the reciprocal of the window mean, not the
pooled elapsed-time-per-step metric used for the fixed-output comparison.
`live-128k-window.json` retains the requested ten-second in-flight snapshot;
it is a partial observation, not a substitute for the complete C1 acceptance.
