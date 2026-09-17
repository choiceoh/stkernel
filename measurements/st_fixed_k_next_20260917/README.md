# Fixed K7 follow-up experiments, 2026-09-17

This campaign starts from the three scoped defaults merged in #1068. The
additional selectors are now enabled as a combined default by explicit user
decision: small or inconclusive component gains do not by themselves block
adoption. Numerical failures and clear regressions remain disqualifying.
Full consumer comparison is still running; no combined speed claim is made.
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
- The fused router introduced by #1075 is now connected behind an explicit
  disabled consumer selector. Resident FP32 biases are budgeted in the arena;
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

Candidate A is temporarily paused under the same immutable reservation to
prioritize the requested MoE input-reuse experiments. Resume its existing
ticket after the component probes; do not silently substitute a new base.
The candidate comparison remains pending. The user subsequently authorized
combined adoption even where an individual gain is small or inconclusive.
The retained input-reuse mode 3 has separate same-build component evidence in
`st_moe_input_reuse_20260917`; this older consumer A/B does not include it.

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

The offline consumer reader is reused without changing its metric definitions:

```sh
python3 measurements/st_fixed_k_cost_followup_20260917/summarize.py \
  measurements/st_fixed_k_next_20260917/consumer.jsonl
python3 bench/st_judge.py judge --cand 9b3a192fd0d58a994cb5b30de0c1ef55d823e593 \
  --base 0330d576a4c9749aaf6b855d81c3d8bd42a87c3b \
  --jsonl measurements/st_fixed_k_next_20260917/consumer.jsonl --write
```
