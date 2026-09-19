# Qwen3.8 real-input GPTQ calibration — 2026-09-19

Status: the first 131,184-row TP4 real-input collection, all-rank filing/reboot
audits and held-out projection scoring passed. Every tested W4/FP8 pack reduces
projection error. Calibration-size convergence has not been measured on Qwen.
The consumer waiter was cancelled while that coverage question is reviewed;
this is not yet a completed-calibration, serving-quality or adoption verdict.

The target is the 193 projection sites already admitted by #1286, with #1294's
FP32 MoE accumulation and `as2` domain. The experiment uses one frozen source
revision and separate pack roots. Existing shared calibration/packs are preserved.
`ST_PACK_ROOT` / `fleet --pack-root` selects that storage path (default `/cache`);
it changes neither model arithmetic nor the calibration identity.

The collection source is `aac20610`, projection scoring is `51903f57`, and
consumer dependency/identity fixes end at `d594521c`. All use engine tree
`969fbb4968c212c8b107e043f7b358c5ed7cc1e3` and the same
launcher tree `7c6be3ec4161a5099da31850f1cd5f026c38f038`. Changes between them
only add/fix offline scoring, the experiment driver and onepass metadata/dependencies. The fleet image is
`sha256:9ac492af3e5acbd35f16297bdc526e95ef46c3ec84d7bb78bb45fddb918b3180`;
each comparison boot checks all four rank image receipts against its first boot.

## Prepared inputs

The existing private Deneb conversation and workload snapshots were verified
against their provenance digests, deduplicated and rendered with the Qwen
checkpoint's actual template and tokenizer. Conversation sessions and mail/phone
source groups retain their original split. Exact messages and token sequences
are disjoint across splits. Categories are interleaved before collection so a
capped collector does not see only one source. No repeated filler was added.

| Split | Prompts | Qwen input tokens | Role |
|---|---:|---:|---|
| train | 294 | 240,490 | Fit Hessians, target at least 131,072 rows per site |
| validation | 72 | 55,441 | Reserved development diagnostics |
| test | 69 | 50,512 | Separate projection-error and response evaluation |

The data reconstruct real conversation/mail/notification inputs. They do not
reconstruct the full provider system prompt, tools or retrieval context. Stored
assistant turns are input context, not trusted quality labels. Splits have been
used for earlier GLM development; this is held-out Qwen calibration evidence,
not a newly blinded final benchmark. Text, responses and row-level provenance
stay outside Git, under srv4's private `st-calibration-private/qwen38-gptq-20260919`.
`dataset-manifest.json` contains only aggregate counts and content digests.

## Completed collection

On 2026-09-19, the previous MTP experiment released its session normally. This
run acquired `session/q38gptq-0919` through the normal quiet production handover.
The two collection boots completed at 21:33:42 and 21:36:22 KST.

| Statistics | Submitted requests / input tokens | Saved rows at every site | Sites per rank | Ranks |
|---|---:|---:|---:|---:|
| Fit | 294 / 240,490 | 131,184 | 193 | 4 |
| Held-out | 69 / 50,512 | 50,512 | 193 | 4 |

The collector stops after its 131,072-row target; the last admitted chunk adds
112 more rows. Thus not every submitted training token enters the fit Gram
matrix. All submitted token counts matched the prepared Qwen template, with
zero prefix-cache hits. All 1,544 filed site records passed FP32, finite-value,
shape, model identity and coverage checks. Fit and held-out roots are separate.
The eight `*-audit-rank*.json` files retain per-site Hessian/amax digests.
The earlier HTTP filing receipt says validation is pending because filing is
asynchronous; the subsequent all-rank audits are the completion proof.

### Calibration amount remains an open question

The earlier GLM experiment fitted 329,580 token rows and evaluated on 86,620
separate rows (`../st_kda_pack_error_20260916/README.md`). This Qwen run inherited
the shared collector's 131,072-row cap; its 131,184 saved rows are about 40% of
that earlier fit count. A row here is a token activation, not a conversation.

The shared cap came from GLM's 17K/87K/330K comparison over 30 sites
(`../st_site_lane_table_20260916/thickness_curve.log`). On the 24 sites starting
near 17K, the 87K arm captured a median 89% of the measured 17K-to-330K gain.
However, every arm was scored on the same 330K statistics that fitted the largest
arm. That experiment neither establishes independent generalization at 330K nor
proves 131K sufficient for Qwen. Model tokenizers and input distributions also
differ, so matching a token count alone is not a convergence test.

Retain this first fit and its results as a control. A larger Qwen fit needs a
separate store and a higher collection cap, with the same source/input arithmetic
and validation statistics across sizes. The current distinct prepared training
inputs total 240,490 Qwen tokens; reaching roughly 330K requires additional real
training inputs as well as raising the cap. Replaying the same inputs merely to
inflate row counts, or moving validation/test groups into training, would not
establish broader coverage. Use the reserved validation split for size selection;
the test split has already been inspected and is not a new blind final test.
No larger fit or size-comparison result exists yet. The background waiter was
confirmed cancelled before it had launched a consumer run; all first-fit packs
and numerical evidence remain available.

The first candidate boot consumed 192 W4 and 193 FP8 GPTQ packs on each rank,
with no live collectors. Its first scoring attempt completed rank 0 but refused
the three peer jobs: the central lease file only exists on srv2. The driver now
verifies that single lease before dispatch, retains it until every rank job
exits, and checks the inherited owner in nested jobs, following the existing
probe runner's nested-lease contract. No lease is copied to a peer. The original
attempt is retained outside Git; final comparison artifacts use a separate
`compare/` directory.

After successful scoring, that window was deliberately closed before consumer
measurement: canonical onepass only identified `st-glm53`. The metadata fix
adds `ONEPASS_ST_CONTAINER=st-qwen38` and refuses to borrow a different active
container's identity. The A1/B/A2 consumer bracket resumes in `serving/`, on the
same engine/image/packs. The first scoring failure and this pre-measurement stop
are not quality or speed samples.

The first A1 boot then passed all RTN audits, but onepass failed before sending
requests because it still imported `bench-dec.py` / `bracket.py`, both removed
by #1152. `onepass_metrics.py` now owns just the ST metric parsing/sampling
helpers. Original counter arithmetic, requests and grading are preserved.
The driver now loads its actual CPU dependencies **before** acquiring a window.
That failed attempt is preserved under `serving-preflight-failure/` on the nodes.
After the fix, another operator session (`session/q38mtp-tune6-0919`, estimated
180 minutes) took the fleet at 22:00:32 KST. No generated-quality, acceptance or
speed result exists yet; no production pack has been replaced.

`wait_for_window.py` prepares a bounded, single attempt at the launcher's
documented session window. It waits behind other sessions, any canonical queue
entries and pending handovers, refuses a changed source snapshot, then calls
the same `collect_window.sh serve`. It is not a canonical queue ticket and
never forces a handover from another session. Failures are retained without
automatic retries. The actual serving source is recorded in `serve-source.sha`.

## Evaluation scope

| Question | Evidence |
|---|---|
| Did real input statistics survive restart and produce served GPTQ packs? | Every rank's filing audit, boot receipt and pack identity |
| Did weight-packing output error improve on separate real inputs? | 69 held-out inputs, actual RTN/GPTQ packs, FP64 evaluation of `sqrt(tr((W-Q) H (W-Q)^T) / tr(W H W^T))` |
| Did generated quality, acceptance and speed change? | Separate deterministic `ko-reasoning-v3` problems and Korean corruption checks in canonical extended onepass, RTN–GPTQ–RTN |

Projection error excludes activation quantization and native accumulation.
Onepass's closed-world answers are mechanically checked; old Deneb replies are
not labels. The real-input collection generates only one token per request and
is not a semantic response evaluation. Neither evidence set establishes general
production conversation quality.

## Held-out projection error

All four ranks: 768/768 W4 and 772/772 FP8 projections improved, zero worsened.
The median **paired** GPTQ/RTN RMSE ratio is 0.70349 (W4) and 0.65070 (FP8):
29.65% and 34.93% lower output error at the median site. These are paired
ratios, not the ratio of the two separately calculated medians below.

| Rank | W4 RTN relative RMSE | W4 GPTQ | FP8 RTN relative RMSE | FP8 GPTQ |
|---|---:|---:|---:|---:|
| 0 | 7.1264% | 4.3082% | 2.4009% | 1.3479% |
| 1 | 7.2718% | 4.5563% | 2.3877% | 1.4009% |
| 2 | 6.9696% | 4.0525% | 2.3564% | 1.2606% |
| 3 | 7.0991% | 4.2123% | 2.3155% | 1.3134% |

Values are medians over that rank's 192 W4 / 193 FP8 sites. The FP8 target
head improves on every rank too: RTN 3.05–3.23%, GPTQ 1.45–1.59% relative
RMSE. The smallest paired improvement is still 1.75% (W4) / 2.87% (FP8).
Recompute with `python3 measurements/qwen38_gptq_20260919/summarize.py`.
`projection-summary.json` retains exact values; `compare/projection-rank*.json`
retain each site's two error/reference energies and relative RMSE.

## Execution contract

1. Acquire an exclusive fleet window after the existing MTP experiment releases
   its lease. Freeze source, checkpoint fingerprints, template, image digest,
   MTP configuration and all four rank boot receipts.
2. Boot with an empty private fit store, collect `train.jsonl`, file and validate
   all 193 Hessians on every rank. A filing HTTP reply is only a request; all-rank
   file audits must finish before continuing.
3. In another empty store on the same source, collect `test.jsonl` under the
   same RTN model. These Hessians score projection error and never enter GPTQ.
4. Restart with the fit store. Require boot proof of 192 W4 and 193 FP8 GPTQ
   projections per rank, matching Hessian digests and zero remaining collectors.
5. Compare RTN and GPTQ under the held-out Hessians using the actual packed
   weights. This is projection output error, not end-to-end language accuracy.
6. Run canonical `bench/onepass.py` with collectors disabled in both arms,
   matched preparation, prompts, K/C settings and fresh prefixes. Retain quality,
   corruption checks, acceptance, tokens/step, actual output tok/s and raw
   transcripts. Use baseline/candidate/baseline order; report failures as failures.

Tools: `probes/qwen38_gptq_data.py` prepares private inputs;
`probes/qwen38_gptq_feed.py` refuses to send requests without this session's
lease and verifies served token counts; `probes/qwen38_gptq_audit.py` checks
the saved statistics and the subsequent boot's pack-consumption evidence.
`probes/qwen38_gptq_score.py` evaluates the actual packs against the separate
held-out statistics. `observe_fleet.sh` records read-only GPU process samples.

The remaining consumer command, from the pinned srv2 worktree, is:

```bash
bash measurements/qwen38_gptq_20260919/collect_window.sh serve
```

It requires the completed `compare/projection-rank*.json` on every rank and
checks that every new boot uses the exact image and calibration weight identity
of the scored packs. It executes A1/B/A2 with two C=1 passes and one C=4 pass per
boot, using the canonical extended workload (nominal 2K/32K/128K C=1 and 2K/32K
C=4, fixed-length 1,024-token concurrency samples). `rc=2` is retained as a
failed quality/evidence result. A missing or incomplete ledger record stops
the driver. Compiling/booting/repacking time is outside consumer measurements.

## CPU checks

67 tests: 63 passed, 4 CUDA-only skips. Includes corpus split/leakage guards,
foreign-owner refusal, Hessian provenance/coverage checks, existing precision
port tests, boot/warmup and pack-store digest tests. Shell syntax and diff checks
pass. After adding the FP64 score equivalence, nested-owner and deferred-window
checks, all 13 focused corpus/scoring tests pass with CPU PyTorch. The onepass
identity/real-dependency/profile/recording/quality/measurement tests pass 68/68.
These checks are not GPU collection, repacking or serving proof.
