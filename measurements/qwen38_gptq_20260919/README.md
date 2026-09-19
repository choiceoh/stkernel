# Qwen3.8 real-input GPTQ calibration — 2026-09-19

Status: inputs and CPU preparation checks complete; TP4 collection, repacking,
held-out projection error and consumer comparison have not run yet.

The target is the 193 projection sites already admitted by #1286, with #1294's
FP32 MoE accumulation and `as2` domain. The experiment uses one frozen source
revision and separate pack roots. Existing shared calibration/packs are preserved.
`ST_PACK_ROOT` / `fleet --pack-root` selects that storage path (default `/cache`);
it changes neither model arithmetic nor the calibration identity.

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

## CPU checks

67 tests: 63 passed, 4 CUDA-only skips. Includes corpus split/leakage guards,
foreign-owner refusal, Hessian provenance/coverage checks, existing precision
port tests, boot/warmup and pack-store digest tests. Shell syntax and diff checks
pass. These checks are not GPU collection, repacking or serving proof.
