# DFlash acceptance experiments

The observed 43–47% → about 54% aggregate change is not a matched precision result.
For the recorded `3c7bcc0a` run `20260913T065924-4fefee2be2e1`, natural-output
2K requests accepted 6,248 / 13,902 proposals (44.943%), while 32K/128K requests
accepted 24,832 / 44,892 (55.315%). Long contexts accounted for 72% of steps.
Keep each question, context length and concurrency separate when judging the
following three experiments. FP32 KDA state remains the default.

## Independent controls

| Arm | `STK_draft_fc_precision` | `STK_draft_fc_calibration` | `STK_draft_diagnostics` |
|---|---|---|---|
| A: shared W4 baseline | `w4` | `shared` | `1` |
| B: FC FP8 | `fp8` | `shared` | `1` |
| Collection, excluded from timing verdict | `w4` | `collect` | `1` |
| C: decode GPTQ | `w4` | `decode` | `1` |
| Combined collection | `fp8` | `collect` | `1` |
| Combined measurement | `fp8` | `decode` | `1` |

Production defaults are `w4/shared/0`. These are experimental knobs in a
non-production boot. `bench/st_bracket.sh` runs immutable commits in production
shape and clears knobs: create arm commits from the same implementation commit,
changing only the three corresponding production facts in `boot.declared` for
each row above. Do not present an environment override as a bracket arm.

1. **FC FP8** changes only the drafter's FC projection of target context states.
   Its existing FP8 pack is reused for small calls, retaining W4 packs for the
   other draft layers. It introduces no new full-precision weight copy. A higher
   acceptance rate can still lose tok/s through more expensive FC compute.
2. **Decode GPTQ** collects only explicitly committed decode rows. Prefill,
   short prompts, capture/warmup, ghost rows and rejected suffixes do not enter
   this Hessian. Blobs use `.committed-decode-v1` names and
   `input_scope=committed_decode_v1`; shared calibration is not overwritten.
   At least 4,096 rows are required; automatic filing targets 32,768. Collection
   stays within the existing 2 GiB calibration budget. Consume on a subsequent
   boot. Missing, incomplete or foreign calibration fails before serving.
   With W4 decode, only the FC W4 pack uses it. With FP8 decode, a separate
   FP8 pack is GPTQ-calibrated on the FP8 grid using this Hessian; shared W4
   and prefill FP8 packs retain their identities. The context call explicitly
   identifies decode, including early projection and synchronous commits;
   small prefill calls never select the decode pack. The extra FP8 pack
   occupies 80.02 MiB per rank for the current FC, declared in both the boot
   arena and budget table, and compacted into arena-owned storage. Native
   qualification refuses a prepared but unexecuted decode FP8 pack.
3. **First-rejection attribution** reads the actual global top-16 support and
   the first mismatching target greedy pick. `candidate_miss` means the target
   token was absent; `selector_miss` means it was present but not selected.
   These are observed error locations, not proof of which layer caused them.
   Later positions after the first mismatch are not attributed. EOS and output
   limits are `output_boundary`; constrained/penalized synchronous decoding is
   `policy_modified`. Sampled rows and mixed stochastic pipeline batches are
   excluded. All-accepted blocks are recorded separately.

## Durable evidence and comparison

Every recorded onepass phase writes `draft-rejections.json` beside `server.json`
and `latency.jsonl`. Counts and first-rejection positions are grouped by request
ID and sequence; ranks remain separate. `recorded=false` means no diagnostics
were observed, not zero rejections. Recording errors and `complete` are retained
so a truncated phase cannot masquerade as complete evidence. Raw records carry the context position.
Metrics expose `st:spec_greedy_first_rejection_total` by reason and accepted
prefix. Boot lane information reports the selected `draft_policy`.

The diagnostic stores candidate IDs by cache slot, including when a request
leaves or joins a bounded decode loop. Two integers per row cross the existing
result readback; no new collective or target-logit edit is added. This still
adds kernels and record-writing cost: enable diagnostics identically in A/B/C.

Use the fleet's normal immutable `st-chain` workflow. Collect first using an
isolated cache, then freeze the shared calibration for all measured arms and
give each arm a copy. Save per-rank calibration digests before/after each boot;
reject attribution if shared calibration changes. Never compare the collector's
Hessian-update overhead with an arm whose collection has finished. A/C need
matching shared files, with C additionally reading the completed decode blob.

Run full onepass with the existing coverage: C=1 at 2K/32K/128K and C=4 at
2K/32K once; C=4 128K remains excluded. Keep the same questions, K, temperature,
token limits, checkpoint, tokenizer and target packs. Use A/B/A and A/C/A warm
comparisons (or A/combined/A for the combined experiment), retaining cold runs separately and resetting prefix reuse. Record
per-question acceptance, first-rejection histogram, quality/logic checks, finish
reason, output length/hash, TTFT, decode tok/s and FC/observe latency. A valid
greedy draft-only change should preserve the target output; higher acceptance
with different/repetitive output is not a win. Do not derive improvement from
the pooled average or from a simulator alone.

## Validation status

Combined testing first collects with `fp8/collect/1`, then boots
`fp8/decode/1` using completed per-rank blobs. Collection is preparation, not a
measured combined result. The baseline uses the same diagnostic recording and
shared calibration. Freeze calibration before either measured arm.

CPU tests cover dispatch, isolated W4/FP8 calibration identities, committed-row
selection, output boundaries, cache-slot ownership, async/burst/shared-queue
retirement and release of diagnostic storage. They check unchanged tokens and
acceptance in the serving oracle when only diagnostics is enabled. These are
plumbing checks; CUDA capture, real model quality and matched throughput have
not yet been measured for these arms.
