# Telemachus generation failure, 2026-09-17

ST generated the corrupted Korean reported through Deneb. The engine's retained
907 output token IDs decode to the same visible text as Deneb's transcript:
both SHA-256 values are
`ef529aa06c55f3c02b3045f81ffbc149aec2de38526b1b67a925d1ab1330b154`.
The failure also reproduces with zero reused prompt tokens. An exclusive audit
has since identified BF16 atomic scatter instability in long MoE prefill. The
FP32 accumulation candidate removes the observed repeat differences, but T=1
output remains corrupted. This is a partial numerical repair, not incident
resolution. A further confirmed omission in input smoothing left the sparse
indexer's FP32 head-gate weight unscaled. Fixing it restores the reference
head-gate function, but the combined repair also fails the original T=1
requests (seeds 7 and 11). PR #1139 merged as `4c447c15` after CI passed. A further replay with its complete
production engine tree, including publication ordering and the default MoE
recipe, still fails all three T=1 cases (12,592, 495 and 386 output tokens).
The incident remains unresolved. A further captured-data audit identifies
rank-dependent smoothing across token-sharded prefill; PR #1151 repairs that
scale pairing, but its live thinking-off replay still severely corrupts Korean
and the PR remains a draft. See [the follow-up audit](code-audit.md),
[the production replay receipts](merged-production-replay-evidence.json) and
[the shared-smoothing replay](shared-smoothing-replay-evidence.json).

## Runtime and original request

- Deneb run `stream_0043`, 2026-09-17 20:49:21–20:50:17 KST;
  model `glm-5.3-flash`, provider `wormhole`, no recorded fallback.
- ST release `fd99f82e2a12e533047159b58ebd74b601d70a90`, container started
  `2026-09-17T11:39:00.297032974Z`; this boot remained unchanged during replay.
- Image `sha256:701f9919e2d674884797245d8212aafa7a8d9d951c41ef7dc79a0611c407eb55`.
- Original engine sequence 21: 50,005 prompt tokens, 47,663 reused tokens,
  907 output tokens, temperature 1.0, max output 32,768, reasoning budget 24,576.
- KDA state FP32; speculative width 7. The release's FC2 scale-search experiment
  defaults off and no runtime override was present. Fused K7 routing defaults on.

The original prompt/record is private and is **not** included in this repository.
On srv2, `/tmp/telemachus-quality-0917/original-engine-record.json` preserves the
exact input/output IDs, prompt length and original request sampling metadata.
The engine record came from the existing conversation tier, without enabling
new capture or restarting the model.

## Initial bounded live reproductions

All requests used independent cache namespaces and reported cached_tokens=0.
They ran sequentially against the existing production boot, with `retain=false`.
There was no model restart, global cache reset, or serving configuration change.
These are incident reproductions, not exclusive throughput measurements.

| Input | Temperature | Output tokens | Observed visible result |
|---|---:|---:|---|
| Exact 50,005 input IDs | 0 | 1,200, cap reached | Repeated the same paragraph containing `수구라 수구라` |
| Exact 50,005 input IDs | 1 | 554 | Broken Korean mixed with English, Thai, Chinese and Cyrillic |
| Exact 50,005 input IDs, repeat | 0 | 426 | No paragraph loop; malformed terms such as `수구` remained |
| Short question, 47 input tokens, natural reasoning | 0 | 876 | Coherent Korean explanation; 636 reasoning tokens, no forced reasoning cutoff |

The exact-input replays used seed 7 and max output 1,200. They preserve the
prompt IDs, not the original unseeded random draw sequence. The native endpoint
does not reproduce the original reasoning-budget option; the original budget
was 24,576, above every bounded replay's entire output limit.

Two preliminary 63-token short requests forced reasoning to end at 256 tokens.
T=1 produced awkward text and T=0 leaked/repeated reasoning in the visible
answer. Those requests are retained in `evidence.json`, but they are not clean
short-context controls: forcing a reasoning boundary changed the condition.

The repeated T=0 output differed after the first 20 output tokens. The server's
served count increased by two during that one request, showing another request
completed in the same interval. Thus this is **not proof of a nondeterministic
kernel**: batch composition and concurrency were not isolated. It is also not
evidence that reducing temperature reliably repairs the incident.

## What is established

1. The corruption exists in ST output token IDs before Wormhole/Deneb display
   processing. Removing visible tool tags would leave the corrupted prose.
2. Cache reuse is not necessary for the failure. The recent Deneb cache metric
   change is not an adequate explanation for these direct raw-token replays.
3. T=1 rejection sampling is not the only affected path: one greedy replay
   entered a long loop. This does not independently absolve speculative state
   commit/rollback, which greedy generation also uses.
4. The short natural-reasoning control succeeded. The exact long agent prompt
   remains a required quality case; a shorter question is not a recovery proof.
5. Canonical `bench/onepass.py:ask_stream` hardcodes temperature 0 and its graded
   workload does not cover this long agent conversation/prose continuation.

## Causal checks after the initial capture

Exclusive same-ID reproduction and a router-fusion-off control both remain
corrupted. Layer snapshots localized repeat instability to the first routed
MoE layer during prefill; the FP32 scatter candidate removes that observed
instability without recovering T=1 quality. An isolated target-only replay has
now reproduced malformed Korean with exactly one position per forward, zero
draft proposals, zero accepted drafts and zero asynchronous steps. Draft
verification and multi-position geometry are therefore not necessary for the
failure. BF16 transport, FP8 dense decode, their combination and a Torch sorting
sampler also failed to restore quality. An offline byte/tokenizer audit confirms
that the malformed original prose is already encoded in the generated IDs;
the two checkpoint tokenizers have the same vocabulary and decoder.
Component correctness and a clean short answer do not establish recovery of
the failing long request. See [the execution receipts](target-controls.json).

`probes/replay_engine_incident.py` retains the exact-ID, fresh-cache replay as a
standalone bounded HTTP reproducer. It requires an idle ST door at admission,
records before/after serving counters, and saves private artifacts outside git:

```sh
python3 probes/replay_engine_incident.py \
  --record /tmp/telemachus-quality-0917/original-engine-record.json \
  --temperature 1 --out /tmp/telemachus-replay-new
```

The idle check is not an exclusive reservation. Use the fleet workflow for
isolated runtime/kernel comparisons. The initial capture above used the
production boot; subsequent experimental boots are recorded separately.
