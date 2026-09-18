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
and the PR remains a draft. A further context ablation rules the
assistant-provenance memory records out as the differentiator: neutralizing them
at the id level (same token count, everything else byte-identical) leaves the
failure intact in both arms, and all three outputs score clean on the canonical
Korean glyph counters while being semantically broken. As of 2026-09-18 the
exact original request's **glyph-level texture improves into the normal band**
on tree `048b682d75f9` (unseen(ko) 0.103 native vs 0.26 on the same morning's
boot) — but texture is not recovery: an ablation arm that scored 0.108 still
contains welded non-words, a follow-up on a #1157 build through the general
inference path corrupted again, and the #1157 causal effect remains
unverified — see [the clues0918 campaign](code-audit.md). As of 2026-09-18 the failing request's own frame is read off its bytes: the
prompt already ends in `<|assistant|><think>` — exactly what its generation prompt writes — so that
turn is not the missing half of anything, and the turn *before* it is already degraded inside the same
prompt
(`그리섬 기준으로`, `대상도끼보다`, `걸사 재단`). This request therefore inherits a broken assistant
turn; the incident's first breakage is upstream of the request every replay so far has reproduced,
and no private record of that earlier turn exists. See
[the prompt frame evidence](prompt-structure-evidence.json) and
[the follow-up audit](code-audit.md),
[the production replay receipts](merged-production-replay-evidence.json) and
[the shared-smoothing replay](shared-smoothing-replay-evidence.json).

## Deviations from the model's own implementation, 2026-09-18

The operator's rule for what remains: every place the engine differs from the model's own
implementation (transformers 5.16.1 `glm5_next`, the checkpoint's author code) goes back to the
model unless there is a large reason not to -- whether or not it is the incident's cause. A
sweep of the served GLM-5.3 path against that reference, op for op (mHC, KDA, the DSA indexer
and its k-pool, MLA, router, experts, norms, sampling defaults), found the semantics identical
except for the entries below.

| Place | Engine | Model | Disposition |
|---|---|---|---|
| Indexer tail (`index_kpool_always_select_tail`) | #1158/#1159 pin the pool that just completed above the top-k on rows whose length is a whole number of pools | `append_visible_tail` appends the `seq % 4` newest tokens and nothing else; every complete pool competes on its score (sglang's `append_tail_to_topk` agrees) | Kept (operator, 2026-09-18): the pin stays as the engine's reading of the flag; it is a deviation from the model on one pool of 512 on every fourth position, and it was in place for the #1174 replays, which still failed |
| Sampling defaults for a request that omits `top_p` | the served meta's `generation_config.json` (cut from the quantised repository) names only temperature 1.0 | zai-org/GLM-5.3-Flash ships temperature 1.0, **top_p 0.95** | **Restored** (this sweep): `facts.GENERATION` fills what the meta omits; a meta or a request that speaks wins. Deneb sends `top_p 1` explicitly, so its requests are unchanged |
| Routed-expert activation | b12x lanes call gpt-oss's `swigluoai_uninterleave` (`gate·σ(α·gate)·(up+β)`, kernel defaults α 1.702, β 1) | `silu(min(gate, 10))·clamp(up, ±10)` | Same function: every served launch passes α 1, β 0, limit 10 and the static dispatcher refuses anything else -- now pinned by `tests/test_engine_moe_activation_contract.py` |
| Activation scale search `as2` (#1125/#1127/#1157) | the routed FC1/FC2 activation scale is searched per 16-block instead of flashinfer's `amax/6` | the model has no activation quantiser; `amax/6` is the serving default, not the model | Kept (operator, 2026-09-18): the search never worsens a block's SSE and reverting restores nothing the model was trained with |
| Precision lanes | dense projections W4A8 (decode) / NVFP4 W4A4 or FP8 (prefill) with GPTQ and channel smoothing, the vocabulary head FP8 GPTQ, the MLA latent FP8 e4m3 at scale 1, the indexer q/k FP8 (Hadamard), NVFP4 W4A4 experts | BF16 everywhere but the checkpoint's NVFP4 experts | Not semantics: the engine's speed budget. Listed because it is the noise floor the incident sits on (below) |

What the sweep says about the incident itself, from the captures already in hand rather than
new runs: on the exact 50,005 IDs the greedy answer is nearly clean (a few near-miss syllables
such as `수아이트스`, `장정답게`), the prior turn generated two minutes earlier at 47K is the
same (`그리섬`, `대상도끼`), the 39-token clean question at T=1 also carried malformed words,
and the recorded 907-token output is a sampling spiral: coherent for two sentences, then one
low-probability draw after another until scripts mix. The engine's own two paths through the
same prefix disagree by a total variation of up to 0.16 at one position, and the same token's
per-stage outputs differ by 6.5-9.5% at layer 0 and 37-66% by layer 22 between prefill and
decode. That is the precision floor of the served lanes, not a discrete defect: at temperature
1 with the whole tail (`top_p 1`, what Deneb sends) the fat tail is drawn from, a drawn slip
self-conditions, and a 47K agentic context with an already-degraded turn is where the model's
own distribution is flattest. The semantic reverts above do not claim to repair that; the
levers that would are the precision lanes (the FP8 head first: ~0.6 ms a step for a BF16
head, and it shapes the sampled distribution directly), a nucleus or min-p on the product
side, or context hygiene (Deneb's R1'/R2, 2026-09-18).

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

## Prompt frame, and the turn this request inherited

[prompt_structure_audit.py](prompt_structure_audit.py) reads the private 50,005-token prompt's turn
frame in the served image, with the tokenizer the door serves (`/repo/st-glm53-meta/tokenizer.json`,
sha256 `0cfe2c099a7702…`). Receipts: [prompt-structure-evidence.json](prompt-structure-evidence.json).

| Fact | Value |
|---|---|
| Head / tail frame | `[gMASK]<sop><|system|>` … `<|assistant|><think>` (indices 50,003 and 50,004) |
| Generation prompt the tail matches | `<|assistant|>{{- '<think>' -}}` in the checkpoint's `chat_template.jinja` |
| Role tokens | 4 `<|system|>`; `<|user|>` at 41,479 and 47,663; `<|assistant|>` at 44,021, 44,492, 44,622, 46,027, 47,171, 50,003 |
| Between the last user turn and the request's own assistant turn | 2,341 tokens with no role token |
| The preceding turn | index 47,171, 492 tokens, thinking closes at +207, visible answer already degraded |

Three claims made while this incident was being chased do not survive that read. The request was never
missing an assistant turn: the frame is what the checkpoint's own template writes. The last Korean
instruction is not 50,000 tokens back — it is the user turn at 47,663, and the recall block after it
is already tagged `trust="untrusted"` with a note that its contents are records and not instructions.
And the third token of the thinking block is not a corruption onset: the arm drew ` 이` at p 0.108
against top-1 ` I` at p 0.706, and its decoded text continues `Follow-up to 이타카` — the intended
Korean continuation of that sentence.

What this read does **not** establish is that the input was assembled the way the model's
template prescribes. The frame's own pairing is regular — every assistant turn opens `<think>`, every
`<tool_call>` is paired, and each of the six tool calls in the turns ends in
`<|observation|><tool_response>…</tool_response>` — the seventh `<tool_call>` sits inside the system
block's tool documentation (the vocabulary holds 12 of each argument tag; turns with no reasoning
echoed back carry the template's empty `<think></think>`, as at 44,622) — but the message list this was
rendered from is private, so a mis-assembly that still leaves balanced markers would not appear here. The corruption itself is not a rendering artifact: the
selected IDs decode to the same text under the served tokenizer and the checkpoint's, and they are what
Deneb displayed (the transcript hash above).

What the read does change is where the failure starts. The visible answer of the **preceding** turn is
already degraded inside this very prompt, so the recorded 907-token output is at least the second
breakage: a prompt-borne copy of a broken answer can amplify a failure, but it cannot be the first
cause. That earlier turn ran on the same unchanged boot `fd99f82e…` about two minutes before the
request under audit, and no private record of its own prompt and engine state is retained. Reproducing
that turn is the next anchor worth having, and the prefix it needs is inside this prompt (tokens
`0..47171`) — though the per-turn recall block means that truncation is not a byte-exact prefix.

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
