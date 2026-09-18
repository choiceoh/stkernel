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
unverified — see [the clues0918 campaign](code-audit.md). As of 2026-09-18 the engine's own problem on this boot is located: the sparse indexer never
honored `index_kpool_always_select_tail` -- nothing pinned the pool that had just completed, so on
every decode step whose sequence length was a whole number of 4-token pools (generations 3, 7, 11, ...
of this request) the model could read none of its own newest four tokens if that pool lost the
indexer's top-k. PR #1159 pins it, ten hours after this boot. See
[the indexer recency evidence](indexer-recency-evidence.json).
As of 2026-09-18 the failing request's own frame is read off its bytes: the
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

## The engine's problem on this boot: the indexer's recency guarantee

The checkpoint declares `index_kpool_always_select_tail`, `facts.architecture` asserts it, and on the
incident's boot **nothing implemented it**. `select_with_tail` appends only the incomplete trailing
pool, `[pool_len * pool_size, seq)` -- EMPTY whenever the sequence length is a whole number of pools
(`index_kpool` is 4) -- so the newest complete pool was left to win the indexer's relevance top-k like
any older pool. `pin_pools_in_logits`/`tail_pin_pools` (PR #1159, `16485bee`, 2026-09-18 06:31:33) is
the pin that closes it, in prefill and in the captured decode path.

| Fact | Value |
|---|---|
| Incident boot | `fd99f82e…`, committed 2026-09-17 20:28:42 +0900; container up 20:39; request 20:49-20:50 KST |
| `pin` call sites in that boot's `engine/profiles/glm53/net.py` | **0** (the only two matches are `typing` and `skipping`); `sparse_indexer.py` at that commit has neither function, only the promise in its docstring |
| Fix | #1159 `16485bee`, 2026-09-18 06:31:33 -- about ten hours after the boot |
| Trigger | `seq % 4 == 0`; the request's prompt is 50,005 tokens, so decode first hits it at generation 3 (seq 50,008), then 7, 11, … |

[indexer_recency_audit.py](indexer_recency_audit.py) runs the engine's own functions on those lengths:
the appended tail is empty on exactly the steps the pin names, and when the top-k drops the newest
pool the selection carries **0 of its 4 tokens** -- on the incident's boot no path could put them back.
Receipts: [indexer-recency-evidence.json](indexer-recency-evidence.json).

This is a *selection* defect, which is why every numeric arm missed it: scales, FP32 constants and
BF16 dense change arithmetic, not what the model may read. It is also inert below the indexer's top-k
budget, which is why the 47-token control was clean, and it is content-dependent, which is why the
50,094-token neutral-filler control was clean while this prompt was not. It is the gap
`engine/modules/selection_capture.py` names for this incident -- "the 2026-09-17 incident's last
unverified path is the selection itself."

What is **not** measured yet is the harm's frequency: whether the newest pool actually lost the top-k
on this prompt during decode. `selection_capture` records only the first PREFILL selection per layer,
and no `selection-*.pt` exists on the fleet. Arming it, extending it to the whole-pool decode steps
and reading `tail_pin.would_have_been_dropped` from `tools/selection_reference.py` is the next step,
not a claim; even then a frequency is not a text, so the pin-on/pin-off pair on the same prompt
remains the end-to-end control.

The sampling row itself is clean, which sharpens the contrast. Across all 2,313 captured positions
`processed` equals `raw` exactly (bf16 to fp32) on the decodable 154,856 lanes, the 24 lanes past the
tokenizer's vocabulary are exactly the ones masked, every width is 154,880 and no pick is out of
range -- [sampling_row_audit.py](sampling_row_audit.py). So the engine is faithful in what it samples
and was unfaithful in what it let the model read.

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
