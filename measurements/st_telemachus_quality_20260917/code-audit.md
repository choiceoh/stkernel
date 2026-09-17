# Incident code and runtime audit

The T=1 incident is unresolved. Preserve the original 50,005 input IDs, seed,
output cap, fresh cache namespace and runtime identity for every causal replay.
Private prompts, token records and activation tensors remain outside git.

## Shared-input smoothing mismatch, 2026-09-18

The next CPU audit identifies a function-changing error before the first
attention recurrence. `smooth_inputs()` computes its factors from each rank's
local weight rows and calibration. The captured L0 factors differ from rank 0
in 8, 16 and 36 channels on ranks 1, 2 and 3, with ratios of 0.5 or 2.
During scalar decode each rank divides its input and multiplies its own
weight by the same factor, preserving the map. Token-sharded prefill instead
normalizes a row on its owning rank and gathers it to every TP projection.
The sender's division and receiver's weight multiplication use different
factors. Neither `TokenShards` nor the projection transport compensates that
ratio. The same boundary exists at dense-MLP `post_norm` outputs.

A CPU oracle rebuilt the actual first-layer input from checkpoint embedding
rows and mHC weights, then evaluated the captured FP8 projection packs. The
last captured rows of both original prefill chunks belong to sender rank 3.
On receivers 0–2, the captured projection differs from the correctly paired
receiver-scale reference by **7.29–9.02% relative L2**. Using the uncorrected
sender scale predicts the captured values within **0.195–0.221%**. Sender and
receiver rank 3 agree with the correctly paired reference within 0.188–0.203%.
Those small remaining differences include kernel and transport rounding not
fully emulated by this CPU oracle. The scalar-decode inputs agree exactly
across all four ranks after undoing their individual smoothing factors.

[The sanitized audit](shared-smoothing-input-audit.json) includes 16 scalar
input comparisons and eight captured prefill projections; private inputs and
embedding rows remain outside git. This evidence locates a real incorrect
scale pairing. It does not yet establish that repairing it resolves the entire
reported generation failure.

Candidate `5d3e0fcef0d9bc6dd5d53bfac25fde7b4e24540a` chooses one factor for
shared `in_norm` and `post_norm` outputs using the maximum activation and
weight-column peaks across TP ranks. A calibration-presence bit travels with
those peaks, so a rank without calibration still participates; when every rank
lacks calibration all ranks leave the group unfolded. The collective runs at
weight preparation. Rank-local `q_a_norm` outputs retain their local factors.
The pack cache already keys the resulting smoothed weights and factors, and
the parked-state tag advances to `shared-smooth-v4`.

The four-rank CPU regression reproduces the old incorrect received-input
projections and proves the corrected projections bit-identical to the original
unsmoothed linear map. It also covers partial and absent calibration. All 20
focused smoothing, projection-owner and boot-breakdown tests pass. The full
engine CI on PR #1151 also passes.

The `st-shared-smooth0918` live hold completed all three T=1 replays on the
candidate. The original seeds 7 and 11 produced 544 and 500 tokens; both are
more structured than the preceding baseline, but malformed expressions and
semantic errors remain. The thinking-off case produced 816 tokens and still
contains severe Korean corruption, including a replacement character. Every
request ended below its cap, used zero cached tokens and advanced the exclusive
owner's served count by one. Ten source files on all four running ranks match
the candidate. [Execution receipts](shared-smoothing-replay-evidence.json)
record the runtime and output hashes without private text.

**This is a confirmed scale-pairing repair, not a recovered incident.** PR #1151
remains a draft after the failed full quality gate. The live hold was stopped
after the three replays; no additional performance claim follows from their
elapsed times or shorter outputs.

## Merged production replay, 2026-09-18

PR #1139 merged as `4c447c151c23c192e7a5cce6ce298cb2e3649673` after its
engine CI passed. The tested source `fea591bd106452f7fcc27d041424ff0a0105c670`
and that merge share complete engine tree
`7a09f692e1c00d04548cc3d4f215724b46a43830`. This arm uses the production
MoE recipe, input-smoothing correction, FP32 long-prefill scatter and the
publication-ordering fixes; it does not use the private diagnostic controls.

The exclusive `st-telemachus-ship0918` fleet hold replayed three requests:

| Case | Prompt tokens | Output cap | Output tokens | Result |
|---|---:|---:|---:|---|
| Original IDs, T=1, seed 7 | 50,005 | 32,768 | 12,592 | Corrupted Korean; also echoes private prompt material |
| Original IDs, T=1, seed 11 | 50,005 | 32,768 | 495 | Corrupted Korean |
| Original IDs plus reasoning-end, T=1, seed 7 | 50,006 | 1,024 | 386 | Corrupted Korean |

All three ended before their caps, reported zero cached tokens and advanced
the exclusive owner's served count by exactly one. They use `top_p=1`,
`top_k=-1` and independent cache namespaces. These are seeded replays of the
original prompt IDs, not the original unseeded random draws. The native
endpoint does not reproduce the original reasoning-budget adapter.

Nine relevant file hashes on each of the four running ranks matched the
source, including all three stock/micro/generic publication fences. The
subsequent overlapping-merge regression on main did not affect this pinned
runtime. This combined-arm failure is not an estimate of any individual
repair's effect on quality, acceptance or throughput. No per-request draft
counters were captured, and elapsed request time is not a matched speed test.
The runtime stopped at 01:34:50 KST and released its fleet lease at 01:35:00.

[Sanitized receipts and runtime identity](merged-production-replay-evidence.json)
retain hashes and counts. Private prompts and output text remain on srv2.
**The production-source replay still fails the incident quality gate.**

### Overlapping publication-fence merges

Main briefly lost all three stock/micro/generic global publication fences:
#1139 removed one copy of each duplicate and #1143 removed the other copy.
On `cd8cc080`, the two publication/source-contract test modules reproduce six
failed subcases. PR #1145 restores exactly one fence at each publication
boundary and matches their provenance hashes. Its full CI passed before merge
`0bf16f005dccb8bb1b29b8732165896ca95be74c`; the seven tests pass on that merged
main. This repairs the merge regression without resolving the response failure.

## Indexer smoothing reader omission, 2026-09-18

`Glm53Net.smoothing_groups()` omitted `idx.w_heads` from the consumers of the
DSA `in_norm` output. The same normalized hidden vector feeds `mla.qkv_a`,
`idx.wk`, `idx.gate`, and the FP32 head-weight projection. Dividing the norm by
the channel factor without multiplying `w_heads` changes the head gates used
to weight pool-selection scores. Both eager execution and the captured owner
bind these weights after smoothing. This is a function-changing omission,
not a choice of accumulation precision.

The serving rank-0 calibration contains channel peaks for all 11 DSA layers,
stamped with the actual `st-glm53-b12x-up-gate-v1` weight layout and 136,806
calibration tokens. The old scale factors are non-unit in 57 to 4,096 channels
per layer. The first layer's factors range from 0.25 to 0.5. These files predate
the failing captured boot and remain in its mounted `/cache` store.

Fix `e6f2cf3200d10b356c18903f42903fba312b7cd2` includes the FP32 reader in both
the scale calculation and the in-place column fold, preserving its FP32 dtype.
It also changes the parked-state compatibility tag so that states computed
with the old attention selection cannot re-enter serving. The dense pack
cache already keys smoothed weights and calibration factors, so changed
factors rebuild their packs without deleting unrelated cache entries.

The new regression constructs unequal channel factors and two pools. On the
old implementation smoothing changes the selected pool from 0 to 1; the fixed
implementation preserves both the selected pool and the FP32 gate values
exactly. The smoothing, projection-owner and head-gate suites pass 13 CPU
tests; their one GB10 GPU-only test is skipped on the CPU runner.

[`indexer-smoothing-invariance.json`](indexer-smoothing-invariance.json) records
an independent CPU check with actual rank weights and calibration, **synthetic
inputs**, and the target's actual `smooth_inputs()` method. Every fixed layer
matches its unsmoothed FP32 head-gate result bit for bit. The old reader omission
has substantial errors on these synthetic inputs; those errors are not a live
generation-quality measurement.

The isolated replay on `03e9c50551666156b89c799d967138a39cbb01d7` **failed to
recover quality**. Original 50,005-ID requests at T=1, seeds 7 and 11, generated
10,006 and 597 tokens before their end tokens; both contain malformed Korean,
and seed 7 enters a long repetition loop. A thinking-off control with one
appended reasoning-end token generated 472 malformed tokens. All three used
fresh cache namespaces, reported zero reused tokens, and advanced the exclusive
owner's served count by exactly one. Per-request draft counters were not
recorded; they must not be inferred from the root status response.

The four running containers' file hashes match the diagnostic source for the
two modified model files and all three FP32 accumulation files. This private
arm keeps its existing diagnostic controls (including the no-activation-search
MoE recipe), so its failures are diagnostic evidence, not a production
performance qualification. Sanitized receipts and per-rank runtime identities:
[`indexer-smoothing-replay-evidence.json`](indexer-smoothing-replay-evidence.json).

Additional T=1 thinking-off controls on that same boot show:

- 39-token clean question: 517 mostly readable tokens, with malformed words.
- Exact original system prefix plus a clean user question, 41,509 tokens:
  a 33-token tool call, which does not establish answer quality.
- Original history plus a clean user question, 47,693 tokens: 356 malformed
  tokens. Removing only the latest user's injected context does not fix it.

These are changed-input controls, not recovery of the original request.
Receipts: [`indexer-context-control-evidence.json`](indexer-context-control-evidence.json).
The combined production branch passes 23 CPU tests with one GB10-only test
skipped. Those repairs merged in PR #1139; neither mathematical repair suffices
to declare the generation incident solved. The production replay above also
fails the original quality gate.

## Confirmed numerical defect

Two isolated replays first differed at layer 3 FFN, before speculative decode.
For the two prefill chunks (32,256 and 17,749 tokens), the sampled last 32 rows
had 1,066 and 1,013 differing elements respectively, with maximum absolute
difference 0.125. The final hidden samples reached maximum differences 1.8047
and 1.0625. Long SF6 prefill used concurrent BF16 accumulation of expert outputs.

The candidate uses FP32 scatter storage for ordinary and packet long-prefill
ABIs, retains BF16 rounding of each contribution, and converts the final sum
back to BF16. Its zeroing extent covers every FP32 word. All 180 sampled stage
comparisons on rank 0, including both final hidden samples, matched bit for bit
in the repeated candidate replay. The top-20 first-token logprob records also
matched. This is observed repeatability, not a general deterministic-sum proof.

The original `0bad46c9` and this branch's `6956b29b` have the identical stable
patch ID `f37d312032e7f6670cd4c52ab25c4e24f89e36cf`. Diagnostic source
`b11061eb21cf` directly contains `0bad46c9`; the live container's three affected
source files were SHA-256 matched to that checkout. The fix was not omitted
from the later failing controls.

The zeroing change must not be described as an existing half-cleared BF16
buffer: one Uint32 clears two BF16 elements, so the old `cols / 2` words
covered the whole old buffer. Changing the accumulator to FP32 requires
`cols` Uint32 words. That is a necessary part of the dtype transition, not
independent evidence that half of the original BF16 plane was uninitialized.

Private comparison receipts are `comparison-rank0.json` under:

- `st-telemachus-audit0917b-hold-ecea6152a0ab/incident-prefill-audit`
- `st-telemachus-fp320917-hold-6f957763522a/incident-prefill-audit`

Both directories are below `/home/choiceoh/glm53-logs/st-bracket-dumps` on srv2.
The FP32 candidate still produced malformed Korean at T=1 with seeds 7 and 11
(343 and 544 output tokens). T=0 was more coherent but retained factual errors.
Warming the exact prompt in 6,912-token pieces also failed at T=1. Thus neither
temperature clamping nor avoiding one large prefill chunk is a proven repair.
The 6,912-token control still exceeds the 2,048-row FP8 transport threshold;
it does not exclude FP8 transport error. The later BF16-transport control below
also remains corrupted. Disabling sequence parallelism would change memory and
execution geometry and is not an equivalent control.

## Excluded controls

The first alleged target-only replay changed `_pick_rich`, which the native
asynchronous pipeline bypassed. The next attempt supplied `grammar` and
`grammar_after` to `/v1/engine/completions`; that route did not bind those fields,
so it also remained asynchronous. Neither result is target-only evidence.

The next diagnostic arm (`89ebe27a`) explicitly selects host block verification,
host target-only, host token-level rejection, device reference verification or
device target-only. Encoded diagnostic seeds normalize to the same underlying
request seed. The replay harness checks async-step and accepted-token counters
and refuses a bypassed control. Modes 2 and 6 retain K=7 forward geometry even
for target-only sampling. Mode 5 additionally runs the target at exactly one
position, without a draft proposal or a speculative target graph. Its host
control smoke check verifies one input row, one committed token and no proposal;
the live replay must additionally report zero async steps and zero drafted and
accepted tokens.

## Completed target controls

The isolated `89ebe27a` boot reproduced malformed Korean with the original
50,005 prompt IDs, T=1, seed 7 and zero reused tokens in all three controls:

| Execution | Outputs | Async steps | Drafted | Accepted | Result |
|---|---:|---:|---:|---:|---|
| Ordinary device block verification | 343 | 207 | 1,442 | 136 | Corrupted; same text hash as the earlier FP32 candidate |
| Host target-only, eight-position target forward | 579 | 0 | 4,046 | 0 | Corrupted |
| Host target-only, one-position eager forward | 537 | 0 | 0 | 0 | Corrupted |

Each request preserved the exclusive fleet owner and advanced served count by
exactly one. These controls show that draft acceptance, asynchronous sampling
and multi-position target geometry are not necessary for the failure. They do
not establish correctness of target computation, prefill, or the scalar sampler.
Sanitized counters, output hashes and source identity are in
[`target-controls.json`](target-controls.json).

The original prompt also contains malformed prose in the preceding assistant
answer. Removing that preceding conversation while retaining the system and
latest user turn reduced the prompt to 43,821 tokens; its 800-token scalar replay
hit the output cap. A separate 47-token T=1 request also hit its 1,200-token cap
after lengthy reasoning. Neither is evidence of restored visible-answer quality.

## Completed precision and sampler controls

Source `654f42cad5cd` repeated the scalar baseline on the same boot as all
controls. All 537 baseline output IDs equal the earlier `89ebe27a` scalar
baseline. The 50,005-token, T=1, seed-7 requests reused zero tokens, completed
one request each, and reported zero asynchronous steps, drafts and acceptances.

| Scalar control | Outputs | Result |
|---|---:|---|
| Baseline | 537 | Malformed |
| BF16 prefill transport, packet FFN disabled | 494 | Malformed |
| FP8 dense decode, shared W4 overlap bypass disabled | 441 | Malformed |
| Both controls combined | 410 | Malformed |
| Torch sorting sampler | 537 | Malformed; first differing ID at output offset 125 |

These controls reject the proposed changes as incident repairs. They do not
exclude every precision defect or qualify the entire target implementation.
The transport control also changes the packet FFN path. The FP8 control also
changes the shared-MLP execution path. Their separate component controls were
not run after neither combined change recovered quality.

The actual boot used **Red Hat compressed-tensors rank files**,
`st-glm53-9391-up-gate-full`, with `st-glm53-b12x-up-gate-v1` layout. Production's
environment explicitly overrides both `RANKS_DIR` and `CKPT`; the NVIDIA
defaults in source are not this runtime's weights. `modelopt=False`: the first
three dense MLPs are DenseLinear readers and are covered by the FP8 control.
Routed expert folded scales remain unchanged. The metadata tokenizer SHA is
`0cfe2c099a7702a0921abc315ee039deb51e4a34b4818fc509bd27fa3dc4acc1`.

A replay ending before the prior Ithaca answer generated 739 tokens with
mostly coherent prose but factual and lexical errors; it is not recovery of
the exact incident. Explicit top_p=0.95 with ordinary device verification also
remained malformed (415 outputs). The actual Red Hat generation config omits
top_p; the NVIDIA file's top_p=0.95 is not the served default. Counters, hashes
and boot identity are in [precision-control-evidence.json](precision-control-evidence.json).
The diagnostic hold ended and all four containers were stopped through the
fleet workflow at 23:39:47 KST.

## Offline tokenizer and byte audit

The tokenizer audit uses the preserved IDs, without inference or a GPU. The
Red Hat and NVIDIA tokenizers have identical 154,856-entry vocabularies,
merges, added-token IDs and decoder definitions. Their tokenizer JSON differs
only in the saved truncation setting, which the engine explicitly disables.
The entire 50,005-token prompt decodes and re-encodes to exactly the same IDs.
This does not prove that the original chat template/prompt contents were right.

All 907 original output IDs exist in the vocabulary. Decoding them with either
tokenizer, independently reconstructing ByteLevel bytes, and streaming batches
of 1/2/7/8 tokens all reproduce the saved full text. Its bytes are valid UTF-8,
with zero replacement characters. The full text, including reasoning, hashes
to `b11ec28f853b6fcd3ff8bb7e7b5b50805ab752a165cc8a50d4e18ef696f06153`;
the earlier `ef529a...` hash covers the visible answer only.

The scalar controls contain one genuine malformed UTF-8 sequence in the raw
generated tokens. At output offsets 124/125 (zero-based), token 27125 supplies
bytes `eb a9`, then baseline token 17130 supplies ASCII `ekt`; the Torch control
instead supplies token 17160, ASCII ` belongs`. A replacement character is the
correct decoding of those bytes, not a streaming conversion fault.

The baseline's chosen-token logprob at offset 125 is -17.9818000793. Its
seed-7 RICH uniform is 0.9985362291. A low probability for one selected token
does not alone establish a sampling defect: the aggregate tail and full CDF
must be checked. The two samplers first diverge here after 125 identical IDs;
this is not a claim that their numeric results match bit for bit.

Synthetic 64-token streaming batches exposed an additional mismatch on the
two scalar records after the stall guard fired. Batches 1/2/7/8 agree exactly;
the original incident agrees even at 64. This separate finding does not
explain the original failure and has not been shipped as a repair.

Receipts: [tokenizer-audit.json](tokenizer-audit.json), tokenizers 0.23.2 on CPU.
This is an offline check, not a claim of the serving image's library version.
`probes/audit_engine_tokenizer.py` reproduces it; its detailed text and ID trace
must remain in a private output directory outside git.

## Prefix segmentation comparison

The same scalar baseline's first 160 generated IDs were also reproduced through
the legacy completions route with logprobs. Fresh-prefilling the identical token
prefix through output offsets 1, 64 and 120 retained the top-1 decoded label but
changed common top-20 logprobs by up to 1.87, 1.87 and 3.13 respectively. This is
not a state-corruption proof: prefill/decode use different arithmetic, and that
API merges distinct partial-byte token IDs under the same decoded U+FFFD key.
Further numeric comparisons must preserve token IDs, not decoded labels.

### Full-ID distributions and stage tails, 2026-09-18

Diagnostic source `d30713e09a8a` retained the same arithmetic and captured raw
logits, processed logits, probabilities, uniforms and picks. Its scalar first
160 tokens reproduced the earlier baseline exactly. Each fresh-prefill control
used the exact baseline IDs through offsets 1, 124 or 125, with zero reused
tokens. At those positions the top-20 **ID** overlap was 16, 13 and 20; probability
total variation was 0.000180, 0.028883 and 0.157325. The previously reported
one-entry decoded-label overlap at offset 125 was a byte-token label collision,
not a one-token vocabulary overlap.

At offset 125, the native pick was 17130, the earlier Torch pick was 17160,
and FP64 inverse-CDF on the captured logits picks **17113** (ASCII `isan`). All
three are invalid continuations of the pending bytes `eb a9`. Native probability
error against FP64 is at most 8.66e-9 on this row. Its chosen interval misses
the uniform by 5.63e-8 because FP32 cumulative sums round differently. This
explains the first different sampled ID between implementations, but an exact
CDF still generates invalid bytes at this draw; it does not explain or repair
the quality failure. The raw model distribution assigns 0.0006926 mass to
invalid next-byte continuations. Fresh-prefilling the same prefix assigns
0.0010481, and at the same uniform selects valid continuation byte `9f`.

Rank 3 stage tails compare the same global final token, including the actual
last non-padding row of sequence-parallel prefill. Differences start at layer 0
attention (relative L2 12.5%, 12.4%, 15.0% for the three positions), before routed
experts. Final hidden relative L2 is 20.6%, 30.5%, 30.7%. This comparison includes
different dense precision and KDA execution paths; neither path is an independent
correctness oracle.

A further control freshly prefills through offset 124, forces only the required
next input token via logit bias, and captures the subsequent one-token forward's
unbiased **raw** logits and stage outputs. Its consumed input ID is verified equal
to the original scalar step's input ID. Layer 0 attention differs from full
prefill by 14.7%, versus 15.0% after 125 scalar steps. The two scalar paths differ
by 3.65% there. Thus the layer-0 difference is already present at a single
prefill-to-decode transition; it is not solely accumulation over 125 decode steps.
No root-cause or quality-recovery claim follows from that localization alone.

Receipts: [exact-logit-analysis.json](exact-logit-analysis.json),
[utf8-next-mass.json](utf8-next-mass.json),
[stage-tail-analysis.json](stage-tail-analysis.json),
[forced-one-step-analysis.json](forced-one-step-analysis.json). Private tensors
are under `st-bracket-dumps/st-telemachus-logits0917d-hold-d30713e09a8a` on their
respective ranks. The isolated owner was `queue/st-telemachus-logits0917d`.

### Actual KDA operand audit, 2026-09-18

Source `b11061eb21cf`, owner `queue/st-telemachus-kda0918b`, adds private operand
capture around layers 0 and 1. It also incorporates main's FC1 TMA proxy-fence
repair `48a23b02`. The 126-token scalar continuation decodes to the same text as
the earlier baseline. Three requests advance served count by one each, use zero
cached tokens, and cover the original scalar continuation, fresh prefill through
offset 125, and fresh prefill through 124 followed by the exact forced input.

The CPU auditor uses the real convolution inputs/history, actual incoming KDA
state, Q/K/V, gates, norm operands and resident linear weight packs. It covers
80 cases across all four ranks. Maximum relative L2 differences are:

| Operation | Prefill | Scalar decode |
|---|---:|---:|
| Convolution | 0.00226% | 0.00303% |
| KDA output | 0.49292% | 0.01622% |
| KDA final state | 0.23576% | 0.00000834% |
| Output norm | 0.00393% | 0% |

The largest linear-reader difference from the same quantized-weight arithmetic
is 0.02602%. The convolution-to-recurrence and recurrence-to-norm handoffs are
bit-identical. All 24 captured adjacent state/history boundaries match bit for
bit, including prefill-to-decode and consecutive scalar steps. This does not
support a large ring-state or kernel-arithmetic error at these first two layers.
It does not validate the entire incoming prefill history, later layers, routed
experts or sparse attention. Prefill recurrence checks cover the last 64-token
kernel chunk (or its partial tail) using its captured starting state.

`probes/audit_engine_kda_capture.py` reproduces the CPU audit from private `.pt`
files. The diagnostic capture was separately checked on CPU to preserve output,
marked-state order, ring mutations and the original lane bindings. The result is
[kda-operands-audit.json](kda-operands-audit.json); private operands are retained
outside git under `st-telemachus-kda0918b-hold-b11061eb21cf/incident-kda-operands`
on each rank, with a private collected copy on srv2.

Four additional T=1, seed-7, thinking-off controls finished below their 1,024-token
cap. A 39-token standalone question produced 434 tokens of readable Korean.
A 50,094-token prompt containing neutral filler and that same question also
produced 434 readable tokens. The actual Deneb input with reasoning closed
(50,006 tokens) still produced malformed Korean; removing the preceding turn
(43,822 tokens) produced unrelated HTML before tool calls. None is a repair of
the original request. The neutral control shows that this context length alone
does not invariably trigger the corruption; it does not separate every effect
of input content, activation distribution, template or quantization.

## Accepted-token counter defect found during review

The host commit counts `min(accepted, emitted_count)`, but both asynchronous
implementations used `min(accepted, emitted_count - 1)`. If EOS or max output
clips off the correction/bonus token, the last emitted token can itself be a
confirmed draft. For example, three verified drafts clipped to two emitted
tokens were recorded as one accepted token asynchronously and two on the host.

The candidate corrects the asynchronous reference and fused commit counter.
Tokens, committed length, stop decisions, contexts and KDA state are unchanged.
This repairs acceptance statistics, not the malformed text. Cross-path tests
cover greedy/sampled verification, all K=3 accepted lengths, output rooms and
stop positions. The focused sampling/pipeline/agreement suite passes 38 tests;
the real Triton commit body also matches counts, tokens, stop flags, kept drafts
and context in 588 K=1/3/7 cases under its CPU interpreter. That interpreter
result is not a CUDA replay qualification.
The scoped metric correction merged as PR #1129 (`418f4168`), after the required
CI passed; its ancestry in `origin/main` was verified. It is not a deployment
receipt for the incident repair.

## Focused source review while awaiting the fleet

- Block verification: compared host and device running ratios, residual-mass
  thresholds, accepted-prefix selection and correction distributions. No
  incident-causing mismatch established. Independent exact rational enumeration
  of 36 context-dependent binary models, K=1/2/3 and a four-token output horizon,
  matched all 16 complete sequence probabilities exactly, including subsequent
  blocks. This is stronger than a first-token-only check, but does not qualify
  the actual CUDA implementation.
- Kernel arithmetic: the actual Triton sampler and block-verification bodies
  were run under the CPU interpreter with CUDA hidden. Thirty-two sampler rows
  at vocabularies 777 and 154,880 matched the sorting oracle (maximum probability
  difference 1.1920929e-7); 12 K=7 verification rows matched accepted counts and
  token IDs. Inputs carried BF16-rounded values in FP32 storage, so this does
  not cover native BF16 loads, compilation or graph replay.
- Draft sampling: traced returned sparse masses to the actual conditional walk,
  and checked rank agreement includes candidate IDs and probability bits.
- Random draws: traced request-seed normalization, generation count and separate
  draft/verify/correction purpose keys through the captured path.
- Commit: checked the clipped accepted count, old-context copy, anchor advance,
  KDA materialization, boundary staging and drafter observation order.
- Ring rollback: checked conv, recurrent-state and kpool-tail history extents
  after accepting only one position of the eight-position target forward.
- Graph storage: traced static target logits, owned probability buffers and
  copies of drafter graph outputs before replay. No lifetime fault established.
- DSA causality: checked complete-pool horizons and incomplete-tail inclusion.

These are source checks, not evidence that these paths are fault-free. The live
target controls above narrow the investigation; precision and native sampler
controls remain necessary before identifying the incident cause.

## Context ablation — the assistant-provenance records are not the differentiator, 2026-09-18

The prompt these replays carry is not a plain transcript. Its tool and memory
records quote the assistant's **own reasoning**: `[ctx] [assistant] The user asks
"..." — likely from conversation history ...` (x4), `[assistant] The user asks
...` (x3), `**[assistant]** ...` (x2) and `[도구 sessions] {"action":"search",...}`.
One of them is stamped `[2026-09-17T20:46:56+09:00]`, two minutes before the
incident request. The memory records that are 95-100 days old quote ordinary,
coherent assistant prose, so the self-referential material is recent.

Two ablations asked whether that material causes the failure. Each replaced the
id span of the selected prompt lines with the same number of a neutral filler id
(`15`), so the prompt keeps its **exact token count** and every id outside the
spans is byte-identical. The request is the incident's own 50,005 IDs at T=1,
seed 7, `top_p=1`, `top_k=-1`, 1,024-token cap, `retain=false` and a fresh cache
salt.

| Arm | Lines neutralized | Replaced tokens | Output tokens | Glyph scan | Result |
|---|---:|---:|---:|---|---|
| Baseline | — | — | 554 | 0 replacement, 0 welded jamo, no Cyrillic/Thai | Derailed Korean |
| `[ctx] [assistant]` records | 4 | 398 | 469 | 0 / 0 / none | Derailed Korean |
| All assistant-provenance records | 11 | 975 | 382 | 0 / 0 / none | Derailed Korean |

All three texts stay topically anchored to Ithaca/Telemachus while being
semantically broken (invented words and wrong referents: `수아이비터`,
`퓔로스`, `미네르바(아테나)가 떠서 맨토 이름으로 변신해`), and **all three score
clean on the canonical Korean glyph counters**. The gate this incident is judged
by counts glyph damage; this failure is semantic, and the gate has no case that
can see it — the incident needs a graded semantic case over a long agentic
context.

The gate this incident lacks is `bench/agentic-recall.py`: it builds one long
agentic context in the product's shape, buries one exact fact (a code, a date) in
the oldest third, puts three decoys in the newest third, and requires the answer
to state the fact and neither decoy -- deterministically, without a judge, beside
the glyph counters of `bench/korean-corruption.py` that scored these answers
clean.

**Ruled out:** the assistant-reasoning memory records are not the differentiator.
Neutralizing either set leaves the failure intact, so "strip the records" is
hygiene rather than the incident repair. The next causal step is the measurement
this incident still lacks — **real-data selection**: capture the sparse indexer's
selected pool ids for the incident prefix and compare them with an fp32 reference
selection recomputed from the same captured operands (`q8`, `w_eff`, `keys`,
`scales`). The selector *kernels* are verified on synthetic logits and the KDA
operands were audited on real ones; the selection itself has never been compared
on real data, and it is the only path here that engages for long contexts and
content-dependently — short contexts take the covered path and never select.

[Receipts](context-ablation-evidence.json), reproducer
`probes/incident_context_ablation.py`. The door was idle at admission but is
production traffic, not an exclusive hold: every arm advanced `served` by two, so
these are quality observations, not an isolated timing measurement. Private
prompt text and outputs remain on srv2.
