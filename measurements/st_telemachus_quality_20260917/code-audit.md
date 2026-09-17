# Incident code and runtime audit

The T=1 incident is unresolved. Preserve the original 50,005 input IDs, seed,
output cap, fresh cache namespace and runtime identity for every causal replay.
Private prompts, token records and activation tensors remain outside git.

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
generation-quality measurement. The isolated original-request replay on
`03e9c50551666156b89c799d967138a39cbb01d7` is pending.

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
