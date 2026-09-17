# Incident code and runtime audit

The T=1 incident is unresolved. Preserve the original 50,005 input IDs, seed,
output cap, fresh cache namespace and runtime identity for every causal replay.
Private prompts, token records and activation tensors remain outside git.

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
