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
it does not exclude FP8 transport error. A transport-only BF16 comparison has
not been run. Disabling sequence parallelism would also change memory and
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

The next isolated source adds FP8 dense decode readers, BF16 prefill transport
and a Torch sorting sampler as separate scalar controls. Packet-FFN-off and
shared-overlap-off controls distinguish those implementation changes from
precision itself. Baseline is repeated on that same source before attribution.

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
