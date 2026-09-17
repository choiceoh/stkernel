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

## Excluded controls

The first alleged target-only replay changed `_pick_rich`, which the native
asynchronous pipeline bypassed. The next attempt supplied `grammar` and
`grammar_after` to `/v1/engine/completions`; that route did not bind those fields,
so it also remained asynchronous. Neither result is target-only evidence.

The next diagnostic arm (`e83aae36`) explicitly selects host block verification,
host target-only, host token-level rejection, device reference verification or
device target-only. Encoded diagnostic seeds normalize to the same underlying
request seed. The replay harness checks async-step and accepted-token counters
and refuses a bypassed control. The arm retains K=7 forward geometry even for
target-only sampling; it does not independently exclude coupling between
positions inside the target forward.

## Focused source review while awaiting the fleet

- Block verification: compared host and device running ratios, residual-mass
  thresholds, accepted-prefix selection and correction distributions. No
  incident-causing mismatch established. Independent exact enumeration of
  context-dependent binary K=2 laws matched the target first-token law to
  floating-point roundoff; this does not qualify the actual CUDA implementation.
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

These are source checks, not evidence that these paths are fault-free. The next
decisive result must come from exact-input, same-runtime target/verification
controls with execution counters proving which path actually ran.
