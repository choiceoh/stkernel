# Prior visible-answer ablation and incident sampling claims

The root cause of the Deneb response-quality incident is not established. This
experiment separates two narrower claims: whether the malformed preceding
visible assistant answer is necessary for this replay to fail, and whether
greedy decoding repairs it on the same build. It changes no serving policy.

## Claims checked against the complete captures

The earlier five-run capture set is documented in
[`../st_telemachus_logits_20260918/README.md`](../st_telemachus_logits_20260918/README.md).
The original prompt ends in the actual `<|assistant|><think>` tokens. The baseline
produces its `</think>` at generation 27, then starts its Korean visible answer.
Its initial English words are reasoning, not evidence of a missing assistant
boundary or a document-continuation mode. Generation 3's sampled Korean token
continues a coherent bilingual reference to the preceding topic; its lower
probability alone does not establish the onset of corruption.

[`summarize_sampling.py`](summarize_sampling.py) recomputes these quantities from
all 477 baseline rows without emitting private output text:

| Quantity | Measured value |
|---|---:|
| Non-argmax choices | 144 / 477 |
| Sum of conditional non-argmax probabilities | 145.064330 |
| Median chosen-token probability, generation 4 to end | 0.764046 |
| Median chosen-token probability, visible answer only | 0.724817 |
| Generation 3 distance to nearest inverse-CDF boundary | 0.007294 |

The observed non-argmax count is consistent with ordinary sampling. It cannot
validate the logits themselves. An approximately 0.99 median from an early
subset cannot be generalized to the whole corrupted answer. A repeated
deterministic result likewise establishes reproducibility, not model correctness.

## Intervention and controls

Source: `f42be51dd3c60de5d7f680efdc7a1f92663de24b`.
Fleet session: `st-visible-ablation0918b`.

The original malformed Ithaca answer precedes the Telemachus question. Only its
visible body, token interval `[47379, 47663)`, is replaced by 284 copies of token
220 (space). Its clean reasoning and every structural marker remain. Both
inputs have exactly 50,005 IDs; every ID outside that interval is identical.
[`prepare_inputs.py`](prepare_inputs.py) verifies the real tokenizer, boundaries,
input hashes and unchanged regions before writing private prepared inputs.

- Original prompt ID hash:
  `e8aefde846b0c67641f9285d2646ca3977ab790534761306422ba81dcc4bb1dc`.
- Neutralized prompt ID hash:
  `20a5e65bf1ea27475229b8cebb75cace0d2e3d10901000fdda159864b5455d03`.

[`run_replay.py`](run_replay.py) verifies nine source files on all four ranks,
waits for this session's exclusive ownership, and replays five controls in order:
original and neutralized inputs at T=1/seed 7, original at T=0/seed 7, then
original and neutralized inputs at T=1/seed 11. All use top_p=1, no top-k,
max_tokens=2048, retain=false and independent cache salts. Each completed
request must increment served by exactly one and reuse zero cached tokens.

After the original five controls, the neutralized T=0 case was added to complete
that comparison: `python3 run_replay.py --neutralized-greedy-only`. It uses the
same hold and source and requires all five prior results to exist.

| Input | Temperature / seed | Tokens | Manual visible-output review |
|---|---|---:|---|
| Original | 1 / 7 | 415 | Severe language and semantic corruption |
| Neutralized | 1 / 7 | 357 | Two search tool calls; no substantive visible answer to grade |
| Original | 0 / 7 | 478 | Invented words and factual errors remain |
| Original | 1 / 11 | 455 | Severe language and semantic corruption |
| Neutralized | 1 / 11 | 604 | Much more coherent, with residual malformed wording and meta-language |
| Neutralized | 0 / 7 | 474 | Coherent prose; factual errors and unsupported literary linkage remain |

All six completed below their 2,048-token cap. Each advanced `served` by exactly
one and reused zero cached tokens. Their receipts, output hashes, source hashes
and separate fluency/semantic reviews are in [`evidence.json`](evidence.json).
Complete prompt IDs, response text/IDs and logits remain outside the repository.

**Result:** T=1 randomness is not necessary for the original input to fail.
The preceding malformed answer is an input contributor to the degradation:
neutralization improves these continuations, including the matched greedy pair.
This is not a complete quality repair, proof of direct imitation, or an
explanation of why the preceding answer became malformed. The tool-call arm is
not counted as a recovered answer. No product or engine policy is changed.

## Interpretation limits

The space intervention holds token count and positions fixed; it is one removal
of semantic content, not a proof about all possible prompt assemblies. If it
repairs a response, the source of the already malformed preceding answer still
needs explanation. If it fails, that preceding body is not necessary for that
failure; this does not absolve all other prompt content or serving computation.

Greedy success would show a mitigation for this input, not engine innocence;
greedy failure would show that stochastic sampling is not necessary. Neither
result alone assigns the cause to the model, prompt, or engine. Final visible
fluency, semantic coherence, factual accuracy and termination must be reviewed
separately. English inside reasoning is not a failure criterion.

Changing the prompt changes every full-prefix hash. The guarded logit-margin
tool correctly skips comparisons across these two prompts; it is intended for
equal-prefix runtime comparisons, not for declaring this intervention harmless.

Private artifacts on srv2 are under
`/tmp/telemachus-quality-0917/visible-assistant-ablation/`.
