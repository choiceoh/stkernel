# Complete incident sampling captures, 2026-09-18

The response-quality incident remains unresolved. Five isolated requests produced
2,313 complete sampling records (3,588,746,501 bytes), and every record passed
prefix, output-token, row-key and uniform verification. The repeat baseline has
477 bit-identical raw-logit rows, identical sampling distributions and identical
output IDs. Restoring scales changes the distribution at an identical prefix and
draw, but none of the three restoration conditions fixes the corrupted response.

## Runtime and controls

- Fleet session: `st-lossless-logits0918`, source `8f87c211515cb5896036af53c3b18ed8d908aacd`.
- Same exact 50,005 prompt IDs; T=1, underlying seed=7, top_p=1, no top-k,
  independent cache namespaces, zero reused tokens, retain=false.
- Each request used the isolated eager N=1 target path. Mode is encoded in the
  private arm's incoming seed; the adapter strips it before computing draws.
- Maximum output 2,048; all five requests ended naturally before the cap.
- First four output-ID hashes exactly match the preceding no-logits-capture
  boot `ee4e494fcca6`. This is capture compatibility evidence, not throughput proof.
- Runtime source checks include the capture helper on all four ranks and a
  separate audit of `draws.py`, sampler, fused draw kernel and adapter sources.

| Case / runtime admission | Control | Output / capture rows | First sampled difference from baseline (zero-based) | TV at that position |
|---|---|---:|---:|---:|
| 00 / 1 | mode20 baseline | 477 | — | — |
| 01 / 2 | mode22 original weight and input global scales | 407 | 19 | 0.231244 |
| 02 / 3 | mode21 original weight globals, input globals=1 | 497 | 26 | 0.089658 |
| 03 / 4 | mode23 mode22 plus original FP32 constants | 455 | 2 | 0.097506 |
| 04 / 5 | mode20 baseline repeat | 477 | none | 0 throughout |

Only equal-prefix positions are compared: 20, 27 and 3 respectively. The
remaining 387, 450 and 452 shared positions are excluded because their prefixes
have diverged. Every shared position has the same uniform, and all 2,313 recorded
uniforms match `RICH@0` for the recorded row key. These rich draws are computed on
the host and copied to the GPU; this is **not** execution of the Triton fused-draw
contract probe, nor evidence that every engine path is correct.

At the first mode22 divergence both arms have top-1 token 82651, yet sample
82651 versus 21754 with the same uniform 0.5584545135498047. Their distances to
the closest inverse-CDF boundary are 0.182896 and 0.025518, respectively. Top-1
margin alone misses this T=1 divergence. The other two first divergences also
retain the same top-1 token. The measured distribution changes do not establish
which remaining component causes the semantic corruption.

Every divergence compared here sits inside the thinking block: the baseline arm closes thinking at
index 27 of its 477 tokens, and the private prompt ends in `<|assistant|><think>` — the checkpoint's
own generation prompt; that frame read is scoped to the frame, not a verification that the whole
input was assembled as the model's template prescribes. The corruption onset is later than any position in the table, and the arms do
not reach it cleanly: the recorded 907-token original ends `</arg_value></tool_call><|observation|>`
and the baseline arm ends `<|user|>`, so both keep writing the conversation after answering. The
preceding turn is already degraded inside this prompt — see
[the prompt frame evidence](../st_telemachus_quality_20260917/prompt-structure-evidence.json) and the
[incident ledger](../st_telemachus_quality_20260917/README.md).

## Private artifacts and reproduction

On srv2, original capture files are under:

```
/home/choiceoh/glm53-logs/st-bracket-dumps/st-lossless-logits0918-hold-8f87c211515c/incident-logits
```

The private result directory is:

```
/tmp/telemachus-quality-0917/lossless-logits-replay
```

It contains `capture-manifest.json`, `capture-comparison.json`, runtime source
identities, request receipts and `margin-input/case00-mode20` through the other
four cases. Each comparison directory contains symlinks named with **logical
admission 1**, because every condition replays the same single incident request.
Their targets and payloads preserve the actual runtime admission. The manifest
documents the mapping; no original capture is rewritten.

Records preserve the existing `raw`, `processed`, `probabilities`, `uniforms`,
`picks` and `input_tail` fields and include scalar `uniform`, `prefix_sha256`,
`prompt_sha256`, `seed`, actual `row_key`, prefix length and committed tokens.
The prefix hash covers the full prompt plus every committed preceding token.

[`audit_captures.py`](audit_captures.py) verifies each prefix against the private
original prompt and returned output IDs, every draw against the served source,
and each captured pick against the actual completion. It prepares the comparison
views and computes full-distribution and sampled-token comparisons. It reads the
private paths above and requires torch plus the repaired margin reader staged at
`/tmp/telemachus-quality-0917/draw-audit-repair/tools/incident_logit_margin.py`.
[`evidence.json`](evidence.json) contains sanitized receipts, hashes and numerical
results; prompt IDs, complete output IDs/text and logits are not committed.

## Analyzer defects reproduced while validating these records

PR #1170's original probe compares a 15-word fused block against all 40 host
words for K=7, then compares each 8-word purpose block against all 40 again.
Even a perfect implementation raises `KeyError`. The repair restricts each
comparison to that implementation's expected address domain. Complete probe
flow tests run with CPU tensors, including an injected one-ULP RICH mismatch;
these tests make no CUDA claim.

Its margin reader also treats the request admission as the draw nonce. For this
seeded baseline, runtime admission=1 but the actual nonce is zero. The original
reader says the recorded uniform matches no word, despite an exact `RICH@0`
match for actual row key 11241344834629033336. The repair reads the captured row
key and no longer substitutes an admission number for missing nonce metadata.
All five purposes include their final searched position, and report counts
distinguish compared rows from top-1 flips. Different uniforms do not by
themselves establish that the distributions are equal.

The earlier combined mode24 BF16-dense control stalled before producing any
token. It is not a completed quality result and cannot justify declaring every
precision combination exhausted.
