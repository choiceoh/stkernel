# Larger public calibration and private Deneb workloads

Follow-up to [the initial 36-prompt experiments](README.md), 2026-09-12.
These are L3/rank0 numerical reconstruction experiments for experts 10, 4,
119 and 178. The reference is the existing ST quantized checkpoint, not BF16.
Relative L2 measures the norm of output error divided by the reference norm;
it is neither an answer failure rate nor model benchmark accuracy.

## What the public-data expansion established

The expanded corpus combines the 36 original fixtures with 160 conversations
from [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
(MIT). It contains 152 training, 22 validation and 22 diagnostic-test prompts.
The final assistant response is removed; earlier user/assistant context remains.
The normal GLM template is applied with thinking disabled, then the first 512
tokens are retained and the first eight positions discarded. Long conversation
windows can therefore end before the final user turn. This is short prefix
calibration, not full conversation or decode replay.

Training tokens increased from 3,402 to **64,903**; total captured tokens are
82,028. The four experts have 10,029 / 1,153 / 2,146 / 3,596 routed training
rows, with a cap of 8,192 rows per expert. Validation and diagnostic-test rows
remain capped at 512 per expert. Training still covers only a small part of the
model. Expanded capture peaked at 3,108,708,352 Torch-allocated bytes.

The separate 16-prompt comparison contains eight Korean authored inputs and
eight English UltraChat inputs, 5,352 kept tokens and 1,400 tokens with at least
one selected expert. These experts account for only **3.758% of routed slots**.
All rows below use the same 16 inputs and the same original expert weights.

| Reconstruction | Weighted selected-expert sum relative L2 | Token median | Token p95 |
|---|---:|---:|---:|
| Magnitude pair pruning | 61.857% | 67.233% | 72.486% |
| Small corpus, SparseGPT | 40.003% | 52.902% | 70.500% |
| Small corpus, independent residual | 38.724% | 50.155% | 69.845% |
| Expanded corpus, SparseGPT | 25.768% | 47.236% | 65.951% |
| Expanded corpus, independent residual | 21.860% | 42.974% | 63.116% |
| Expanded corpus, deployed-input down residual | **20.453%** | 44.479% | 64.152% |
| Expanded corpus, reselected down-projection pairs | 23.834% | 46.963% | 66.286% |
| Compact 256-channel dense NVFP4 expert | 27.589% | 56.395% | 75.973% |

The aggregate gain hides a domain regression. Per-prompt macro-average error
for Korean inputs rose from **20.156% to 22.864%**, and all eight Korean inputs
worsened, comparing the small independent residual with the expanded deployed
residual. The English macro-average fell from **45.735% to 21.954%**, improving
all eight English inputs. The best aggregate method also worsened the token
median/p95 relative to the expanded independent residual. A single pooled norm
cannot establish a uniform quality gain.

A later joint factor fit balanced Korean and English minibatches and required
neither language's validation error to worsen from initialization. Its replay
of this development holdout reached **20.154%**. It did **not** establish an
error below 20%, and the reused holdout is explicitly marked as a development
revisit. On the earlier eight-prompt holdout where the original residual had
30.159% error, the expanded deployed residual gave 22.900%; that is also a
revisit. The 30.159%, 38.724% and 20.453% figures must not be compared as if all
came from the same prompt set.

## Executed reconstruction changes

- `engine_sparse_chain_residual.py` fits the existing down-projection factors
  on the *deployed* sparse expert's intermediate activation and the original
  complete expert output. Candidates include plain ridge and router-weighted,
  feature-RMS-conditioned ridge. The old factors remain eligible; validation
  routed error selects the result. This addresses upstream error propagation
  without adding factor matrices or inference operations.
- `engine_sparse_sequential_reconstruct.py` solves an anchored dense target
  for W2, then reselects legal adjacent pairs with SparseGPT and requantizes.
  This adds no residual GEMMs. Four W2 exports passed native sparse checks.
- `engine_sparse_compact_expert.py` keeps complete K16 groups of intermediate
  channels, reducing 512 to 256, refits W2 and compares max/4 versus max/6 scale
  choices. All four exports ran through the existing dense b12x kernel. Native
  versus dequantized-reference error was about 0.24%, including output rounding.
  This is a **dense compact alternative**, not sparse-MMA acceleration.
- `engine_sparse_joint_residual.py` trains BF16 residual factors jointly through
  the complete expert, with frozen sparse matrices and FP4 straight-through
  activation rounding. Equal Korean/English batches and per-language validation
  guards prevent a lower average from selecting a language regression. This
  script's language labels refer specifically to the public fixture corpus; it
  is not used as a generic language detector for private workloads.
- K16/K32 reference activation encoding is chunked into 256-row pieces to bound
  temporary allocations. Both encoders matched their original unchunked output
  bit-for-bit on a 513-row GPU check; no production quantizer was modified.

The public residual variants use exactly **4,161,536 bytes** of BF16 factors
across four experts. New corpus comparisons cap every factor rank at that same
prior budget. Sparse weights plus metadata/scales occupy 8.25 MiB in the pilot's
logical accounting, versus 13.5 MiB for the original dense expert matrices;
compact dense weights/scales occupy 6.75 MiB. These are weight-storage counts,
not serving latency measurements. No new end-to-end throughput gain is claimed.

## Deneb workload sources and privacy boundary

The first private pilot used client conversation windows only. That omitted
major background workloads. A read-only inventory subsequently found:

| Source | Observed records | Use |
|---|---:|---|
| Mail archive messages | 3,632 (3,619 with a nonempty body) | Reconstructed analysis input from headers and archived body |
| Phone notification ledger | 3,266 notifications | Inventory of available raw event data; location events excluded |
| Phone judgment run starts | 5,233 | Actual logged incoming task previews, including event text |
| Notification digest run starts | 108 | Background input previews; training only |
| Morning briefing run starts | 109 | Background input previews; training only |

Counts describe the inspected live snapshot and can drift. Mail archive bodies
are already cleaned/bounded by Deneb; they are not original MIME attachments.
Agent logs cap incoming messages at 4,096 bytes and do not retain the full
system prompt, retrieved history or tool results. Likely truncated previews
were excluded. Mail inputs use an explicit reconstructed analysis instruction;
phone/digest/briefing inputs reuse logged incoming previews. No stored analysis
answer is treated as a supervised target.

`engine_sparse_deneb_corpus.py` reconstructs bounded conversation windows,
excluding thinking/tool blocks. `engine_sparse_deneb_workloads.py` adds archive
mail and background/event inputs. Mail Message-ID/References connections and
normalized reply subjects form connected groups. All notifications sharing a
source are assigned together. Existing conversation-session splits are retained.
Repeated payloads and detected cross-source input overlap are excluded; this is
a deterministic text/link check, not proof against every semantic paraphrase.
Background batches are training-only because they can combine other records.
The mixture is a balanced diagnostic sample, **not a measured traffic-frequency
mixture**. Quantitative workload comparisons must retain per-category results.

The workload extractor read 33,596 records, excluded one malformed record,
169 likely truncated previews, 2,085 repeated payloads and 178 detected
cross-source evaluation overlaps. It did not repair or alter source logs.
Credential, email-address and phone-number patterns are masked locally. This is
not full anonymization: business contents remain private. Raw text, token IDs,
captured tensors and source provenance stay in mode-0700 directories on srv4,
with mode-0600 files, outside Git. Only aggregate numerical evidence is exported.
No production configuration, running model, original checkpoint or service was
changed by these experiments.

After prefix deduplication, the workload capture contains 197 training, 48
validation and six public diagnostic prompts, with 101,899 kept tokens. Training
contains 63 conversation, 64 mail, 64 notification and six background inputs.
Validation has 16 inputs in each main category. The separate final set contains
12 conversation, 16 mail and 16 notification inputs. Its contents are not passed
to calibration or model selection.

The container companion [run_deneb_workloads.sh](run_deneb_workloads.sh) renders
and deduplicates 512-token prefixes before fitting, preserves the original six
public diagnostic prompts, freezes artifact hashes, captures separate final
inputs, and evaluates every frozen variant on those identical inputs. It expects
read-only `/repo`, `/work`, `/conversation`, `/ranks`, `/meta` and `/native`
mounts and a writable private `/private` mount.
[aggregate_deneb_workloads.py](aggregate_deneb_workloads.py) verifies frozen
artifact identities and exports an explicit allowlist of numerical aggregates;
it omits private prompt identities, token sequences and source provenance. Run in the recorded
`st-engine:9391` image with a 10-GiB container memory limit and three CPU cores;
never mount the production state writable.

## Frozen evaluation on real workload inputs

All five variants were frozen before capturing the same 44 private evaluation
inputs. The capture contains **19,589 tokens**, with 5,715 tokens activating at
least one selected expert. The four experts cover **4.019% of routed slots**.
All four TP replicas matched exactly. Frozen artifact hashes and disjoint input
prefixes were verified before publishing these numerical aggregates.

| Calibration / residual variant | Combined weighted relative L2 | Conversation (12) | Mail (16) | Notification (16) |
|---|---:|---:|---:|---:|
| Small public corpus | 27.002% | 31.478% | 38.785% | 21.193% |
| Expanded public corpus + balanced joint factors | 26.849% | 31.122% | 37.807% | 21.473% |
| Deneb conversation-only + deployed residual | 17.838% | **14.889%** | 32.088% | 14.679% |
| Deneb mixed workloads + independent residual | 16.165% | 24.858% | 19.575% | 8.534% |
| Deneb mixed workloads + deployed residual | **15.528%** | 24.044% | **17.988%** | **8.355%** |

**The mixed-workload result is below 20% on this combined numerical metric.**
Every category improves relative to expanded public calibration, but conversation
error worsens from 14.889% to 24.044% relative to the conversation-specialized
model. Increasing relevance to mail/notification workloads does not preserve the
best chat-only reconstruction automatically. The corpora also differ in token
counts and selection, so this is an operational comparison rather than a clean
causal ablation of a single variable.

The combined norm emphasizes high-energy outputs. Token error remains material:
with the deployed workload residual, its median is **30.819%** and p95 is
**57.123%**. Independent workload factors have a better median/p95
(28.695% / 56.184%) despite a higher combined norm. No variant was promoted to
production on these results. A subsequent fit needs validation guards for every
workload category and a new final holdout before claiming uniform improvement.

The mixed-workload capture has 79,964 training and 21,092 validation tokens,
plus the 843-token public diagnostic set. Before the per-expert fitting cap,
routed training counts are 13,337 / 4,324 / 3,914 / 4,681 for experts
10 / 4 / 119 / 178. All eight reconstructed projection exports matched native
references and sparse-versus-dense checks bit-for-bit. BF16 factors remain
**4,161,536 bytes**, identical to the original budget. Capture peak allocation
was 2,963,875,840 bytes; final capture peaked at 2,907,586,560 bytes.

The [private-workload aggregate](deneb-workload-aggregate.json) contains these
metrics and per-expert numerical results, with no private text, prompt IDs,
token sequences or source identities. Private captures/provenance remain on
srv4 and the completed experiment containers have exited.

## Evidence and verification

The initial [provenance](provenance.json), engine compatibility checks and raw
small-corpus results remain historical evidence. Expanded reports record their
own source and artifact hashes; the [follow-up provenance](expanded-provenance.json)
records the current probe sources and public evidence artifacts. Later additions to holdout metadata and category
reporting changed the script hash; old reports retain their original hashes.
The full public capture metadata remains on srv4; compact summaries record its
hash and split counts without duplicating token sequences in Git.

- [Expanded capture summary](expanded-capture-summary.json),
  [fresh public comparison capture](expanded-final-capture-summary.json).
- [Small-corpus comparison](expanded-final-small.json),
  [expanded residual](expanded-final-large.json),
  [deployed residual and reselected W2](expanded-final-large-chain.json).
- [Korean/English joint-factor development revisit](joint-residual-revisit.json),
  [compact dense experiment](compact-expert.json),
  [K16/K32 chunk exactness](quant-chunk.json).
- [Public source manifest](st-sparse-expanded-prompts-9391.manifest.json),
  [fresh public holdout manifest](st-sparse-expanded-holdout-9391.manifest.json).
- [30 numerical and data-boundary tests](workload-tests.log) passed on the
  recorded ARM image, with no skips. These include covariance-shift recovery,
  anchored regression, BF16/FP4 export, language guard behavior, secret masking,
  mail-thread grouping, phone-payload parsing, retry isolation and corrupt/live
  log handling. Python compilation and `git diff --check` also pass.

The next acceptance gate is broader full-model evaluation and measured serving
cost. Low selected-expert vector error alone cannot authorize sparse production
adoption or establish that BF16-to-NVFP4 quantization error is smaller/larger by
an equivalent numerical amount.
