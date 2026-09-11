# GLM-5.3 context failure: KDA output norm epsilon

The ST composition used `O_NORM_EPS = 1e-6`. The GLM model constructs
`FusedRMSNormGated(head_dim, activation="sigmoid")` without specifying epsilon;
that **class** defaults to **1e-5**. The lower-level `rms_norm_gated` function
defaults to 1e-6, but the class explicitly passes its own epsilon to it.

This mismatch is shared by the reference and served lanes because output
normalization lives in `engine/profiles/glm53/net.py`, outside the lane table.
It therefore survives replacing all attention kernels with torch references.
Small core activations can be amplified by nearly sqrt(10) by the wrong
epsilon. The model has 34 KDA layers, so this is not a harmless tolerance
difference.

The patch changes the composition's constant to 1e-5. No cache, indexer,
runner, graph, drafter, tier, or rank-file changes are needed for this fix.

## Sources and scope

- Baseline source: main `5c6c649f621e7695e9c86dd124bcd076ec4b105e`.
- Native runtime: `st-engine:9391`, PyTorch 2.13.0+cu130.
- Compared model source: `glm53:v13-b12x-it`, image
  `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
- Verified the constructor in the image's
  `vllm/third_party/flash_linear_attention/ops/kda.py`, the call site in
  `vllm/models/glm5next/nvidia/kda.py`, and the active fleet overlay's
  `/home/choiceoh/overlays/glm53/kda.py`. All use the class default 1e-5.
- Rank files: `/home/choiceoh/models/st-glm53-9391-up-gate-full/`.
- All GPU numerical evidence here was collected on **GB10**, not RTX 5050.

## Context plumbing experiment

`probes/engine_context_check.py` uses real rank-3 attention weights at layers
0, 3, 43 and 44, reference lanes, and 33 synthetic activation rows. It does
not claim to measure language quality or distributed collectives.

Both the deployed block size (2304) and a small block size (16) with
noncontiguous physical pages `[1, 2, 0]` are exercised. Prefix changes retain
an identical final activation. Single-token, 3-, 6-, 7- and 17-token chunks
are compared against one whole prefill. Removing history is a negative
control.

Results in `context-rank3.json`:

- Changing the prefix changes the last attention output substantially:
  relative L2 0.487–1.405 at block size 2304.
- Whole versus split prefill: KDA relative L2 at most 0.005563;
  DSA at most 0.018519. These differences are reported rather than called
  byte equality.
- 240 DSA invocations independently check the selected physical slots
  against **all causal positions**. None is empty, missing a prior token,
  or includes a future token in these short sequences.
- The independent dense causal softmax oracle differs from sparse MLA by
  at most 0.000102084 relative L2.

This excludes complete context loss in the tested attention/cache paths.
It does not cover arbitrary long-context top-k truncation or every runner
transition.

## Full 45-layer target trace

`probes/engine_context_full_trace.py` streams one real layer at a time to
bound device memory. It runs the target composition directly with reference
lanes and fresh caches, without runner, graphs, drafter or tier. This is a
first-next-token diagnostic, not an autoregressive chat acceptance test.

Inputs:

| Prompt | Exact token IDs |
|---|---|
| The capital of France is | 785, 6722, 315, 9621, 374 |
| The capital of Korea is | 785, 6722, 315, 11856, 374 |
| ` is` (no prefix) | 374 |

| Prompt | Baseline 1e-6, four-node NCCL | Corrected 1e-5, one-GB10 LocalTP(4) |
|---|---|---|
| France | ` the` (12.5), ` of` (12.125) | ` Paris` (16.125), ` known` (14.75) |
| Korea | two newlines (11.25), comma (11.1875) | ` Seoul` (16.5), ` a` (14.25) |
| No prefix | `ish` (10.5) | ` a` (17.125) |

Numbers in parentheses are logits. The four rank files agree exactly on
their recorded layer statistics and final top-10 in **each** run.

The baseline final hidden states for France and Korea differ by 0.948304
relative L2 and have cosine similarity 0.563365. Thus the bad token output
does **not** mean the final hidden state ignores context. It also reproduces
without the serving runner.

Artifacts:

- `full-rank0.json` through `full-rank3.json`: baseline four-node NCCL trace.
- `local-eps-rank0.json` through `local-eps-rank3.json`: corrected local TP4
  trace, all four real model shards on srv4's GB10.

**Limit:** the full baseline and corrected runs use different collective
implementations (NCCL BF16 sum versus LocalTP FP32 sum then BF16). This is
not a strictly controlled full-model A/B. The independent normalization
regression below isolates the epsilon error without that confound.
The four-node corrected trace was refused because another `st-glm53`
owner occupied the fleet. Its container and lock were left alone.
No final service boot, multi-token chat, DFlash acceptance or performance
claim is made here.

## Regression

`tests/test_engine_kda_norm.py` injects small known core outputs into the
actual `Glm53Net._kda` composition and compares its result to the vendored
`FusedRMSNormGated(..., activation="sigmoid").forward_native`, using the
class's default rather than a duplicate reference epsilon. The cases cover
decode (1), verify (6), chunk prefill (7), and amplitudes 1e-5, 1e-3, 1e-1.

- Corrected value: all **9 subcases byte-exact** on CPU.
- Mutation back to 1e-6: all **6 small-amplitude subcases fail**; no test
  errors or skips. The largest amplitude rounds identically in BF16.

Run with:

```bash
python -m unittest discover -s tests -p test_engine_kda_norm.py -v
PYTHONPATH=. python probes/engine_context_check.py \
  --checkpoint /home/choiceoh/models/glm53-redhat-nvfp4 \
  --rank-file /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors \
  --rank 3 --output context-rank3.json
PYTHONPATH=. python probes/engine_context_full_trace.py --local \
  --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full \
  --metadata /home/choiceoh/st-engine/st-glm53-meta --output full.json
```

For the distributed full trace, omit `--local` and run one process per node
with ranks 0–3 (srv2, srv1, srv3, srv4) and a shared NCCL rendezvous.
`--kda-o-norm-eps` permits the legacy-value negative control inside the
diagnostic only. No production configuration knob was added.
