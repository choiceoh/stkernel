# Drafter residual/RMS → cuBLAS head producer, 2026-09-17

Default prepared cuBLAS draft-head calls now fuse final residual addition,
RMS, non-anchor row compaction, FP8 group-128 quantization and MX32 scale
publication in one launch. The selector and calibration observer retain the
same compact BF16 hidden values. Logical vocabulary padding is still sliced
by FP8Linear. The original BF16 residual sum, RMS reduction and both BF16
normalization rounding boundaries are preserved. No weight recipe changes.

## Matched direct RTX 5050 result

Both arms call actual `Glm53Net.head_local` / `FP8Linear` with the same
prepared cuBLAS reader. The baseline is main's separate `add_norm`, non-anchor
selection/compaction and ordinary BF16 head input. Candidate uses the fused
producer and `project_mx`; that entry point skips the old input quantizer.
The same head weight pack is read, with physical width 38784 and logical width
38720. Synthetic BF16 residuals and norm weights; real FP8 head weights.

Two warm B/A/A/B brackets, CUDA graphs amortizing submission overhead:

| Blocks × block rows | Head rows | Baseline ms | Candidate ms | Whole change | Input preparation µs |
|---|---:|---:|---:|---:|---:|
| 1 × 8 | 7 | 0.595465 | 0.595516 | +0.009% | 2.393 → 1.941 |
| 2 × 8 | 14 | 0.602583 | 0.599472 | -0.516% | 5.248 → 2.216 |
| 4 × 8 | 28 | 0.614061 | 0.608612 | -0.887% | 5.480 → 2.496 |
| 8 × 8 | 56 | 0.625773 | 0.619272 | -1.039% | 6.845 → 2.716 |
| 1 × 2 | 1 | 0.588470 | 0.588640 | +0.029% | 3.007 → 1.622 |
| 2 × 5 | 8 | 0.597887 | 0.596192 | -0.283% | 4.427 → 1.675 |
| 128 × 2 | 128 | 0.671109 | 0.647261 | -3.554% | 14.021 → 4.889 |
| 19 × 8 | 133 | 0.862984 | 0.851716 | -1.306% | 12.044 → 4.905 |

C=1/K=7 whole-head change is +0.009%, effectively neutral: **no demonstrated
C=1 head speedup**. Common wider K=7 batches improve about 0.52–1.04% in this
run. Producer-only timings show launch/memory savings but include occasional
microsecond-scale desktop noise; they are not engine speedups. M128 is an
edge-case padding check, not a supported K=7 concurrency performance claim.
No GB10/TP4 step/s, token/s or acceptance measurement. No fleet queue,
deployment, service stop or restart. FC and the target model's ordinary head
input producer are unchanged by this patch.

## Boundaries and validation

- Eight shapes: K=7 widths 1/2/4/8 plus K=1/K=4 and exact/crossed 128-row
  scale-tile boundaries. BF16 hidden, FP8 bytes, all MX bytes and final logits
  match exactly. Each has two changed-input graph replays, zero replay Torch
  allocation bytes, and separate graph outputs verified not to alias.
- Head observers receive the same selected BF16 rows; tests cover supplied
  output buffers and logical vocabulary slicing. Producer selection depends
  on the prepared cuBLAS head, not dynamic batch-width tuning.
- Greedy, sampled and batched draft proposals use the new input path. Plain
  block calls retain their full-hidden result and the existing CPU path.
- Execution proof requires `producer_mx` when a drafter is prepared. Shared
  mutable output caches are absent; every capture owns its scratch.
- No added persistent weights or workspace. Discarded residual output and
  anchor hidden rows are no longer materialized at this final boundary.
- Related CPU checks: 62 passed, six skipped (68 total). Host native build
  and 115 SM121 Triton variants compiled without a GPU. These checks do not
  prove GB10 execution or end-to-end generation.

## Runtime, receipts and reproduction

Pinned image `sha256:9f496f0dabe3a7b495d9b97181913cc20be1e4b3d3fcf2694407e34f24b3981b`,
Torch 2.13.0+cu132, CUDA 13.2, RTX 5050. Pack SHA and tested source hashes are
in `rtx5050/producer-final.json`. `run_5050.sh` uses the previously authorized
owned scratch on ost-97x and cooperative GPU lock, with a uniquely named
container, two CPUs and 4 GiB. It uses no fleet queue. Source identity is
checked before recording this document.

The initial producer assigned all neutral scale padding to one CTA. It was
bit-exact but made C=1 preparation slower; `single-cta-padding.json` and
`single-cta-padding.py` retain that rejected implementation. Final uses one
padding CTA per scale group in the same launch, and directly emits only
non-anchor hidden rows. Retained exploratory data is not the final verdict.
