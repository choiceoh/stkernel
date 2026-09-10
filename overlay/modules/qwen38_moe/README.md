# qwen38_moe — the shared expert as an eleventh slot

Qwen3.8-Flash-Next routes top-10 of 512 experts and adds a shared expert every
token uses. This module folds that shared expert into the routed grouped GEMM
as one more group, instead of running it as a separate dense GEMM.

## Why it is possible at all

`moe_intermediate_size` and `shared_expert_intermediate_size` are both **640**.
The shared expert has exactly a routed expert's shape, so a grouped GEMM can
host it. A different shared size would need its own GEMM, and
`fuse_expert_weights` refuses rather than reshaping.

## Why it is worth doing

Alignment, not just the saved launch:

| | intermediate | gate+up rows | % 128 |
|---|---:|---:|---:|
| shared expert, TP-split over 4 | 160 | 320 | 64 |
| shared expert, fused slot | 640 | 1280 | **0** |

Expert parallelism already gives the routed experts the second row — 512
experts split 4 ways is 128 whole experts per rank, intermediate stays 640.
Fusing hands the shared expert the same one.

On this fleet that is not a preference. Padding a misaligned FP4 intermediate
is recorded as booting and then destroying the model (`'안녕하세요'` → `'1'`,
MEASUREMENTS.md), so the only safe move is to not be misaligned.

## The constraint that makes it non-trivial

**The fused slot is rank-local and never enters the all-to-all.** Every token
uses the shared expert; routing it as a real expert id would send the whole
batch to whichever rank owns that id. So the shared expert is replicated on
every rank — about 150 MiB per rank across 48 layers at NVFP4 — and addressed
by the sentinel `-2`, which `dispatch_ids` strips and `local_ids` resolves to
the same replicated index everywhere.

`-1` is already "no expert" in this stack's sparse contracts, so `-2` is
distinguishable from both a real id and from emptiness.

## The ordering trap

`norm_topk_prob` is true: routed weights are normalised **across the top-k**.
The shared gate is an independent sigmoid and is not part of that
distribution. Appending the slot before normalising renormalises every routed
weight against a value that does not belong to them — every shape correct,
every value finite, every token slightly wrong. `probes/qwen38_shared_fuse.py`
carries that as a control and measures it at 8.8e-04 on its inputs, against
a fused result that is bit-identical (max |d| 0.0) to routed + gate*shared.
