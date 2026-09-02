# glm53_indexer_gate_splitk

The sparse indexer's fp32 head-gate projection as a deterministic split-K
Triton path, behind `VLLM_GLM53_INDEXER_GATE_SPLITK` (default `0`, stock
`torch.mm`). This README is the single home of the measured numbers; the
ledger and RUNBOOK cite it.

## What the stock code does

`Indexer.forward` (`vllm/models/glm5next/nvidia/attention.py`) computes

```python
weights = torch.mm(hidden_states.float(), self._wp_fp32)   # [M, 4096] x [4096, index_n_heads]
```

once per full-attention layer, in fp32 on purpose: bf16 head-gates (~1e-2
error) flip near-tie pool rankings, and the ranking is what the sparse
attention selects. The fleet checkpoint (`glm53-redhat-nvfp4`) has
`index_n_heads = 32`, so the weight is `[4096, 32]` (the first version of this
module admitted N <= 16 and never ran on the fleet; the offline numbers were
taken on a synthetic N=16 -- corrected 2026-09-03). cuBLAS answers the M<=16
shape with a two-block `gemmSN` kernel: 88 us DRAM-cold on an idle GB10 and
88 us in the 2026-09-01 serving trace, eleven times per decode step, for
512 KB of weights.

## What this module does

Two small kernels (`glm53_indexer_gate.py`): program `s` of the first loads
one `[128, N]` weight slice once and multiplies it against every row of x
(bf16 loaded and cast to fp32 in registers -- the same exact conversion
`.float()` does), writing a `[16, N]` partial; the second sums the 32 partials
in a fixed order. No atomics and no memset, so the result is **bitwise
reproducible run to run and identical on every TP rank** (the indexer is
replicated per rank and the ranks' top-k pool selections must agree). Used
only when the knob is `1` **and** `splitk_applicable` admits the shape:
2-D x with unit inner stride, M <= 16 (C=1 verify batches are M=8, C=2 M=16),
`x.shape[1] == w.shape[0]`, fp32 contiguous w with N <= 32, K a multiple of
128. Everything else -- prefill, C>=3, unexpected layouts -- keeps `torch.mm`.
The fused-indexer forward in `glm53_prefill_fastpath.py`
(`VLLM_GLM53_FUSED_K_GATE=1`) routes through the same helper.

Both paths accumulate in fp32; only the summation order differs, so this is
**not bit-exact** with stock -- it is a served-numerics change and carries the
full quality bracket.

## Offline numbers (GB10, `probes/indexer_gate_check.py --config <checkpoint>/config.json`, 2026-09-03)

CUDA-graph replay with 46 distinct weights cycled (DRAM-cold), N=32, K=4096:

| M | stock `torch.mm` | split-K | route |
|---|---|---|---|
| 1 | 20.8 us | 9.5 us | split-K |
| 2 | 51.0 us | 10.1 us | split-K |
| 4 | 86.4 us | 11.1 us | split-K |
| 8 (C=1) | 89.5 us | 12.7 us | split-K |
| 12 | 93.9 us | 14.7 us | split-K |
| 16 (C=2) | 87.7 us | 16.8 us | split-K |
| 24 (C=3) | 17.3 us | -- | `torch.mm` kept |
| 32 (C=4) | 17.7 us | -- | `torch.mm` kept |

Numerics over 200 trials / 1,571 rows (bf16 activations x1.5, fp32 weights
x0.02): max |diff| 2.4e-6 absolute, 7.2e-7 of the row's max gate, 0 top-1
flips, 0 top-4 set changes; 50 repeated launches bit-identical; strided x,
K mismatch, M=0 and M=17 all route to `torch.mm` (checked by the probe, which
exits non-zero on any failure).

Ceiling: 11 layers x (89.5 - 12.7) us = 0.84 ms/step at C=1, **~1.2%** of the
71 ms step (C=2: 0.78 ms). Below the ledger's boot bar on its own; it rides
the EXP-7 boot (bit-exact, so this stays the only numerics axis on that
boot) -- not EXP-8, whose gate is that acceptance must not move.

## Arming and verdict

`VLLM_GLM53_INDEXER_GATE_SPLITK` is a profile-declared key: pass it as caller
env, never through `EXTRA_ENV`.

```bash
VLLM_GLM53_INDEXER_GATE_SPLITK=1 bash launchers/start-glm53-nvfp4-tp4.sh
```

Boot log: the helper announces its routing once per shape --
`[indexer-gate] VLLM_GLM53_INDEXER_GATE_SPLITK=1: x(8, 4096) torch.bfloat16 @ w(4096, 32) -> split-K`.
A line ending in `stock torch.mm (shape not admitted)` means the knob is on
but the kernel never runs (that is how the N=16 version would have shown up).
If `glm53_model_wiring` is mounted without this module, its fastpath logs
`[indexer-gate] ... not mounted -> stock torch.mm`. In a trace the eleven
`gemmSN` launches become eleven `_gate_splitk_partial_kernel` +
`_gate_splitk_reduce_kernel` pairs.

Gate: quality 9/9, Korean 0/16, pos-1 acceptance +/-2pct, C=1 step/s bracket
base -> cand -> base.

## Shape of the overlay

`attention.py` is overlaid whole (2 changed lines, preimage `a0870c31...` of
`glm53:v13-b12x`, identical in `glm53:sm121-fi618`), the repo's standard for
image-bound files: an image bump that touches `attention.py` fails the
launcher's preimage check and stops the boot even with the knob at `0`, which
is the intended signal (the alternative, a runtime copy of `Indexer.forward`,
would make a second source of truth for that function next to the fastpath's).
No other module may own `attention.py`. `requires` is empty: the wiring
fastpath optionally imports this module, not the reverse.
