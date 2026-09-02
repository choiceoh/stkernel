# glm53_indexer_gate_splitk

The sparse indexer's fp32 head-gate projection as a split-K Triton kernel,
behind `VLLM_GLM53_INDEXER_GATE_SPLITK` (default `0`, stock `torch.mm`).

## What the stock code does

`Indexer.forward` (`vllm/models/glm5next/nvidia/attention.py`) computes

```python
weights = torch.mm(hidden_states.float(), self._wp_fp32)   # [M, 4096] x [4096, 16]
```

once per full-attention layer, in fp32 on purpose: bf16 head-gates (~1e-2
error) flip near-tie pool rankings, and the ranking is what the sparse
attention selects. cuBLAS answers this shape with a two-block `gemmSN`
kernel: 47 us on an idle GB10, 86 us under CUPTI in the 2026-09-01 serving
trace, eleven times per decode step (0.95 ms/step CUPTI) for 256 KB of
weights. Two blocks on 48 SMs is the whole problem.

## What this module does

A split-K kernel (`glm53_indexer_gate.py`): one program per (row, K-slice of
512), each accumulating 16 outputs in fp32 and reducing with fp32 atomics
into a zeroed output. Used only when the knob is `1` **and** `M <= 16`
(decode: C=1 verify batches are M=8, C=2 M=16); every other shape, prefill
included, keeps `torch.mm`. The fused-indexer forward in
`glm53_prefill_fastpath.py` (`VLLM_GLM53_FUSED_K_GATE=1`) routes through the
same helper, so the two arms cannot disagree.

Both paths accumulate in fp32; only the summation order differs, so this is
**not bit-exact** with stock -- it is a numerics change and needs the
quality bracket, not just a timing one.

## Offline numbers (srv4 GB10, `probes/indexer_gate_check.py`, CUDA-graph replay)

| M | stock `torch.mm` | split-K | route |
|---|---|---|---|
| 1 | 15.5 us | 8.5 us | split-K |
| 8 | 50.0 us | 9.9 us | split-K |
| 16 | 50.1 us | 11.7 us | split-K |
| 32 | 15.9 us | 17.4 us | `torch.mm` kept |

Numerics over 300 trials / 2,480 rows (bf16 activations, fp32 weights of the
indexer's scale): max |diff| 2.4e-6 absolute, 6.7e-7 of the row's max gate,
0 top-1 flips, 0 top-4 set changes.

Ceiling: 11 layers x ~40 us = ~0.44 ms/step at C=1, about 0.65% of a 66 ms
step -- below the ledger's 1% boot threshold on its own. Ride it on another
bracket's boot (it is independent of `glm53_prep_fused` and
`glm53_async_dflash`), never on a boot of its own.

## Arming

`VLLM_GLM53_INDEXER_GATE_SPLITK` is a profile-declared key: pass it as caller
env, never through `EXTRA_ENV`.

```bash
VLLM_GLM53_INDEXER_GATE_SPLITK=1 bash launchers/start-glm53-nvfp4-tp4.sh
```

Verdict in the trace: the eleven `gemmSN` launches are replaced by eleven
`_gate_splitk_kernel` launches (no boot-log line; the helper is called per
layer inside the captured graph). Gate: quality 9/9, Korean 0/16, C=1 step/s
bracket base -> cand -> base.

Preimage: `attention.py` of `glm53:v13-b12x`
(`a0870c31...`), identical in `glm53:sm121-fi618`.
