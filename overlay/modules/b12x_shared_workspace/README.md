# b12x_shared_workspace

One `B12xMoEWrapper` per geometry instead of one per layer.

Each wrapper carries graph-stable scratch sized by the MoE geometry. Measured on
this deployment (288 experts, top_k 8, hidden 4096, intermediate 2048,
max_num_tokens 2048):

```
wrapper 1개: allocated +541.1 MiB
MoE 층 43개: 22.72 GiB per rank
공유 시 절감: 22.19 GiB
```

GLM-5.3-Flash has 43 MoE layers of identical geometry, so the duplicate buffers
were a quarter of the entire GMU budget. The arithmetic matched what the engine
reported: 45.8 GiB of weights + 22.7 GiB of wrappers against an 87.4 GiB budget
at GMU 0.73 leaves ~16 GiB for KV, and the engine logged 16.52 GiB.

Sharing is safe because MoE layers run one after another on the same stream: the
wrapper writes into its buffers on every call and nothing outlives the call.
Values are held weakly, so the wrapper dies with the last layer referencing it.

Same idea as vllm-project/vllm#48698, written against the file this image ships.
(#53081 shares the *workspaces* rather than the wrapper, but needs FlashInfer's
`shared_static_workspace` from flashinfer-ai/flashinfer#4603, which this build --
0.6.18.dev20260819 -- does not have.)

## Expert parallelism

The fused kernel still raises at `num_local != num_experts` (flashinfer #3383:
spec sizes weights by `weight_E`, then illegal-address on local tensors). This
module does not lift that. With the default `VLLM_B12X_EP_NO_DUMMY=1`, EP keeps
only the local weight rows (`E = num_local`) and remaps remote routes to a
sentinel that is removed before the kernel call. The default #146 decode
fallback replaces remote slots with zero-weight repeats and submits at most
eight `top_k=1` rows per call. Stable 8/16/32-token C=1/2/4 shapes therefore
take 8/16/32 micro calls per MoE layer and never enter the static backend that
hangs on this EP geometry. Its `pair_out` remains zero-initialised and masked
before `index_add_`.

Exact `VLLM_B12X_EP_STOCK_TOPK_MICRO=1` enables a narrower experiment for the
pinned GLM E=288/global, E=72/local, K=4096, N=2048, top-k=8 geometry and only
the stable 8/16/32-token shapes. Remote sentinels become same-token local IDs
at router weight zero, then the original token-major tensors run in balanced
chunks of at most five tokens / 40 routed rows. That removes pair flatten,
sort, gathers, pair output, mask, and `index_add_`, reducing the call counts to
2/4/7. The knob is strict, latched, default-off, and mutually exclusive with
`VLLM_B12X_EP_ZERO_WEIGHT_MICRO`; live GPU numerics and graph replay are still
pending. Every shape outside the exact gate retains the #146 fallback.

Prefill (`tokens * top_k > 640`) drops remote slots instead of paying GEMM for
them. All direct paths return before the large graph wrapper and its capacity
probe are constructed.

An experimental third direct lane is mounted by `b12x_zero_weight_micro` but
stays off by default. Exact `VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1` requires
`VLLM_B12X_EP_NO_DUMMY=1` and `VLLM_B12X_EP_DISABLE_MICRO=0`. It preserves the
same physical E=72 weights and leaves each remote route as sentinel id 72 at
weight zero for the micro kernel to discard before row materialization. Only
stable 8/16/32-token GLM decode shapes are admitted, as 1/2/4 disjoint
eight-token top-k=8 calls; every mismatch uses this module's existing fixed
#146 fallback. A separate pinned top-k=8/max_rows=64 workspace preserves the
#147 CUDA-graph lifetime contract. This has no GPU correctness or E2E win yet.

The default fallback and both experiments hold separate shared workspaces:
top-k=1/r8, stock top-k=8/r40, and overlay top-k=8/r64. Compact prefill keeps
using FlashInfer's replaceable functional cache, but growing that cache cannot
invalidate addresses captured by decode CUDA graphs.
`VLLM_B12X_EP_NO_DUMMY=0` is the rollback path: it restores the historical
`E = num_local + 1` zero-weight dummy and the wrapper-backed call. The remap
always writes into preallocated scratch (`out=` / in-place), and vLLM's EP
all-reduce (DP=1) combines the partial hidden states.

`ENABLE_EP=1` on the glm53 launcher. Off by default — the TP-sharded path is
the measured one. EPLB is refused (`_supports_parallel_config`).
`VLLM_B12X_EP_COMPACT=0` keeps every batch fixed-shape; with the default
no-dummy path, remote slots become slice-local zero-weight repeats.
`VLLM_B12X_EP_DISABLE_MICRO=1` is a plain-static diagnostic and is rejected
with the default no-dummy path; also set `VLLM_B12X_EP_NO_DUMMY=0` to use it.
Default setup verifies `_MICRO_MAX_TOKENS >= 8`. The stock experiment also
requires `_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK >= 40` and automatic backend
selection. A missing or smaller private boundary fails closed instead of
silently selecting static.
PIECEWISE graphs force compaction off so prefill capture keeps a fixed shape.
