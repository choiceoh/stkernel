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
sentinel that is removed before the kernel call. Decode replaces remote slots
with zero-weight repeats and submits at most 8 `top_k=1` rows per call. Stable
DFlash verification shapes use 8/16/32 tokens at C=1/2/4, so top-k=8 flattens
to 64/128/256 pair rows and 8/16/32 micro calls per MoE layer. One unsliced
fixed call selects the static backend observed to hang on this EP geometry.
This establishes a safe dispatch shape; no end-to-end throughput win has been
established. Prefill (`tokens * top_k > 640`) drops remote slots instead of
paying GEMM for them. Both direct paths return before the large `top_k=8`
graph wrapper and its capacity probe are constructed.

An experimental third direct lane is mounted by `b12x_zero_weight_micro` but
stays off by default. Exact `VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1` requires
`VLLM_B12X_EP_NO_DUMMY=1` and `VLLM_B12X_EP_DISABLE_MICRO=0`. It preserves the
same physical E=72 weights and leaves each remote route as sentinel id 72 at
weight zero for the micro kernel to discard before row materialization. Only
stable 8/16/32-token GLM decode shapes are admitted, as 1/2/4 disjoint
eight-token top-k=8 calls; every mismatch uses this module's existing fixed
top-k=1 fallback. A separate pinned top-k=8/max_rows=64 workspace preserves the
#147 CUDA-graph lifetime contract. This has no GPU correctness or E2E win yet.

Fixed decode uses one shared, strongly held 8-row workspace. Compact prefill
continues to use FlashInfer's replaceable functional cache, but growing that
cache can no longer invalidate addresses captured by decode CUDA graphs.
`VLLM_B12X_EP_NO_DUMMY=0` is the rollback path: it restores the historical
`E = num_local + 1` zero-weight dummy and the wrapper-backed call. The remap
always writes into preallocated scratch (`out=` / in-place), and vLLM's EP
all-reduce (DP=1) combines the partial hidden states.

`ENABLE_EP=1` on the glm53 launcher. Off by default — the TP-sharded path is
the measured one. EPLB is refused (`_supports_parallel_config`).
`VLLM_B12X_EP_COMPACT=0` keeps every batch fixed-shape; with the default
no-dummy path, remote slots become slice-local repeats.
`VLLM_B12X_EP_DISABLE_MICRO=1` is a plain-static diagnostic and is rejected
with the default no-dummy path; also set `VLLM_B12X_EP_NO_DUMMY=0` to use it.
Setup also verifies the pinned FlashInfer `_MICRO_MAX_TOKENS >= 8`; a missing
or smaller private boundary fails closed instead of silently selecting static.
PIECEWISE graphs force compaction off so prefill capture keeps a fixed shape.
