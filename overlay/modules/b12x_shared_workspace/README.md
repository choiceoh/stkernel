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
module does not lift that. When `--enable-expert-parallel` is on it constructs
the wrapper as a local-only MoE (`E = num_local + 1`), remaps global top-k ids
onto that space, and parks every remote slot on a dummy expert at scale 0 so
dynamic FC2 quant cannot bleed into a real expert. The remap writes into
preallocated scratch (`out=` / in-place) so decode CUDA graphs stay
alloc-free. Decode replaces remote slots with zero-weight repeats of local
pairs and submits at most 8 `top_k=1` rows per call. That keeps C=1/2/4 on
FlashInfer's micro backend; flattening 16/32 rows into one call selects the
static backend that hangs on this EP geometry. A mixed call only repeats rows
already present in that same call so per-call FC2 amax is unchanged. Prefill
(`tokens * top_k > 640`) drops dummy slots instead of paying GEMM for ~3/4
remote routes. vLLM's EP all-reduce (DP=1) combines the partial hidden states.

`ENABLE_EP=1` on the glm53 launcher. Off by default — the TP-sharded path is
the measured one. EPLB is refused (`_supports_parallel_config`).
`VLLM_B12X_EP_COMPACT=0` keeps every batch fixed-shape; with
`VLLM_B12X_EP_NO_DUMMY=1` (default), remote slots become slice-local repeats.
PIECEWISE graphs force compaction off so prefill capture keeps a fixed shape.
