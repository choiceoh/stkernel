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
