# glm53_model_wiring

Constructs the router through `DenebGateLinear` instead of `GateLinear`, so
GLM takes the fused small-M gate. The kernel itself is `moe_gate_sm121`;
this is only the construction site, which is why it is a separate module.

It also installs a cache-only sparse-indexer path when the separate
`VLLM_GLM53_SM121_MLA_PREFILL=1` arm admits a fresh dense-MLA prefill. Dense
MLA never consumes the indexer's sparse scores, so the path projects and
normalizes only K, computes the kpool gate, and lets the existing kpool op
write its index-K and persistent tail caches. It avoids the otherwise dead
query projection, FWHT/FP8 quantization, fp32 head-weight projection/scaling,
and the unused rows of the fused K/head-weight projection before the custom op.
K and gate consume the same hidden rows, so their bf16 source weights are also
combined once after checkpoint loading. The cache-only path produces both with
one 256-row GEMM instead of two independent 128-row GEMMs. The extra weight is
a non-persistent buffer; the original parameters stay intact for every fallback.

The model path and kpool op share the same request-level no-consumer predicate.
CUDA graph capture, mixed decode/prefill, cached-context or long MQA prefill,
profiling metadata, module-name drift, and any production shape/dtype drift
fall back to the original `Indexer.forward`. When the dense-MLA env arm is not
the exact value `1`, the installer does not modify `Indexer.forward` at all.
If the fused weight cannot be built, that layer also uses the original path.
Rollback remains that single env value.

Base contract from `glm53:v13-b12x`.
