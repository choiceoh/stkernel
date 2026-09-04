# moe_gate_sm121

Fused small-M MoE router gate for GB10. Any MoE model on this fleet: the
image's GateLinear tiers are gated on SM90/SM100 families, so sm_121 always
takes the bf16 Tier-4 fallback. Measured C=1 +3.5% on DeepSeek-V4-Flash.

Armed by `VLLM_DSV4_GATE_FUSED=1`; M > 32 keeps the stock path.

## Tried and rejected: the router's top-k as an epilogue (2026-09-05)

A last-arriver epilogue of this launch reproduced vLLM's `topk_sigmoid`
routing bit for bit (the recipe is in MEASUREMENTS 29차: the build's sigmoid
rounds like `1/(1+exp(-x))` with an IEEE division, the renormalization sum
in rank order), but one CTA doing the whole selection cost more than the
kernel it replaced: 28.2 -> 31.3 us/layer at M=8, 28.1 -> 45.6 at M=16,
34.3 -> 86.8 at M=32 (quiet GB10, graph replay). Not shipped; a grid barrier
would fix the parallelism but cannot coexist with the megakernel's 48-block
barrier kernels on the same stream.
