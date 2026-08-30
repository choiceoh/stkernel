# moe_gate_sm121

Fused small-M MoE router gate for GB10. Any MoE model on this fleet: the
image's GateLinear tiers are gated on SM90/SM100 families, so sm_121 always
takes the bf16 Tier-4 fallback. Measured C=1 +3.5% on DeepSeek-V4-Flash.

Armed by `VLLM_DSV4_GATE_FUSED=1`; M > 32 keeps the stock path.
