# dsv4_model

V4-Flash model body. Also carries DenebGateLinear, the fused small-M MoE router gate -- that part is NOT V4-specific (the image's GateLinear tiers are gated on SM90/SM100 families, so every MoE on sm_121 falls to the bf16 Tier-4 path) and is a standing candidate to lift into a module of its own.
