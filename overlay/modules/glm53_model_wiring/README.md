# glm53_model_gate

Constructs the router through `DenebGateLinear` instead of `GateLinear`, so
GLM takes the fused small-M gate. The kernel itself is `moe_gate_sm121`;
this is only the construction site, which is why it is a separate module.

Base contract from `glm53:v13-b12x`.
