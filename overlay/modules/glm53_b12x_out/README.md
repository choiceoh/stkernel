# glm53_b12x_out

Takes over flashinfer's `b12x_moe.py` to add one thing: `B12xMoEWrapper.run(...,
out=...)`, which makes the caller's buffer the MoE kernel's scatter target.
Without it, every MoE layer materializes its result into the wrapper's static
workspace and the caller (our `flashinfer_b12x_moe.py`) copies it out — two
writes of the same bytes per layer, ~42 copy kernels/step on this lane. With
`VLLM_B12X_DIRECT_OUT=1` (default) the apply passes `out=output` and returns.

Capture-safe by the same evidence the copy provided: the replaced `copy_`
wrote that exact address under capture and replays, so a direct kernel write
to it does too. The base-preimage contract pins this fork to the image's
flashinfer file (43a12178…) — an image bump that touches b12x_moe.py aborts
deploy loudly instead of silently dropping the patch.

Only glm53 mounts this module (dsv4's MoE path does not go through
b12x_shared_workspace).
