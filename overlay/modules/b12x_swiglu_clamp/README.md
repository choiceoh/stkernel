# b12x_swiglu_clamp

Lets --moe-backend flashinfer_b12x be accepted for checkpoints that set swiglu_limit, by forwarding the clamp FlashInfer already takes. Any b12x MoE whose checkpoint carries swiglu_limit.

Base contract taken from `glm53:v9`.

**Superseded from `v11` onward.** v13's `flashinfer_b12x_moe.py` already has
this clamp -- same comment, same fields -- plus thirty lines this v9-derived
copy lacks, and its `nvfp4.py` matches ours byte for byte. Mounting it over v13
would only remove those thirty lines, so the glm53 profile no longer loads this
module. Kept for a v9 image.
