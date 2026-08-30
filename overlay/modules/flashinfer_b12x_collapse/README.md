# flashinfer_b12x_collapse

Repairs flashinfer 0.6.18's borrowed _collapse_to_vmk at the lender, so all four MoE kernel classes work. Any b12x MoE on flashinfer 0.6.18. Production runs 0.6.17, where the gap does not exist.

Base contract taken from `glm53:v9`. (The README said `v12-b12x`; the pinned
sha is v9's. Nothing verified these contracts until the launcher started
reading the manifest, so the claim went unchallenged.)

**Superseded from `v11` onward.** The image carries this file byte-identically,
so the glm53 profile no longer loads this module. Kept for a v9 image.
