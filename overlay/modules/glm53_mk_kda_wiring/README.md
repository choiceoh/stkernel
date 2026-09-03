# glm53_mk_kda_wiring

The image-bound hook for `MK_SEG_KDA`: GLM-5.3's linear-attention block
(`vllm/models/glm5next/nvidia/kda.py`) with the megakernel takeover and the
shadow epilogue spliced in. Behind `VLLM_GLM53_MK_KDA=1` /
`VLLM_GLM53_MK_KDA_SHADOW=1` (master `VLLM_GLM53_MEGAKERNEL=1`); disarmed, the
file runs the stock forward body verbatim.

It used to be the third row of `glm53_megakernel`'s manifest. It is its own
module now because it is the ONE thing in that module that is GLM's: the
container path `vllm/models/glm5next/` does not exist in another model's
image, and its pinned preimage would fail the deploy gate there. The kernel
core (`glm53_megakernel.py` + `.cu`) binds only
`vllm/model_executor/layers/`, relative to the profile's `TARGET_PREFIX`, and
carries no model file at all -- which is what lets a second profile mount it.

The split is bytes-neutral for GLM: `glm5next_kda.py` moved unmodified, and
`compose-overlays.sh glm53` renders the same `build/glm53/` as before (the
manifest's rows are the same three, redistributed across two modules).

## Requires

`glm53_megakernel` (the kernel it calls) and `glm53_fp8_dense`: the KDA packs
are built from a stock fp8-dense arm (`isinstance(in_m, Fp8DenseMethod)`), so
without that module the takeover has no weights to stream. That requirement
used to sit on the core module; it belongs here, on the hook that actually
needs it.

## Gate

Unchanged, and it is the strict one: MK-KDA's state-index contract is checked
by the boot self-test AND by eager shadow mode
(`VLLM_GLM53_MK_KDA_SHADOW=1`, outputs and next-step states diffed every 64
calls at the e2m1 by-design class). Run shadow on a bench boot, read the log,
then decide the arm. See `overlay/modules/glm53_megakernel/README.md`.
