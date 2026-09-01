# glm53_dflash_aot_guard

Disables FlashAttention's AOT split schedule for sliding-window DFlash and
DSpark draft attention groups. The schedule is derived from a model-global
window scan and is invalid when the drafter is the only FlashAttention
consumer, as it is for GLM-5.3's MLA target plus its FlashAttention DFlash2
drafter.

The guard runs after `SpecDecodeBaseProposer.set_attn` has built the drafter's
own attention groups. It only changes a builder when both conditions hold:

- `aot_schedule` is currently enabled; and
- that builder's KV-cache specification has a sliding window.

Full-attention drafters, target attention builders, eager execution, and every
non-DFlash proposer are unchanged. This is the narrow fix from upstream vLLM
PR #54374. That PR reports GLM-5.3-Flash + DFlash2 acceptance length recovering
from 1.000 to 5.542, but this repository does not claim that result until its
own serving gate is run.

Base preimage: `glm53:v13-b12x`, SHA-256 `bd7f4c63...`.
