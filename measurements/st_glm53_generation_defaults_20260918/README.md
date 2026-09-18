# Preserve the original GLM-5.3 sampling default

The production Red Hat metadata carries `temperature: 1.0` but omits
`top_p`. The original [ZAI generation_config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/generation_config.json)
and the NVIDIA NVFP4 derivative carry `temperature: 1.0, top_p: 0.95`
(checked 2026-09-18). The Red Hat file is copied faithfully; ST did not lose
the field during metadata preparation.

Before this change, omitted `top_p` in chat/text completions reached the
adapter's unrestricted `1.0` default. The GLM-5.3 profile now supplies `0.95`
only when the metadata omits that key. Explicit checkpoint values, including
`1.0`, and explicit request overrides still win. No other profile changes.

`temperature` and `top_p` control different parts of sampling. This omission
did not set the temperature to 1; the metadata already sets temperature 1.
The change restores the original nucleus cutoff and does not switch to
greedy decoding.

Deneb's GLM profile has no sampling override and uses the server defaults.
The raw `/v1/engine/completions` dialect deliberately does not inherit model
defaults; reproduction callers must continue to spell out both values.

## Validation and incident limit

The four new CPU regression tests cover derivative metadata, explicit
checkpoint policy, real chat/text HTTP admission, and the raw engine dialect.
The HTTP cases cover omitted values, `top_p=0.8`, explicit `top_p=1`, and
`temperature=0`. Seventy existing OpenAI dialect tests also pass.

This is **not an incident recovery claim**. The prior same-50,005-ID replay
with explicit `top_p=0.95`, temperature 1 and seed 7 produced 415 malformed
tokens, with zero cached tokens. That native device/MTP replay used source
`654f42cad5cd5123e28a46d2b4b800701fdbbbac`; see the
[`top-p095-device-t1` receipt](../st_telemachus_quality_20260917/precision-control-evidence.json).
It establishes that restoring the default alone did not recover that build.
It is not a matched performance comparison against this change.

No private prompt, output, or logit data is included here. Production must
boot the new source before the fallback takes effect; editing the repository
does not modify an already running process.
