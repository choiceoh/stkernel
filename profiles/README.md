# Profiles

A profile names the modules a model loads. Nothing else decides what gets
mounted.

```
MODULES="tp_oneshot_ar spec_fp8_head mla_indexer dsv4_model ..."
```

`launchers/compose-overlays.sh <profile>` renders `build/<profile>/`: the flat
directory and single `manifest.tsv` that the deployer, the launcher preflight
and the 4-node SHA-256 verification already expect. Splitting the repo into
modules did not change anything the fleet sees.

A module may declare what it cannot run without in a `requires` file, and a
profile that omits a requirement aborts the compose rather than failing later
as an ImportError inside a rank.

Two modules may not claim the same source filename, and two rows may not bind
the same container path -- either aborts the compose. That is the check that
keeps a module honest: if a second model needs a different version of a module's
file, the module was never model-agnostic and has to be split, not overridden.

| profile | model | modules | state |
|---|---|---|---|
| `dsv4` | DeepSeek-V4-Flash-0731 | 16 | production |
| `glm53` | GLM-5.3-Flash NVFP4 | 2 | bring-up, blocked |
| `qwen38` | Qwen3.8-Flash-Next NVFP4 | 1 | bring-up |

`glm53` carries two modules of its own and can load none of `dsv4`'s: its image
installs to dist-packages rather than the venv site-packages, and one of its
modules targets flashinfer rather than vllm at all. That is what `TARGET_PREFIX`
is for -- the allowlist for container paths describes an image, so it belongs to
the profile. A module binding outside it aborts the compose.

`qwen38` stays at one module: it ran on stock image code, and its b12x path is
closed rather than pending (MEASUREMENTS.md).

A profile also carries the serving knobs that are the model's rather than the
fleet's -- backend, speculative depth, draft placement -- and, where a bring-up
is blocked, says so and names the one flip that would isolate the cause.
