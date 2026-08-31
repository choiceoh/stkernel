# glm53_kda_prefill_regime

Default-off, two-bucket Triton autotune-cache split for the generic
flash-linear-attention KDA chunk path that GLM-5.3 actually imports.

The live `glm53:v13-b12x` model layer imports
`chunk_kda_with_fused_gate` from
`vllm.third_party.flash_linear_attention.ops.kda`; it does **not** call the
Kimi-K3 vendored KDA fork. This module therefore replaces the generic FLA
`kda.py` plus its external `chunk_delta_h.py` dependency only.

## Contract and bucket

The arm requires the exact string `VLLM_GLM53_KDA_PREFILL_REGIME=1` and all of
the following live contracts:

- compute capability 12.1, model type `glm5_next_text`, TP=4,
  `max_num_batched_tokens=8192`;
- packed/varlen chunk KDA with B=1, H=16, K=V=128;
- bf16 q/k/v/raw-g, the production fp32 sigmoid-beta and fp32 initial state;
- bounded gate (`safe_gate=True`, `lower_bound=-5.0`) and
  `output_final_state=True`.

Any mismatch returns regime `0`, the unsplit stock/short config domain. On an
admitted call, the model/config/device portion of the contract is latched once while
vLLM constructs the GLM KDA layer (the only interval in which vLLM exposes its
current config). Forward dispatch reads that latch and computes the bucket once
from tensor and `cu_seqlens` shapes:

```text
regime = int(total_T >= 1024 * (cu_seqlens.numel() - 1))
```

Thus packed average sequence length below 1024 stays in bucket 0 and long
prefill uses bucket 1. No `cu_seqlens` value is read and raw T is never an
autotune key, so each existing shape key has at most two config-cache entries.
The 1024 boundary is an unmeasured hypothesis; the profile remains off until
an engine-down bracket shows a repeatable prefill win.

Arm a bracket from the head checkout with the profile knob as a caller env:

```bash
VLLM_GLM53_KDA_PREFILL_REGIME=1 \
  launchers/start-glm53-nvfp4-tp4.sh
```

Do not pass this profile-owned key through `EXTRA_ENV`: the launcher's later
profile env block would otherwise emit the default `0` after it. A direct
caller value is preserved across profile loading and reaches every rank as
`1`.

## Core-six ownership

One call-level scalar is threaded through exactly the production chunk
Autotuners:

1. KKT inter-subchunk,
2. KKT intra-subchunk,
3. W/U recompute,
4. gated-delta state propagation (`chunk_delta_h.py`),
5. GLA output, and
6. bounded gate + chunk cumsum.

`AUTOTUNE_REGIME` is a regular runtime scalar, appears in each autotune key,
and is explicitly listed in `do_not_specialize`. It therefore splits config
selection without creating separate kernel-code specializations. Low-level
wrappers default to `0`; only `chunk_kda_with_fused_gate_fwd` derives the
request bucket, once, and forwards the identical value to all six.

The decode path uses `fused_recurrent_kda`, not these chunk Autotuners. This
module intentionally leaves fused recurrent, l2norm, solve-tril and the
standalone non-cumulative gate kernel unchanged.

## Validation and rollback

Regime-selection rollback is the env value alone; malformed values, `0`, and
unset all retain regime 0 and skip even the model-init capability/config
lookup. The mounted source still has the added unused scalar ABI and therefore
is not byte-for-byte stock; full rollback removes this module from the profile
and recomposes the overlay. Before adoption, run `probes/kda_prefill_bench.py`
in a fresh GPU container. The first admitted long request creates a second
autotune entry for each core kernel (including the 24-config KKT-inter and
36-config GLA-output sweeps), so prewarm both short and long shapes before
timing. Use matched fresh-cache arms in both `short -> long` and
`long -> short` order; do not let the first 75K user request absorb the tune
sweep. Then bracket a 75K prompt and shorter/multi-sequence controls with
retrieval, Korean-corruption and output/state comparisons.

The standalone probe imports the base image's production entry and directly
sweeps its config inventory. It does not construct the GLM layer or prove that
this overlay's init latch and 0/1 dispatch arm in service; that proof belongs to
the full-engine bracket above.

Live preimages from srv4 `glm53-worker`:

- `kda.py`: `ac2260c84a36936ad7d56ef63dbceb4618b2c499d7637e08b407f0cd706f9d02`
- `chunk_delta_h.py`: `1b3ad391f939d9443c6b7adb19e57fe381bd5dccea064e8417a4f85b0e713b26`

No GPU kernel was launched while preparing this overlay.
