# SF6 direct prefill

The default `VLLM_GLM53_B12X_STATIC_V2=t,r,sf6` uses compressed scales in
decode and every prefill consumption path (operator adoption, 2026-09-09). The
prior PR #498 v5 campaign remains frozen at 932b3fc4 in its own checkout.
Its measurements cannot validate this implementation.

The packed format remains SF6-v1: 2048 original scale bytes become 1552
bytes with exact integer reconstruction. A stage spanning more than 64 byte
codes declines both planes for that layer; no clamping is permitted.

* Static decode keeps its K256/N256 reform and existing 1552-byte loads.
* Static prefill keeps its original tile geometry. FC1 loads two consecutive
  packed stages (3104 bytes) into its existing 4096-byte shared stage. FC2
  gathers only the selected row half in five aligned loads totaling 784
  bytes, expanding in its existing 1024-byte shared stage.
* Dynamic prefill reads packed global bytes into producer registers and
  expands them into its existing shared scale stages. Scale descriptors and
  scale DMA transaction bytes are absent. Stage acquisition waits for the
  prior consumer; publication occurs after shared writes and warp sync.
  The inherited gated kernel supports M128 tiles. SF6 workspace allocation
  and cache selection both use M128, with its existing valid-row tail handling
  for smaller requests. Vendor gated.py remains byte-identical and is SHA pinned.

There is no global raw reconstruction buffer. After the full weight load,
above the traced model forward and before profiling/KV sizing, the model
prepares immutable packed owners. Eligible non-EP layers release registered
raw scale Parameters, quantization descriptors, MMA views and dispatcher
cache aliases. Incompatible backends and unrepresentable layers retain both
original planes. Packed owners cannot dispatch into raw-scale kernels;
changing sealed weights requires loading a fresh model.

For the 42 eligible layers observed in the SF6-direct boot, each rank logged
4,756,340,736 raw scale bytes released and 3,604,414,464 packed bytes retained.
Compared with raw-only scales, storage changes from 4.42969 GiB to 3.35687 GiB,
or 1.07281 GiB less per rank. The larger 4.42969 GiB release is the original
allocation removed after packing; it is not the net saving over raw-only.
These are tensor-size calculations, not matched host-memory measurements.
Restart with `VLLM_GLM53_B12X_STATIC_V2=t,r` to use raw scales again; changing
the flag on a live packed-only model cannot restore released Parameters.

The operator chose this default after reviewing the completed onepass and
isolated baseline retry. Raw decode was 20.281702 vs 20.206613 step/s, but
MHC consumer activation differed between runs; no SF6-only speedup is claimed.
The original A/B INVALID verdict is preserved. See the
[adoption record](../measurements/glm53_sf6_default_adoption_20260909/README.md)
for the decision, memory accounting and validation.

## Validation

CPU tests exercise the actual pack/unpack and producer methods against
independent byte maps, staged overwrite barriers, raw-reference lifetime,
fallback admission, wrapper reuse and dynamic runtime arguments. The
production-image compiler must additionally validate descriptors, shared
layouts, DSL lowering and ptxas. Neither replaces CUDA numerical/ordering
checks or canonical onepass measurements.

From a clean, composed remote checkout, preserving the existing GPU queue:

```bash
REPO="$PWD" bash /home/choiceoh/stkernel/bench/fleet.sh run --cpu \
  sf6-direct-cpu 15 'SF6 direct prefill compiler and owner checks' -- \
  python3 probes/run_decode_transport_sf_cpu.py --out /absolute/fresh/evidence \
  --stage contracts --stage sf-m6 --stage sf-m8 --stage sf-m16 \
  --stage sf-compat --stage sf-expand --stage sf-direct-tm128 --stage sf-wrapper
```

The runner uses an immutable serving image, runc, no GPU/network visibility,
two CPU cores and bounded memory/swap. Selected-stage receipts do not claim
coverage of unrelated transport modes. GPU testing uses only the canonical
onepass workflow; the legacy standalone numerical harness is not admission
for an independent GPU run.

The decorated-wrapper CPU stage invokes the installed FlashInfer wrapper
with raw scale arguments set to None and an explicit packed owner. It mocks
only dispatch/launch and checks both workspace routes; it does not claim
trace-dump mode, actual model boot, CUDA graph replay or kernel numerics.
