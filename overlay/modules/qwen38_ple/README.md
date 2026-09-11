# qwen38_ple — the 51 GiB n-gram table, and the scale nobody reads

Qwen3.8-Flash-Next carries a PLE (per-layer embedding) n-gram table at layer 2:
`ngram_vocab_size_base` 20,000,000 split into `split_ngram_parts` 128 shards.
It is about 51 GiB of the checkpoint's 126, and on a DGX Spark host memory *is*
the GPU pool, so where it lives decides whether the model fits.

## `weight_utils.py`

The loader path. Two things it does that the stock one does not: keeps the
table out of device memory, and supports per-rank local direct-IO loading
(`DENEB_FST_LOCAL` + `--load-format fastsafetensors`) by forcing
fastsafetensors' `SingleGroup`, so the loader never opens its own cross-node
NCCL context — which dies on this RoCE fabric and is pointless when every node
holds the full checkpoint locally.

## `ple_layer.py`

`VLLM_PLE_CPU_OFFLOAD=1` keeps the table in host RAM. That path is the only
stock reader of the PLE scale, **and it is gated to `nnodes=1`**. So at TP>1
the on-device path runs instead, and upstream only builds the FP8 PLE embedding
method for `Fp8Config` — an NVFP4 checkpoint never gets one.

`DENEB_PLE_FORCE_FP8_EMBED=1` builds it anyway. Without it TP>1 has no reader
for the scale at all, which is why `FORCE_FP8_EMBED=1` is the profile default
rather than an experiment.

Both files are overrides and carry the image's preimage SHA. With the env knobs
at 0 the added branches are dead code identical to upstream, so mounting them
unconditionally is safe.
