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

## `qwen38_ple_ssd.py` — the table leaves the device (2026-09-11)

`DENEB_PLE_SSD=1` swaps the FP8 embedding method for one whose rows live on
SSD. `VocabParallelEmbedding` keeps everything it owns — the row partition
(`[r*per, (r+1)*per)`, `per = padded_vocab / tp`), the masking of foreign ids,
the all-reduce — and the method only answers `embedding(local_ids)`: distinct
rows are read once from this rank's block file through the O_DIRECT reader in
`dsv41_engram_io` (hence `requires dsv41_engram`; 160-byte rows straddle
512-byte sectors and `min_read_bytes(160)` = 1024 covers that). The `weight`
parameter is a zero-row placeholder; `load_weights` skips the 128 checkpoint
shards. Per rank at TP=4 that is 11.9 GiB off the device.

Per token the layer asks for 16 rows (`ngram_size` 3 × `heads_per_ngram` 8);
a rank owns a quarter of them and the masked rest collapse to one read of
row 0. The lookup does host I/O and a D2H of the ids, so it cannot sit inside
a CUDA graph: `--enforce-eager`, or split the piecewise graph at the PLE op —
still to be wired; measured first.

```
python3 tools/qwen38_ple_shard.py build  --out /home/choiceoh/models/qwen38-ple-ssd   # 4 x 11.92 GiB, 66 s on srv4
python3 tools/qwen38_ple_shard.py verify --out /home/choiceoh/models/qwen38-ple-ssd   # 4,352 rows vs source, 0 problems
python3 probes/qwen38_ple_ssd.py                                                       # synthetic, byte-exact per rank
python3 probes/qwen38_ple_ssd.py --out /home/choiceoh/models/qwen38-ple-ssd --repo /home/choiceoh/models/qwen38-flash-next-nvfp4
```
