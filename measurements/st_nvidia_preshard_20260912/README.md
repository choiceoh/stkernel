# NVIDIA GLM-5.3-Flash NVFP4: lossless offline TP=4 preshard

Source: `nvidia/GLM-5.3-Flash-NVFP4`, pinned revision
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3`. The original download on srv4
passed Hugging Face verification for all 44 files. Its 33 safetensors files
occupy 204,439,103,396 bytes. A separate copy was transferred to srv3 over
SSH/rsync because srv4 could not hold the input and all output ranks together.

## Current status

Paused after writing layer 27 so the separately coordinated ST fleet recovery
and adoption can finish first. The process is preserved as
`paused-offline-nvidia-preshard-9391` on srv3, with container ID
`d528ddea7ab75b50916d60caac78e8f0876f4da1e24c4586330573651746b672`.
The recovery task owns restoring its original name and unpausing it.

The four files remain in
`/home/choiceoh/models/st-glm53-nvidia-tp4-9391.incomplete/`. They are **not
finished checkpoints**. Layers 28–44, full readback, checksums, vision output
and atomic publication remain pending. Do not recreate the paused process or
rename the partial directory into a completed model.

## Representation

The NVIDIA checkpoint uses ModelOpt NVFP4. Its `weight_scale_2` is an FP32
multiplier. The existing Red Hat importer reads a compressed-tensors divisor
and folds it into E4M3 block scales. Reusing that transformation would change
NVIDIA's original scale values. In addition, NVIDIA keeps the nine dense MLP
projections of layers 0–2 in packed NVFP4.

`engine/profiles/glm53/modelopt_weights.py` defines a separate offline layout,
`st-glm53-modelopt-up-gate-v1`:

- Packed E2M1 bytes retain their values and nibble order; TP slices the
  intermediate dimension into four pieces.
- FC1 rows are merged as `up|gate`, matching the b12x weight order.
- Raw E4M3 scales are sliced and interleaved, without folding or rounding.
- `w13_alpha` and `w2_alpha` retain FP32 weight multipliers separately.
  `a13_scale` and `a2_scale` retain the calibrated input scales.
- The importer requires aligned gate/up global scales for fused FC1 and fails
  explicitly on disagreement.
- Dense layers 0–2 retain packed weights/scales, using the same one-expert
  tensor form; they are not converted to stored BF16 matrices.
- Other tensors use the existing GLM placement rules, including FP32 kernel
  parameters. The BF16 vision tower is written separately.

**This is a prepared offline checkpoint, not a live-serving promotion.** The
current serving loader intentionally rejects this different layout. A serving
adapter still needs to bind the separate multipliers/input scales and route the
packed dense layers through the corresponding NVFP4 operations. The original
checkpoint and current production files/configuration are untouched.

## Conversion and verification

`engine/profiles/glm53/preshard_modelopt.py` checks the exact source encoding,
all required source keys, quantized-projection coverage and available output
space. It streams one layer at a time into four rank writers. Every output
tensor is checked for dtype, shape and finite values where applicable, hashed,
then read back through `RankLoader` and compared byte-for-byte with the tensor
that was written. Rank identity and encoding are recorded in each file.

Files remain under an `.incomplete` directory until every rank has passed
readback and SHA-256 generation. The completed output includes a manifest,
`SHA256SUMS`, original tokenizer/config metadata and a separate vision file.
The completed directory is published by a same-filesystem rename.

The source plan covers 146,425 source tensors and **36,297 quantized
projections**, including the nine dense projections. Each rank has 1,306
logical tensors and 47,623,689,640 payload bytes (about 44.35 GiB), before
safetensors headers/alignment. Four placement/scale/roundtrip tests passed
on both inspected ARM runtimes. Model architecture facts matched exactly
between the Red Hat and NVIDIA configs, and the existing Red Hat encoding
validation continued to pass. The loader suite also passed all 11 tests with
the real Red Hat checkpoint mounted, with no skips in that run.

```sh
python3 engine/profiles/glm53/preshard_modelopt.py \
  --ckpt /source \
  --out /models/st-glm53-nvidia-tp4-9391 \
  --source-revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3
```

The conversion runs on srv3 in `st-engine:9391`, limited to 16 GiB of container
memory and four CPU cores. It does not need a GPU.

## Portable Deneb inputs saved separately

Private input datasets were saved on srv4 at
`/home/choiceoh/datasets/deneb/2026-09-12/`, with a portable archive at
`/home/choiceoh/datasets/deneb/deneb-inputs-2026-09-12.tar.gz`.

The conversation collection contains 237 inputs (160/40/37); the workload
collection contains 292 inputs (200/48/44). Both provide UTF-8 train/validation/
test JSONL, `messages`, category and grouping fields, a manifest, private source
provenance and Korean usage documentation. Some conversation examples overlap
between collections, so they must not be counted as independent data.
These are input-only calibration datasets, not verified SFT target answers.
The original longer text is retained independently of ST's 512-token capture.
All file and archive checksums were verified. Private input text, identifiers
and provenance are not included in this repository.
