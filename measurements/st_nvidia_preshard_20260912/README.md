# NVIDIA GLM-5.3-Flash NVFP4: lossless offline TP=4 preshard

Source: `nvidia/GLM-5.3-Flash-NVFP4`, pinned revision
`09b04e5e74bca08ca8549fc736d4cdd8624bfde3`. The original download on srv4
passed Hugging Face verification for all 44 files. Its 33 safetensors files
occupy 204,439,103,396 bytes. A separate copy was transferred to srv3 over
SSH/rsync because srv4 could not hold the input and all output ranks together.

## Current status

**Completed on srv3 at 2026-09-12 13:09 KST; final audit passed at 13:09:44.**
The preserved process resumed under its existing name,
`paused-offline-nvidia-preshard-9391`, and exited successfully with code 0.
It is no longer running or paused. No fleet service or model configuration was changed.

Completed output: `/home/choiceoh/models/st-glm53-nvidia-tp4-9391/`.
The `.incomplete` directory no longer exists. The output contains 16 files,
191,643,255,409 bytes (191.64 GB), including the independent verification receipt.

| Files | Count | Bytes each | Verification |
|---|---:|---:|---|
| `rank0of4.safetensors` through `rank3of4.safetensors` | 4 | 47,623,920,136 | 1,306 tensors per rank read back exactly; file SHA-256 |
| `vision.safetensors` | 1 | 1,127,303,288 | 347 tensors compared byte-for-byte with source; file SHA-256 |
| Original metadata | 8 | — | Every output hash matches its original source file |
| Conversion manifest, checksums, completion receipt | 3 | — | Saved alongside the weights |

See [final verification](final-verification.json), [process/output state](final-state.json),
[conversion manifest](preshard-manifest.json), [file checksums](SHA256SUMS), and
[completed log](build-completed.log). The converter verified 5,224 logical rank
tensors in total. A separate audit opened all four ranks with the standard
`safetensors` reader, checked tensor names/shapes, checked vision against source,
and verified metadata and the checksum list. It did not recompute the large rank
file hashes a second time; those are from the converter's completed readback.

The converter's elapsed time includes the earlier pause for ST recovery. Its
original manifest pins the frozen conversion script. [Complete source accounting](complete-plan.json)
and [completion code provenance](completion-provenance.json) record the added
MTP exclusion audit without rewriting any rank weights. The receipt is also
stored in the output as `completion-verification.json`.

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

The checkpoint's native MTP layer 45 is intentionally excluded: ST uses
**DFlash2** as its drafter. The original NVIDIA source retains all 889 MTP
tensors (14,865,185,408 bytes); no duplicate MTP file is needed in this target.
The full [source accounting](complete-plan.json) classifies all 147,661 input
tensors as 146,425 text tensors, 347 vision tensors, and those 889 excluded MTP
tensors, with **zero unaccounted tensors**. Future full conversions fail if
anything outside that explicit MTP policy is omitted. DFlash2's own weights
remain a separate existing checkpoint.

**The conversion receipt is not a live-serving promotion.** At conversion
completion the serving loader rejected this layout. The subsequent
[serving adapter](../st_nvidia_serving_20260912/README.md) binds the separate
multipliers/input scales and routes packed dense layers through NVFP4. Its
validation is recorded separately; the historical receipts below are unchanged.
The original checkpoint and current production files/configuration are untouched.

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
on both inspected ARM runtimes. On completion, all nine importer/source-accounting
[tests passed on srv3](completion-tests.log), including explicit MTP exclusion,
unknown tensor rejection, absent source keys, and truncated source payloads. Model architecture facts matched exactly
between the Red Hat and NVIDIA configs, and the existing Red Hat encoding
validation continued to pass. The loader suite also passed all 11 tests with
the real Red Hat checkpoint mounted, with no skips in that run.

```sh
python3 engine/profiles/glm53/preshard_modelopt.py \
  --ckpt /source \
  --out /models/st-glm53-nvidia-tp4-9391 \
  --source-revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3
```

The conversion ran on srv3 in `st-engine:9391`, limited to 16 GiB of container
memory and four CPU cores. The final audit used at most 3 GiB and two CPU cores.
Neither operation used a GPU. [verify_completed.py](verify_completed.py) can
repeat the independent audit, writing a new receipt outside the model directory:

```sh
python3 measurements/st_nvidia_preshard_20260912/verify_completed.py \
  --ckpt /source --directory /models/st-glm53-nvidia-tp4-9391 \
  --receipt /new-path/verification.json
```

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
