# V4.1 packed compressed KV

The official reference already applies E2M1/E4M3 fake quantization to compressed
KV, then stores BF16 values. This change stores the original packed quantizer
outputs and decodes only the selected positions inside direct sparse attention.
It retains the full-prefix copy removal from #524 and can run with #522's
packed indexer. The compressed-KV adapter and BF16 dual-pool adapter are
mutually exclusive because they own the same Attention methods and caches.

This remains an explicit reference implementation. Actual TileLang packing,
GPU numerics, NCCL/model quality and tok/s, step/s and TTFT are unmeasured.
V4.1 serving integration is incomplete and defaults are unchanged.

## Static storage arithmetic

Pinned `deepseek-ai/DeepSeek-V4.1-Flash` source revision:
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
The source hashes in `storage-budget.json` match the #522/#524 reference.

A compressed D512 row becomes 256 packed bytes plus 32 E4M3 scale bytes,
instead of 1,024 BF16 bytes: 71.875% fewer allocated bytes. Owners 2/8/14/20
use ratios 2/2/2/1, so at batch one and configured capacity 1M their total
is 2.5 GiB versus 720 MiB. Batch four is 10 GiB versus 2,880 MiB. This is
cache capacity, not a promise that a 1M prompt can currently be served.

The selected compressed operands for one decode query are at most 19 MiB
versus 5.34375 MiB across 38 layers with 512 valid selections each, counting
each selected row once. The window and other tensors remain separate; head
tiling, invalid slots and device caches affect physical reads. Static byte
counts do not establish allocator-reserved memory, bandwidth or speed.

```sh
python measurements/dsv41_packed_kv_20260910/storage_budget.py \
  --reference-dir /path/to/pinned/inference --output /tmp/fresh-budget.json
```

## Reproduction

```sh
python -m unittest discover -s tests -p 'test_dsv41_*.py' -v
python probes/dsv41_packed_kv_diff.py \
  --reference-dir /path/to/pinned/inference --output /tmp/fresh-packed-kv.json
bash launchers/compose-overlays.sh dsv41
python tests/test_logic.py --component core
```

The CPU oracle must independently compare all E2M1/E4M3 byte combinations,
packed-byte order, the pinned producer wrapper and staged sparse arithmetic.
Its integration fixtures use synthetic projections and explicit CPU replacements
for GPU kernels; those replacements do not validate model weights or actual
TileLang code generation. CPU NaN comparison covers positions, not payloads.

The normal-fleet `cpu_compile_runner.py` runs from a frozen clean source with
an already-present pinned image. It caps CPU/memory, binds neither GPU devices
nor a model, disables network and requires CUDA to stay uninitialized. The
source is hashed before/after and all fetched compiler artifacts are verified.
Only a subsequent actual device/consumer campaign can establish speed or
admission for a serving default.

## CPU results: PASS

- `cpu-unit2.log`: 76 tests, zero skips, including 20 new packed-KV tests.
- `cpu-oracle2.json`: all 4,096 nibble/scale combinations and 256 packed-byte
  patterns; signed zeros and NaN locations; 22 pinned original-wrapper producer
  boundary rows using explicit independent CPU arithmetic. NaN payload identity
  is not claimed.
- All 32,640 nonnegative finite BF16 maxima were examined for the scale rule.
  The 17,711 nonoverflow casts agree with the independent E4M3 RNE oracle.
  Torch 2.14.0 on this macOS CPU saturates the other 14,929 casts, so that
  observed behavior is not assumed to specify TileLang. Both explicit overflow
  policies (NaN and saturation) pass packed/in-place CPU wrapper comparisons;
  actual GPU producer bytes remain unverified.
- Six staged sparse-attention fixtures pass under each of FP32 and BF16 global
  defaults. Eight complete 38-layer Attention sequences cover batch one with
  129-token prefill/ring wrap/odd-even decode, batch two, coexistence with the
  packed indexer and fresh prefill after restoration. Projections/compressors
  are synthetic; FP8 window quantization is an identity fixture, and TP4 sums
  repeat the local rank on CPU rather than executing NCCL.
- Weak references verify eight displaced BF16 owners are released across two
  fixtures and all 16 packed planes are released after restore. This verifies
  tensor lifetime, not GPU allocator-reserved memory.
- `cpu-core.log`: 71,769 core checks and 74 megakernel regressions pass.
  `compose.log`: 25 overlays from seven modules, including 13 model files.

## Initial SM121 compile admission: rejected

Normal CPU fleet session `dsv41packedkvcpu0910v1` used frozen source
`82385f1ff7568be6a46be0b07145464cc61ac0f5`. Code generation produced H8, but
its 147,456-byte shared allocation exceeded the unchanged 98,304-byte budget,
so the runner stopped before compiling the remaining variants. No kernel ran.
The failed receipt, original metadata and manifest are retained in `aot-cpu1`.
All 21 source hashes and all 11 fetched files were verified.

TTGIR shows two simultaneous 64x512 BF16 KV allocations, with different QK/PV
swizzles, plus a 16x512 BF16 Q allocation: 64+64+16 KiB. Triton 3.7.1 follows
the pure conversion chain back to uint8 transport when choosing the dot's
original operand width. The correction explicitly marks the final BF16 value
with a bit-preserving `mov.b16` boundary whose side-effect annotation stops
that compiler traversal. This instruction is neither arithmetic nor a memory
fence. Tile order, compact loads, eight warps and the admission limit remain.
See the compiler's [`computeOrigBitWidth` implementation](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp).

## Final SM121 compile admission: PASS

Normal CPU fleet session `dsv41packedkvcpu0910v2` compiled frozen source
`2b2299df3fd86d6a9bc721827e1bc2e572315151` on srv3. The terminal
`aot-cpu2/fleet-ack.json` reports `finished-cpu`, return code 0. All four
H8/H16/H32/H64 variants use 83,968 bytes (82 KiB) of static shared memory,
below the same 96 KiB budget. The kernel still uses eight warps, one stage
and disabled FP fusion; only the BF16 identity boundary changed device code.

The already-present image was
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
with Torch `2.13.0+cu130` and Triton `3.7.1`. CUDA stayed uninitialized and
the container had no GPU devices or model. The 10.09-second compiler run is
not a kernel timing. AOT register/spill counts were unavailable without device
initialization and are not inferred from PTX virtual registers.

All 21 source-file hashes match before/after compilation and the local final
implementation. All 44 manifest files, including PTX, cubins and compiler IR,
were fetched and verified. Full artifacts remain on srv3 at
`/home/choiceoh/dsv41-packed-kv-cpu2-evidence`. The committed receipt subset
is textual; `SHA256SUMS` covers the full remote directory.

`aot-cpu2/ptx-audit.json` verifies all four variants use `kWidth=2`, one shared
KV allocation, compact payload/scale IR loads, the bit-preserving BF16 boundary,
original staged QK/PV accumulators and output-only global stores. Logical IR
shapes and static PTX instructions are not physical memory-transaction counts.

`cpu-unit2.log` and `cpu-oracle2.json` cover the final source; the earlier
unsuffixed files preserve the initial CPU evidence. The source still requires
actual GPU producer, output and consumer-speed validation before deployment.
