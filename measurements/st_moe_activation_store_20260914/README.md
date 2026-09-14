# C1 packed intermediate activation stores

Implementation `80674dd9`, base `1c9ec633` (merged PR #915). The M1–8
decode reform now stores each quantizer result as one aligned 64-bit shared
write instead of unpacking it into eight byte writes. It preserves all eight
payload bytes, the FC2 consumer mapping, scale writes, BF16 rounding and
FP4 quantization. Publication barriers and pipeline ordering are unchanged.

The producer converts FP4 nibble offsets to byte offsets before applying
the shared swizzle. Each aligned eight-byte block stays contiguous because
S<2,4,3> changes only bits 4 and 5. The native setup checks every output byte
against the actual FC2 layout and verifies block alignment/contiguity.
The default is ON for M1–8 decode reform, including its unpacked-scale
recipe; wider rows retain their prior store path. The internal
`packed_activation_store=False` control has distinct memory/disk cache keys.
Enabled handles carry `a2u64` in the lane-selection log. No public serving
knob was added; K=7, FP32 KDA state and existing fastpaths remain enabled.

## Native compilation, not serving speed

`native-compile.json` records actual CuTe/PTXAS/TVM-FFI compilation of six
handles: M1/M7/M8 enabled, M8 byte-store control, and unchanged M16/M32.
Both M8 arms use the same current source and differ only in the store flag.

| M8 handle | Static instructions | Excluding NOP | Registers | Stack/local bytes |
|---|---:|---:|---:|---:|
| Byte-store control | 3795 | 3741 | 126 | 0/0 |
| Packed store | 3763 | 3708 | 126 | 0/0 |

The quantization loop loses eight `STS.U8` and gains one `STS.64`; its
separate scale-byte store remains. Changes in other opcode counts are
integer address/payload operations and one NOP. Floating-point, MMA and
barrier opcode counts are unchanged. C1 staged allocation remains 100352
bytes, with the binary separately reporting 1024 static shared bytes.
The 33 fewer non-NOP static instructions are **not a throughput forecast**.

## Verification and reproduction

- `cpu-tests.log`: 12 focused tests pass. New tests execute the production
  helper against the independent FC2 consumer mapping, cover all 64 payload
  bits in all 128 blocks, randomized store order, active row counts 0–16,
  changed tile reuse, untouched inactive rows and whole-buffer canaries.
  Default/control dispatch and repeated cache normalization are covered.
- `native-compile.json`: all six complete kernels pass native layout and
  resource checks. Source, binary and SASS hashes/opcode counts are retained.
- `helper-compile.json`: both production-helper fixtures compile with the
  real shared layout. GPU mode is prepared for 64 changed-input graph
  replays per arm with partial-row reuse and exact bytes/canaries. It was
  not executed.
- The PR's complete engine and onepass CPU checks provide the full-suite
  result. Full real-weight GPU numerics, quality, acceptance and consumer
  step/s remain unmeasured.

```sh
python3 -m unittest -v tests.test_engine_moe_activation_store tests.test_engine_moe_sf6_staging tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --activation-store --sass --output /out/native-compile.json
python3 probes/engine_moe_activation_check.py --cpu --output /out/helper-compile.json
```

Compilation used existing image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`
on srv2, runc, CUDA hidden, no network, two CPUs/four GiB, one build worker,
and explicit Python/shell entrypoints. For SASS, the existing host CUDA 13.0
nvdisasm was mounted read-only. No image/baseline engine build, model boot,
service restart, GPU context or GPU queue submission was issued.

## Upgraded Oracle

PR #875's Oracle at `e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3` compares
base `1c9ec633` to implementation `80674dd9` at C1 2K/32K/128K with the
retained checkpoint configuration. `--acc 0` is a timing-only assumption,
not an acceptance estimate. The MoE edit has no paired timing coefficient,
so the total decode delta remains null. No coefficient is invented from
instruction counts. `oracle-c1.json` and its empty paired-profile template
retain the comparison identities.
