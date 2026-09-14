# C1 SF6 expansion in four byte lanes

Implementation `fcf71388`, base `0d3d6d59` (merged PR #913). The enabled
M1–8 separate FC1 staging path now reconstructs four scale bytes in one
32-bit word. It uses the existing packed ring, volatile loads, publication
barrier and pipeline ownership. No additional shared memory is allocated.

## Exact integer reconstruction

Two packed low-plane bytes hold four nibbles; one high-plane byte holds
four two-bit groups. Shift/mask operations spread them into four byte lanes.
Each reconstructed code is at most 63. Adding the base's low seven bits
therefore produces at most 190 in each lane, preventing a carry into its
neighbor. XOR applies the base's high bit, preserving the original
modulo-256 addition even when base + code wraps. No floating-point operation,
scale encoding, MMA input or rounding boundary changes.

The compile-time control `sf6_word_expand=False` retains PR #913's scalar
reconstruction with separate staging still enabled. It has distinct cache
identities and is not a new public serving knob. The default C1 handle is
marked `fc1sepword` in the existing lane-selection log. In-place FC2,
legacy q and wider-row restoration retain scalar reconstruction and their
existing barriers. K=7, FP32 KDA state and the enabled decode fastpaths remain.

## Native code evidence, not throughput

`native-compile.json` records seven actual CuTe/PTXAS/TVM-FFI compilations
and source/native-binary/SASS hashes. At M8, the controls differ only in
the word-expansion flag and share the same staging/cache geometry:

| M8 compiled kernel | Static instructions | Excluding NOP | Registers | Stack / local bytes |
|---|---:|---:|---:|---:|
| PR #913 scalar restoration | 3,875 | 3,825 | 126 | 0 / 0 |
| Four-byte restoration | 3,795 | 3,741 | 126 | 0 / 0 |

The 84 non-NOP instruction reduction is about 2.20% of the disassembled
kernel's static instruction count. It is **not** a 2.20% latency or throughput
forecast. Changed opcode counts are integer operations and NOPs; the full
histograms are in the report. C1 staged allocation remains 100352 bytes;
the native binary separately reports 1024 static shared bytes. M1/M7/M8
retain 126 registers and zero stack/local bytes. M16/M32 stay on the scalar
path, with 117 registers and zero stack/local bytes.

## Validation and reproduction

- `cpu-tests.log`: nine focused tests pass. They execute the actual helper
  with randomized 128-thread load/store schedules, all 256 bases and all
  64 codes, scalar/word comparison, ring reuse and whole-buffer canaries.
  A separate 65,536-case check varies each byte lane through every base/code
  against an independent encoder with mixed neighboring values. Cache
  normalization, controls and scatter bounds remain covered.
- `native-compile.json`: word M1/M7/M8, separate-scalar M8, original in-place
  M8, and unchanged M16/M32 compile. Native resources and instruction
  histograms are read from the retained binary without a CUDA context.
- `helper-compile.json`: three graph fixtures compile: in-place scalar,
  separate scalar and separate word. The GPU mode is prepared for exact
  bytes/canaries and 64 changed-input graph replays per arm, but was not run.
- The PR's engine and onepass CPU checks provide the complete-suite verdict.

Checks used existing image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`,
runc, CUDA hidden, no network, two CPUs/four GiB, one build worker and an
explicit Python/shell entrypoint. The image has cuobjdump but no nvdisasm;
for the optional SASS check, the existing host CUDA 13.0 nvdisasm was mounted
read-only at `/usr/local/bin/nvdisasm`. No image build, baseline engine,
model boot, service restart, GPU context or queue submission was issued.

```sh
python3 -m unittest -v tests.test_engine_moe_sf6_staging tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --sass --output /out/native-compile.json
python3 probes/engine_moe_sf6_check.py --cpu --output /out/helper-compile.json
```

`--sass` requires nvdisasm on PATH; native compilation/resource checks still
work without it when that option is omitted. Full real-weight MoE, serving
quality/acceptance and matched consumer speed remain unmeasured.

## Upgraded Oracle

The PR #875 tool at `e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3` compares
base `0d3d6d59` to implementation `fcf71388` with the retained checkpoint
configuration at C1 2K/32K/128K. `--acc 0` is a timing-only scenario, not
an acceptance forecast. The changed MoE code has no paired timing
coefficient, so the total decode delta is null. No timing coefficient is
inferred from the instruction count. `oracle-c1.json` and the empty paired
template preserve the source/configuration identities.
