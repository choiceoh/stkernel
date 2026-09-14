# C1 FC2 SF6 restoration in four byte lanes

Implementation `e88d8621`, base `b02bff42` (including merged PRs #914–916).
The C1 M1–8 FC2 scale stages now use the exact four-byte reconstruction
already used by separate FC1 staging. FC2 retains its in-place buffers,
volatile packed reads, read-before-write barrier, publication barrier and
pipeline slot lifetime. Only integer reconstruction changes.

Each reconstructed code is at most 63. Adding the base's low seven bits
cannot carry to a neighboring byte (maximum 190); XOR then applies the
base's high bit, preserving modulo-256 scale bytes. All packed reads finish
before the first in-place write. All owners publish their expanded bytes
before the MMA scale fragments can be read.

`sf6_fc2_word_expand` is ON by default only for M1–8 SF6 decode reform. It
is independent of FC1's separate-staging and word-expansion controls.
`sf6_fc2_word_expand=False` keeps the scalar FC2 reconstruction, with distinct
in-memory/disk cache identities. Enabled handles include `fc2word` in their
lane-selection log. No public serving knob was added. Wider rows, legacy q,
FP32 KDA state, K=7 and existing decode fastpaths retain their behavior.

## Native evidence, not a throughput forecast

`native-compile.json` records seven actual CuTe/PTXAS/TVM-FFI compilations:
M1/M7/M8 enabled, M8 scalar-FC2 control, M8 with original FC1 in-place
staging, and the unchanged M16/M32 paths. The primary M8 arms differ only
in `sf6_fc2_word_expand` in the same current source.

| M8 handle | Static instructions | Excluding NOP | Registers | Stack/local bytes |
|---|---:|---:|---:|---:|
| Scalar FC2 control | 3763 | 3708 | 126 | 0/0 |
| Four-byte FC2 | 3595 | 3540 | 126 | 0/0 |

The 168-instruction reduction affects only integer reconstruction opcodes.
Floating-point, MMA, shared load/store, NOP and barrier opcode counts are
unchanged; both arms contain 32 `BAR.SYNC.DEFER_BLOCKING` instructions.
C1 staged allocation remains 100352 bytes and the native binary separately
reports 1024 static shared bytes. M1/M7/M8 retain 126 registers and no stack
or local usage. M16/M32 retain 117 registers and their prior instruction
counts. Static instruction reduction is not an elapsed-time improvement.

## Validation and reproduction

- `cpu-tests.log`: 14 focused tests pass. The new in-place test executes the
  production helper under randomized 128-thread load/store schedules for
  all 256 bases and 64 codes, comparing scalar/word reconstruction to an
  independent encoder. The scheduler requires all threads at both barriers
  and asserts no stores occur before the first barrier.
- A second test reuses 1/2/3 pipeline slots with changed packed inputs,
  checking exact output, untouched neighbors and whole-buffer canaries.
  Existing FC1, activation-store, dispatch/control/cache tests also pass.
- `native-compile.json`: all seven complete kernel handles pass native
  compilation/layout checks. Source, binary and SASS hashes and opcode
  histograms are retained.
- `helper-compile.json`: four production-helper graph fixtures compile:
  in-place scalar, in-place word, separate scalar and separate word. GPU
  mode is prepared for 64 changed-input graph replays per arm with exact
  bytes/canaries, but was not executed.
- The PR's engine and onepass CPU CI gives the full-suite result. Real-weight
  GPU numerics, serving quality/acceptance and consumer speed remain unmeasured.

```sh
python3 -m unittest -v tests.test_engine_moe_sf6_staging tests.test_engine_moe_activation_store tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --fc2-words --sass --output /out/native-compile.json
python3 probes/engine_moe_sf6_check.py --cpu --output /out/helper-compile.json
```

Checks used existing image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`
on srv2, runc, CUDA hidden, no network, two CPUs/four GiB, one build worker
and explicit Python/shell entrypoints. The existing host CUDA 13.0
nvdisasm was mounted read-only for SASS. No baseline engine/image build,
model boot, service restart, GPU context or GPU queue submission was issued.

## Upgraded Oracle

PR #875's Oracle at `e2bfbb9afdcc6e8fe1e5fe47ddddd23180b278b3` compares
base `b02bff42` to implementation `e88d8621` at C1 2K/32K/128K using the
retained checkpoint configuration. `--acc 0` is a timing-only assumption,
not an acceptance estimate. With no paired coefficient for the changed
MoE component, total decode delta is null. Instruction counts are not used
as a timing coefficient. `oracle-c1.json` and the empty paired-profile
template retain the comparison identities.
