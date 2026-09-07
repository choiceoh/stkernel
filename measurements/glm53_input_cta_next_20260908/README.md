# GLM-5.3 Flash further decode probes, 2026-09-08

The baseline is the enabled CTA=2 default merged in PR #454, with input
reuse enabled. These private generated kernels do not change the profile
or production source. Kernel timings include invocation-owned input
preparation. They are not serving step/s or output tok/s measurements.

## First attempt: prefetch and vector reduction

Fleet probe `inputnext0908`, source `bebfa9e`, image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Generated CUDA SHA256:
`de3bb8a21564eebc1afd169d98302e291c5c116f7cac8b5a356ac466833e874e`.

| Mode | Warm median us | Read-evicted median us |
| --- | ---: | ---: |
| Current default CTA2 | 24.496 | 66.752 |
| Early input | 24.640 | 66.208 |
| Vector reduction | 24.576 | 66.208 |
| Both | 26.320 | 67.136 |
| Both, three weight buffers | 27.472 | 66.960 |
| Both, four weight buffers | 28.448 | 67.168 |

No candidate improved the warm median. The small read-evicted differences
do not establish a useful improvement. Baseline and candidates alternated
order across 32 samples per cache condition; raw values are in
`initial/result.json`.

The main kernel probe completed 240 exact/oracle numerical rows, 200
changing-input retained-graph checks, and all six startup modes. **The
runner did not complete**: the operator stopped its own test container
when available host memory dropped to 7,635,692 KiB. The public head was
still running and `/health` returned HTTP 200 after that stop. The runner
returned 137; complete sanitizer and continuous service-safety evidence
are absent. This attempt is diagnostic evidence only.

The initial runner incorrectly used an 8 GiB admission threshold without
a continuous memory monitor. Existing fleet probe practice required
16 GiB before admission and at least 12 GiB throughout. The guarded
replacement now enforces both limits, monitors identity and traffic,
stops only its own container on a violation, and records failed attempts.
Do not reuse the original `bebfa9e` runner.

## Three-slice structural follow-up

Source `b29148a` adds a private CTA-local reduction for foreground
M6/N4096 or N6144/K4096 with the existing three-slice plan. It keeps the
original K-group partitions `[0,10)`, `[10,21)`, `[21,32)` and their
summation order. Four variants compare one/two output tiles per CTA and
two/three weight buffers. Unsupported shapes, background work, and the
existing N6416 path retain their current route.

Fleet `inputcta30908` completed all numerical and sanitizer gates:
200 numerical rows, 160 changing-input retained-graph checks, racecheck
zero hazards, memcheck zero errors. The minimum observed available memory
was 90.34 GiB and the continuous guard reported no issues. Approved main
was restored at 04:12:32 KST; runner exit was zero. Raw artifacts and the
restore receipt are under `three_slice/`.

Generated CUDA SHA256:
`e2464cd5ec85a51f2304f5764b9fd8d5ba2d62036bb28e924a9a21543b73ef9e`.
The selected prototype is mode 2 (six warps, two tiles, two buffers),
with 72 registers, zero local spill bytes, four blocks/SM, and 17,152
bytes of shared memory per CTA.

| M6 shape (N,K) | Cache | CTA2 baseline us | Selected us | Reduction | Faster pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096,4096 | warm | 24.320 | 24.320 | 0.00% | 9/32 |
| 4096,4096 | read-evicted | 49.840 | 45.808 | 8.09% | 31/32 |
| 6144,4096 | warm | 32.512 | 24.352 | 25.10% | 31/32 |
| 6144,4096 | read-evicted | 69.632 | 66.096 | 5.08% | 32/32 |
| 6416,4096 control | warm | 24.640 | 24.592 | 0.19% | 14/32 |
| 6416,4096 control | read-evicted | 70.112 | 70.240 | -0.18% | 14/32 |

The N6416 path does not change in this experiment; its small differences
illustrate the measurement noise. There is no measured warm improvement
for N4096. The N6144 improvement is 8.16 microseconds per invocation,
not a 25% improvement in whole-model decoding.

## Production integration and pending serving bracket

`b7b7029` integrates the selected kernel behind `VLLM_GLM53_MK_INPUT_CTA=4`.
The default remains `2`. Mode 4 keeps the existing N6416 CTA2 kernel;
its extra three-slice routes require foreground M6, K4096, N4096 or
N6144, no low-rank correction, enabled input reuse, and the original
three-slice plan. A failed new startup check retains independently
validated CTA2. Unsupported shapes and split overrides fall back.

Integrated source SHA256:
`24b4e23d27dc49a742ef4531e4032f99f8a25400d93187d76d949e97817d8b62`.
CPU validation: 6,689 logic checks, 30 megakernel regressions, 107 fleet
regressions, plus 37 focused tests and 16 subtests. Only the two reviewed
kernel/occupancy counts changed in the logic gate; its CPU dependency
audit was refreshed after confirming the extracted contracts were unchanged.

Fleet `inputcta3prod0908` is queued to verify the integrated source,
forced-split fallbacks, sanitizers, all-rank startup, and a CTA2 → CTA4 →
CTA2 serving bracket. It requests three fixed 2K/2,048-token decode
samples per arm, step windows, and the existing 2K/32K/128K quality gates.
Do not treat the prototype's timings as integrated-source or serving proof.
