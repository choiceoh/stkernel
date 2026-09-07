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

The runner uses a fleet maintenance hold, checks memory continuously,
and unconditionally restores approved main. Results and restore proof
will be recorded after completion.
