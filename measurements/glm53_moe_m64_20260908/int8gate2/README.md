# INT8/M64 full gate with detector controls, 2026-09-08

The gate failed a local M64 numerical comparison. It cannot admit serving.
The runtime is unchanged from int8gate1; source `6ee06be889a19adec6dffc21b38ee4b19e58b2c3`
adds driver-first sanitizer initialization and mandatory detector controls.

All three controls worked: deliberately invalid `cuDeviceGet(999999)` produced
exactly one API error and exit 99; deliberately invalid CuTe global writes and
shared-memory races were both detected with exit 99. Source hashes match the
frozen control programs. No error-reporting option was disabled.

The ten BF16 and ten FP8-gather/INT8-reduction TP4 numerical cases all passed,
including independent stock controls, changed inputs, repeated use, retained
outputs, local MoE and changed-input graph replay. These are synthetic component
checks, not full-model quality or speed acceptance. Their timings remain in
`summary.json` and the raw logs without granting a performance verdict.

Memcheck then completed 6144 balanced/concentrated and 6912 balanced cases.
During changed-input candidate reuse at 6912/concentrated, eight rows exceeded
the original numerical criterion; maximum relative L2/peak were
0.013218882493674755/0.13671875. The code had no independent changed-input stock
control in this local phase and stopped at the first assertion, so this record
alone cannot attribute the exceedance to M64 or characterize repeat variation.
The printed maxima cover all rows, and do not identify each failing row's limit.

The executed memcheck portion reported zero API/device errors, but the application
exited 1. The remaining M64 cases, INT8 sanitizer checks, racecheck and direct
TTFT were not reached. This is not a complete clean memory/sanitizer result.
No numerical limit was weakened and no experimental flag was promoted.

Normal fleet session `moem64int8gate20908` received GO at 06:02:00 KST. The probe
ran 06:03:14–06:06:24. All four original container IDs, image/configuration/mount
hashes, source overlays, manifest and public port match the restored snapshot.
The original endpoint recovered health before outer exit 1 at **06:09:02 KST**.
Raw rank, sanitizer, control, CPU and lifecycle files are losslessly preserved.

The separate follow-up fixes a bounded repeated-control plan for the same local
fixture under plain/memcheck/racecheck, with all failing rows and actual BF16
payloads retained. It preserves the runtime and original thresholds. Its
completion marker can never approve numerical or serving acceptance.
