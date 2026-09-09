No-device CUDA binding diagnostics, 2026-09-08.

Three bounded normal fleet CPU sessions epbindings0908v1/v2/v3 ran in the
same immutable image and pinned Compute Sanitizer as GPU attempt4. Each
fresh runc container had no network or CUDA device nodes, 512MiB memory,
one CPU and 64 PIDs. The host libcuda.so.580.159.03 was mounted read-only,
with SHA256 and all commands preserved in result.json. No MoE/CuTe code or
GPU kernel was invoked. These jobs neither paused nor modified serving.

- Diagnostic1: cuDriverGetVersion returned result0, driver API13000 and
  cuda-bindings13.3.1. Compute Sanitizer returned255 because the process
  ended before its first instrumented API call; the 34 errors did not recur.
- Diagnostic2: adding the original hardware-info cuDeviceGetCount call
  exposed an output-format bug: its uninitialized result includes None,
  which the diagnostic incorrectly tried to convert to int. The traceback
  and original script are retained. Sanitizer again returned255.
- Diagnostic3 fixes only that output handling. Driver-version result0 and
  API13000 were confirmed; cuDeviceGetCount returned [3,null] with no CUDA
  device nodes. Sanitizer again returned255 without instrumentation.

This confirms the binding/driver version combination but neither reproduces
nor clears attempt4 sanitizer errors. The no-device setup cannot establish
device API or memory-check correctness. Attempt4 remains failed. Its 34
reported stacks are all initial compact hardware-info cuGetProcAddress_v2
lookups before candidate compilation, with no device-fault headings reported.

A next minimal reproduction needs a normal GPU reservation: a fresh process
first opens a tiny Torch CUDA context, then invokes cuda-bindings
cuDeviceGetCount. This matches the original ordering and can separate
resolver initialization from MoE. No such GPU reproduction has run here;
no API errors were suppressed and no failed run was reclassified as PASS.

manifest.json covers original bytes, including decompressed logs; all raw
reports and scripts are preserved. These are diagnostic records, not CPU
compilation receipts or a replacement for attempt4 GPU proof.
