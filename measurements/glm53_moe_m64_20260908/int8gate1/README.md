# INT8 / M64 full gate, 2026-09-08

The gate failed in memcheck and did not admit serving. All 20 TP4 numerical cases passed: ten BF16 and ten FP8-gather/INT8-reduction cases, including every stock control, changed input/route, repeated use, retained output, local MoE, and 8192-token M128 graph replay. The per-row thresholds remain unchanged.

Memcheck completed the six M64 numerical cases and all forty INT8 packet/output/padding/lifetime cases, but reported **34 CUDA_ERROR_INVALID_VALUE errors on cuGetProcAddress_v2**. Its error-exitcode 99 correctly stopped the runner. All 34 reported errors have that same API lookup type; no device-access error was printed. This does not establish a clean sanitizer result. Racecheck and direct TTFT were not reached.

The backtraces point to the CUDA Python binding bootstrap reached by `HardwareInfo.__init__` during stock MoE setup. Read-only inspection of the pinned image found cuda-bindings/cuda-python 13.3.1, CUTLASS DSL 4.6.2, PyTorch 2.13.0+cu130, host driver 580.159.03 and Compute Sanitizer 2025.3.1.0. A driver/binding compatibility issue is a hypothesis pending the separate minimal torch-only/driver-only/torch-driver reproduction. No error-reporting option, numerical threshold or source guard was weakened.

NVIDIA documents that Compute Sanitizer reports CUDA API failures separately from device-memory errors, and that applications must handle their return status. The current host defaults to explicit API reporting. The available `--ignore-getprocaddress-notfound` switch concerns error 500, whereas this run reports error 1, so it was not used. See the [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

Source `183df3a2065c5c7bab1729c66282e633cbbe3fa0` was frozen on all four nodes. Normal session `moem64int8gate10908` received GO at 05:14:11 KST; the probe ran 05:15:22–05:19:02. Exact incoming recovery completed before the worker exited 1 at **05:21:40 KST**. Container IDs, image, configuration hashes, mounts, overlay/manifest hashes and public port match the incoming snapshot on all four nodes. Both experimental flags remain default-off.

`summary.json` preserves the failure and component timings without treating them as accepted performance. Raw CPU, eight rank, memcheck and lifecycle logs are losslessly gzip-compressed. The prepared serving collector requires this gate to pass and therefore continues to reject it.
