# EP tiled K3 shape CPU evidence

Normal CPU fleet session: `eptiledcpu0910k36`, worker `10.10.10.4` (srv4).
Original output: `/home/choiceoh/glm53-ep-tiled-k3-6-cpu-evidence`. Collection used only read-only archive operations, with no tests or GPU work.

The original receipt reports **119 tests, zero failures/errors/skips**, **seven static and two dynamic CuTe lowerings**, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process. Original no-device, no-network, 4 GiB/2 CPU and 12 GiB host-availability constraints remain.

| Variant | A-ring / word unpack | BF16 scatter | Output dtype | Registers | Stack bytes | Local bytes |
|---|---:|---:|---|---:|---:|---:|
| Static M4, M6, M8 / SF6 | true | true | BF16 | 123 | 0 | 0 |
| Static M12, M16, M24, M32 / SF6 | false | false | FP32 | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | unchanged | FP32 | 168 | 112 | 0 |

Actual constructor flags and fake output element types were checked before and after lowering. M4/M6/M8 have 19-field keys with `bf16_scatter` at index 15 and the ordered A-ring, word-unpack, BF16-scatter suffixes. Other static keys retain 16 fields and FP32 output. New K3 request batches use native rows M4/M8/M12/M16; this CPU receipt proves compilation of those shapes, not that a serving process selected K3. The source adds actual-weight canary M4/M8/M16 after all nine prior cases, preserving their input seeds. GPU execution of all twelve cases and live speculation configuration/counters remain separate gates.

The real CPU process read imported `flashinfer.cute_dsl.fp4_common` and checked its complete SHA, size and path against the pinned oracle; the source-bound probe also checked imported helper object identity. Receipt: SHA `a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1`, 87,909 bytes. `helper-proof.json` preserves this scope. Capsule runtime identity exactly matches the prior CPU5 actual receipt and the fixed manifest; the worker image, manifest and clean source were checked before and after copying.

Exact compiled source: `6977199cf699f82696f925f77bc930d500135532`. All **22 mounted sources and 26 contracts** match immutable committed bytes. No source rebind was needed. Original result SHA: `93e28079eff1ec4493633ef253c00442cf3f960f0060731ed348b7eadb547348`.

All **29 original worker files**, before/after inventories, exact execution metadata and CPU stdout are preserved. `execution.json` came from the process executing the normal fleet command and records rc0; elapsed time was 56.514 seconds. `verification.json` binds that execution and stdout. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON remains unchanged. Resource numbers describe compiled artifacts, not runtime timing or occupancy. SHARED1024 is static resource reporting, not total dynamic shared allocation; LOCAL0 alone is not a runtime spill-performance claim.

This archive proves CPU compilation and contracts only. GPU numerical correctness, graph behavior, throughput, quality, the 67 tok/s target and default adoption are **not accepted by this evidence**.
