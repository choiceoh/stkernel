# EP tiled native BF16 scatter CPU evidence

Normal CPU fleet session: `eptiledcpu0910bf165`, worker `10.10.10.4` (srv4).
Original output: `/home/choiceoh/glm53-ep-tiled-bf16-5-cpu-evidence`. Collection performed only read-only archive operations, no tests or GPU work.

The original receipt reports **109 tests, zero failures/errors/skips**, four static and two dynamic CuTe lowerings, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process. The original no-device, no-network, 4 GiB/2 CPU and 12 GiB host-availability constraints remain.

| Variant | A-ring / word unpack | BF16 scatter | Output dtype | Registers | Stack bytes | Local bytes |
|---|---:|---:|---|---:|---:|---:|
| Static M6 / SF6 | true | true | BF16 | 123 | 0 | 0 |
| Static M12, M24, M32 / SF6 | false | false | FP32 | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | unchanged | FP32 | 168 | 112 | 0 |

Actual constructor flags and fake output element types were checked before and after lowering. M6's 19-field key has `bf16_scatter` at index 15 and the ordered A-ring, word-unpack, BF16-scatter suffixes. Other static keys retain 16 fields and FP32 output. The kernel preserves stock weighted BF16 contributions but changes their accumulation from FP32 to the stock BF16 vector RED; this is not bitwise-equivalent summation. Startup numerics and measured performance remain required.

The real CPU process read the imported `flashinfer.cute_dsl.fp4_common` file and checked its complete SHA, size and path against the pinned oracle; it also asserted that the EP module imported the exact same helper object. Receipt: SHA `a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1`, 87,909 bytes. `helper-proof.json` preserves that scope. It does not claim that a host file or the test oracle was the running module.

Exact compiled source: `e88f5fd368ef8c895000496101fc7fcf8e7fb344`. All **22 mounted sources and 23 contracts** match immutable committed bytes. No source rebind was needed. Original result SHA: `8c39170913365b171f39f257ab10f4bbe0192900767c3de373d6f9f31bddbff9`.

All **20 original worker files**, before/after inventories, exact execution metadata and CPU stdout are preserved. `execution.json` was written by the process executing the normal fleet command, with rc0 and start/end times; elapsed time was 45.590 seconds. Its stdout hash is bound by `verification.json`. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON remains unchanged. Resource numbers describe compiled artifacts, not runtime timing or occupancy.

This archive proves CPU compilation and contracts only. GPU numerical correctness, graph behavior, throughput, quality and default-adoption acceptance remain separate; none is claimed here.
