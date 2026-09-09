# EP tiled SF6 word-unpack CPU evidence

Normal CPU fleet session: `eptiledcpu0909word4`, worker `10.10.10.4` (srv4).
Original output: `/home/choiceoh/glm53-ep-tiled-word4-cpu-evidence`. This collection performed read-only archive operations, no tests or GPU work.

The original CPU receipt reports **107 tests, zero failures/errors/skips**, four static and two dynamic CuTe lowerings, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process. The runner kept the original no-device, no-network, 4 GiB/2 CPU and 12 GiB host-availability constraints.

| Variant | A-ring | Word unpack | Registers | Stack bytes | Local bytes |
|---|---:|---:|---:|---:|---:|
| Static M6 / SF6 | true | true | 123 | 0 | 0 |
| Static M12, M24, M32 / SF6 | false | false | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | unchanged | 168 | 112 | 0 |

The actual M6 constructor flags were checked before and after lowering; its 18-field key ends with `glm53_ep_static_sf6_a_ring_v1`, then `glm53_ep_static_sf6_word_unpack_v1`. Other static keys retain 16 fields. Source `549fd55319c3379438c3cb99694df9f28c01aefb` is unchanged; all **22 mounted sources and 23 contracts** match its immutable committed bytes. No source rebind was needed. Original result SHA: `453d1dd74340e9bcdb1be67ff2fd5f202c3cf5ab649cc708038a65395e990078`.

Against `../ring_cpu3`, M6's first emitted PTX SF6 restore region, strictly between its read/write `bar.sync 3,128`, changes integer ALU/address instructions **130 to 80**, and total instructions **134 to 84**. Four shared stores remain. Each of the six emitted restore regions has 50 fewer instructions; all 12 barrier-3 instructions remain. REG123/STACK0/LOCAL0 is unchanged. These are static PTX counts, excluding the preceding packed loads; they are not SASS timing, numerical or performance acceptance. Full hashes, line ranges, opcode counts and the first-region excerpts are in `ptx-comparison.json`. `compare.py` is a post-measurement read-only utility; reproduce with `python3 compare.py --old-root ../ring_cpu3 --new-root .` from this directory.

All **20 original worker files**, before/after inventories, exact execution metadata, CPU stdout and compiler outputs are retained. `execution.json` was written by the process that executed the exact normal fleet command, with rc0 and start/end times; total execution was 38.872 seconds. Its stdout hash is bound by `verification.json`. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON contents are unchanged. The archive adds no runtime-source changes.

This is CPU compilation and contract evidence only. GPU numerical correctness, graph behavior, throughput, quality and default-adoption acceptance remain separate; none is claimed here.
