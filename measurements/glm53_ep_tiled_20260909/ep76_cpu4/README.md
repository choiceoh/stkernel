# EP76 CPU4 PASS originals

Source `3fab1ce81a79936b3b76fa4d87b447d9f55bacba`, normal fleet CPU session `epdecode76cpu0910v4`, worker srv4. Terminal rc0 after 151.246s. The original CPU gate reports **181 tests, zero failures/errors/skips**, isolated CPU contracts and **23 actual CuTe lowerings**: baseline seven local, ten global, two dynamic and four optimized static.

The four optimized M6 local/global M4/M6/M8 variants record actual CuTe storage 98,304B plus cubin static shared 1,024B, total 99,328B below the asserted 101,376B block limit. All four use 96 registers and zero stack/local bytes. The previous CPU2 FC2 candidate reported 123 registers; this is a code-generation observation, not a performance result. Cache lengths are 20 local and 24 global, with final tag `glm53_ep_static_sf6_fc1_register_v2`. Each actual lowering records a proven 128-thread, two-stage copy mapping with 2,048-byte coverage, four K blocks, copy shape `((1, (1, 4)), 4, 4)`, and `words_per_thread=[16]`. The frozen pure mapping-receipt validator was replayed against these original witnesses; the mapping was not recomputed or lowered during archival. The immutable source validators were replayed over every original artifact path/hash and specialization. Resource values are compiler metadata, not runtime timing or measured occupancy.

| Variant | Registers | Stack bytes | Local bytes | Static shared bytes |
|---|---:|---:|---:|---:|
| static/M4 | 123 | 0 | 0 | 1024 |
| static/M6 | 123 | 0 | 0 | 1024 |
| static/M8 | 123 | 0 | 0 | 1024 |
| static/M12 | 118 | 0 | 0 | 1024 |
| static/M16 | 118 | 0 | 0 | 1024 |
| static/M24 | 118 | 0 | 0 | 1024 |
| static/M32 | 118 | 0 | 0 | 1024 |
| global-static/M4-map288-i32 | 123 | 0 | 0 | 1024 |
| global-static/M6-map288-i32 | 123 | 0 | 0 | 1024 |
| global-static/M8-map288-i32 | 123 | 0 | 0 | 1024 |
| global-static/M12-map288-i32 | 118 | 0 | 0 | 1024 |
| global-static/M16-map288-i32 | 118 | 0 | 0 | 1024 |
| global-static/M24-map288-i32 | 118 | 0 | 0 | 1024 |
| global-static/M32-map288-i32 | 118 | 0 | 0 | 1024 |
| global-static/M6-map288-i64 | 123 | 0 | 0 | 1024 |
| global-static/M6-offset216-i64 | 123 | 0 | 0 | 1024 |
| global-static/M6-empty-i64 | 126 | 0 | 0 | 1024 |
| opt-static/M6-local | 96 | 0 | 0 | 1024 |
| opt-static/M4-map288-i32 | 96 | 0 | 0 | 1024 |
| opt-static/M6-map288-i32 | 96 | 0 | 0 | 1024 |
| opt-static/M8-map288-i32 | 96 | 0 | 0 | 1024 |
| dynamic/M33 | 168 | 112 | 0 | 1024 |
| dynamic/M8192 | 168 | 112 | 0 | 1024 |

All71 original worker files remain unchanged after decompression. Original22 mounted source hashes and44 contract hashes match immutable source `3fab1ce81a79936b3b76fa4d87b447d9f55bacba`. The imported stock BF16 helper identity and binding capsule receipt validate. Both receipts record CUDA uninitialized and successful final runtime recheck. Before/after inventories bind all files; image and worker HEAD/clean state are retained. CPU1 failure and CPU2 PASS remain separately archived; neither is relabelled as CPU4.

Collection used remote file reads and private `/tmp` writes only. The frozen pure artifact validator performed local file checks; no tests, compilation, GPU or service operations were run. This archive establishes CPU compilation and contracts only. GPU numerics, graph behavior, throughput, quality and adoption acceptance remain separate.

CPU3 to CPU4 comparison: all 69 original PTX/cubin/resource-log files are byte-identical, as are the binding runtime and imported helper receipts. All 22 CPU-mounted hashes match; of 44 contract paths, only `probes/glm53_ep_tiled_compile.py` and `probes/run_glm53_ep_tiled_cpu.py` changed. The copy-layout validation is now also invoked after artifact transfer. Native EP and PREP source/generated files are byte-identical.

The complete immutable git diff additionally changes `overlay/modules/glm53_model/glm53_fp8_dense.py` and its generated counterpart to add startup device-memory accounting/logging. Those two files are outside this CPU gate's mounted/contract set; their hashes and full diff are preserved separately. This distinction does not establish serving startup or throughput equivalence. See `cpu3-artifact-comparison.json`, `cpu3-source-comparison.json`, and `cpu3-to-cpu4-source.diff`.
