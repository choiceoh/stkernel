# EP76 CPU2 PASS originals

Source `b41d0da24059379d41c079626cc67c3e83cd14e3`, normal fleet CPU session `epdecode76cpu0910v2`, worker srv4. Terminal rc0 after 142.952s. The original CPU gate reports **181 tests, zero failures/errors/skips**, isolated CPU contracts and **23 actual CuTe lowerings**: baseline seven local, ten global, two dynamic and four optimized static.

The four optimized M6 local/global M4/M6/M8 variants record actual CuTe storage100352B plus cubin static shared1024B, total101376B equal to the asserted block limit. All four use123 registers and zero stack/local bytes. Cache lengths are20 local and24 global. The immutable source validators were replayed over every original artifact path/hash and specialization. Resource values are compiler metadata, not runtime timing or measured occupancy.

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
| opt-static/M6-local | 123 | 0 | 0 | 1024 |
| opt-static/M4-map288-i32 | 123 | 0 | 0 | 1024 |
| opt-static/M6-map288-i32 | 123 | 0 | 0 | 1024 |
| opt-static/M8-map288-i32 | 123 | 0 | 0 | 1024 |
| dynamic/M33 | 168 | 112 | 0 | 1024 |
| dynamic/M8192 | 168 | 112 | 0 | 1024 |

All71 original worker files remain unchanged after decompression. Original22 mounted source hashes and43 contract hashes match immutable source `b41d0da24059379d41c079626cc67c3e83cd14e3`. The imported stock BF16 helper identity and binding capsule receipt validate. Both receipts record CUDA uninitialized and successful final runtime recheck. Before/after inventories bind all files; image and worker HEAD/clean state are retained. CPU1's four test fixture errors remain a separate failed prerequisite and are never reinterpreted as PASS.

Collection used remote file reads and private `/tmp` writes only. The frozen pure artifact validator performed local file checks; no tests, compilation, GPU or service operations were run. This archive establishes CPU compilation and contracts only. GPU numerics, graph behavior, throughput, quality and adoption acceptance remain separate.
