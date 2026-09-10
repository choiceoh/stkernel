# EP76 CPU1 failed prerequisite

Source `680a899d7de0c823905b89eed1e27cfc375feaf8`, normal fleet CPU session `epdecode76cpu0910v1`, worker srv4. Terminal rc1 after 131.659s. This is a **failed CPU gate**: 181 tests, zero failures, four errors, zero skips. All four errors are missing `_report_decode_opt` in the existing checkpoint test AST namespace. The canonical failure and raw traceback are preserved; later fixes do not change this verdict.

Before the test failure, all **23 CuTe lowerings** produced PTX/cubin/resource artifacts: baseline seven local, ten global, two dynamic and four optimized static. The optimized M6 local and global M4/M6/M8 record actual CuTe storage100352B plus cubin static shared1024B, total101376B equal to the asserted block limit. All four use123 registers, zero stack/local bytes. Cache lengths are20 local and24 global. This is compilation/resource evidence, not GPU execution or speed evidence.

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

All71 original worker files are preserved unchanged after decompression. Original and stored hashes are listed in `manifest.json`; inventories before/after copy match. The original22 mounts and43 contracts were compared against immutable source `680a899d7de0c823905b89eed1e27cfc375feaf8`. Actual imported BF16 helper identity and initial capsule receipt validate. CUDA is recorded uninitialized, but **post-runtime recheck is absent** because the gate failed; no successful final recheck is claimed. Current worker HEAD in capture files may be a later CPU2 source and does not rebind CPU1.

Collection only read files remotely and wrote this private `/tmp` archive. The frozen pure artifact validator was replayed over local original bytes; no tests, compilation, GPU or service operations were run. Numerical, graph, quality and performance acceptance are not established.
