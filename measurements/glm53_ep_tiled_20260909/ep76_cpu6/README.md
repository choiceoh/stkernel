# EP76 CPU6 PASS originals

Source `4618859c90131b33c5d9ebd85a67f1537497a357`, normal fleet CPU session `epdecode76cpu0910v6`, worker srv4. Terminal rc0 after 178.053s. **183 tests PASS, zero failures/errors/skips; 23 actual CuTe lowerings**: seven local, ten global, four optimized static, two dynamic. Frozen artifact and physical mapping receipt validators were replayed against all original files locally. This repeats file validation only, not tests or lowering.

All **69 PTX/cubin/resource files are byte-identical to CPU5** (23 of each type), including the four Q1 pair variants. Full immutable git comparison changes only `tests/test_glm53_ep_tiled_static.py`, fixing the retired AST branch extractor. CPU5 remains a separate original FAIL. All22 mounted source hashes, binding runtime and imported helper identities are unchanged; among46 contract sources only that test hash differs. Exact artifact hashes and before/after source diff are preserved alongside this README.

The four optimized variants carry `glm53_ep_static_sf6_q1_pair_v4`, actual fast-math mode and source-bound R0..8 address/ownership digests. Each records dynamic shared98,304B plus cubin static shared1,024B =99,328B within the101,376B cap. Actual Q1 PTX contains three `shfl.sync.idx.b32` instructions, clamp/mask field31 and member mask-1 (`0xffffffff`). The first peer uses warp-lane XOR1; the next two lane AND30. Exact paths, hashes, lines and context are in `q1-ptx-verification.json`. All four use128 registers; M6-local reports8 stack bytes, global M4/M6/M8 report0. LOCAL0 alone is not a no-spill or performance claim.

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
| opt-static/M6-local | 128 | 8 | 0 | 1024 |
| opt-static/M4-map288-i32 | 128 | 0 | 0 | 1024 |
| opt-static/M6-map288-i32 | 128 | 0 | 0 | 1024 |
| opt-static/M8-map288-i32 | 128 | 0 | 0 | 1024 |
| dynamic/M33 | 168 | 112 | 0 | 1024 |
| dynamic/M8192 | 168 | 112 | 0 | 1024 |

All71 original worker files retain exact decompressed bytes. Their22 mounted and46 contract hashes match immutable git `4618859c90131b33c5d9ebd85a67f1537497a357`. Imported BF16 helper and binding runtime receipts validate. Both receipts record CUDA uninitialized and successful final runtime recheck. Worker inventories bracket collection and bind source HEAD/clean state and capsule manifest. Image identity is retained from the original submission command; collection did not invoke Docker.

No tests, compiler, GPU or service operation was performed during archival. CPU compilation/contracts and byte equality do not establish GPU numerics, graph correctness, throughput, quality or adoption acceptance.
