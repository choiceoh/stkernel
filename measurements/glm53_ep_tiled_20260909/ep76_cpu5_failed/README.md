# EP76 CPU5 failure originals

Source `48df51174e0f76a726ca87cc7652dc57f2ad8724`, normal fleet CPU session `epdecode76cpu0910v5`, worker srv4. Terminal rc1 after 183.309s. **183 tests: two failures, zero errors/skips. Full CPU gate FAIL.** Both failures reach the retired `len(branches)==5` baseline AST extractor. Preserve the original failed result and contract receipts; successful lowerings do not change that verdict.

All **23 actual CuTe lowerings** were produced: seven local, ten global, four optimized static, two dynamic. Frozen artifact and physical mapping receipt validators were replayed against all original files locally. This repeats file validation only, not tests or lowering. The four optimized variants carry `glm53_ep_static_sf6_q1_pair_v4`, actual fast-math mode and source-bound R0..8 address/ownership digests. Each records dynamic shared 98,304B plus cubin static shared 1,024B = 99,328B, within the 101,376B cap.

Actual Q1 PTX contains three `shfl.sync.idx.b32` instructions per optimized kernel, clamp/mask field31 and member mask-1 (`0xffffffff`). The first peer is warp-lane XOR1; the next two use lane AND30. Their PTX blocks reconverge before the shuffles and branch around invalid pair stores afterward. Three earlier shuffle instructions belong to preexisting code. Exact paths, hashes, line numbers and context are preserved in `q1-ptx-verification.json`. All four optimized variants use128 registers; M6-local reports8 stack bytes, the three global variants0. LOCAL0 alone is not a no-spill or performance claim.

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

All71 original worker files retain exact decompressed bytes. Their22 mounted and46 contract hashes match immutable git `48df51174e0f76a726ca87cc7652dc57f2ad8724`. Imported BF16 helper and initial binding runtime receipts validate. Both failed receipts record CUDA uninitialized; **the final runtime recheck field is absent and no completed post-recheck is claimed**. Worker inventories bracket collection and bind source HEAD/clean state and capsule manifest. Image identity comes from the exact original submission command; collection did not call Docker.

No tests, compiler, GPU or service operation was performed during collection. GPU numerics, graph correctness, throughput, quality and adoption acceptance remain separate.
