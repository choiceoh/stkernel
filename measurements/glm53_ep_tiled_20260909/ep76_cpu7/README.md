# EP76 CPU7 register-max PASS originals

Source `ca076d35e64a6a19e90dffe54054269d1a5e1887`, normal fleet CPU session `epdecode76cpu0910v7`, worker srv4. Terminal rc0 after 149.248s. **183 tests PASS, zero failures/errors/skips; 23 actual CuTe lowerings**: seven local, ten global, four optimized static, two dynamic. Frozen artifact and layout validators were replayed against all original files locally. This is file validation, not another test or lowering run.

The four optimized variants carry `glm53_ep_static_sf6_q1_register_max_v5`. Each has an actual source-bound `q1_register_layout` receipt generated during CuTe setup: identity `partition_D` and `partition_S` shape/dense register index checks cover 128 threads and 2,048 values; the full sC1 mapping covers 4,096 bytes. The R0..8 records preserve 512-byte maximum scratch ownership, matching peer loads, packed A2 and scale consumer addresses and their hashes. The receipt binds the actual fast/precise mode and selected low-row branch. Prior FC1, FC2 and Q1 pair receipts cannot substitute for this witness.

The resource table below is the actual cubin report, including any nonzero stack or local values. Dynamic shared storage is checked against 98,304 bytes; actual cubin static storage must stay within 1,024 bytes and total within 101,376 bytes. `q1-ptx-verification.json` preserves observed shuffle instructions and local load/store counts without assuming the previous pair-v4 instruction count. `LOCAL=0` alone is not a no-spill or performance claim. No byte-equality claim is made against CPU5/6; their original evidence remains separate.

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
| opt-static/M4-map288-i32 | 128 | 8 | 0 | 1024 |
| opt-static/M6-map288-i32 | 128 | 8 | 0 | 1024 |
| opt-static/M8-map288-i32 | 128 | 8 | 0 | 1024 |
| dynamic/M33 | 168 | 112 | 0 | 1024 |
| dynamic/M8192 | 168 | 112 | 0 | 1024 |

All 71 original worker files retain exact decompressed bytes. Their 22 mounted and 47 contract hashes match immutable git `ca076d35e64a6a19e90dffe54054269d1a5e1887`, including the native source oracle from `ep76_onepass4`. Imported BF16 helper and binding runtime receipts validate. Both receipts record CUDA uninitialized and successful final runtime recheck. Worker inventories bracket collection and bind source HEAD/clean state and capsule manifest. Image identity comes from the original submission command; collection invokes no Docker command.

The original result, contracts, execution metadata, fleet stdout, PTX, cubins and resource logs are preserved with separate original/stored hashes. Collection utilities are separate from measured source. No tests, compiler, GPU or service operation is performed by these collection/finalization helpers. CPU compilation and static layout witnesses do not establish GPU numerics, graph correctness, throughput, quality or default acceptance.
