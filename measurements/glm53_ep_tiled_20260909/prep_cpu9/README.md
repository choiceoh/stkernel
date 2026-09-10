# EP native route fusion + PREP_FUSED CPU9 originals

Source `96a8cab45a1eb049298315a77ed881ee74f04f9b`. Normal no-device CPU session `eptiledcpu0910prep9` ran through the head fleet controller with a bounded SSH CPU payload on srv4. The original receipt reports **154 tests, zero failures/errors/skips**, and **19 actual CuTe lowerings**: seven local static, ten global-route static and two dynamic. CUDA stayed uninitialized; runtime identity was rechecked and CPU contracts ran in a fresh process.

All **59 original worker files** are preserved: result/contracts JSON plus 19 PTX, 19 cubins and 19 resource logs. The source receipt's **22 mounted sources and 38 contracts** match the exact immutable git source. Actual global constructor/map fake ABI and every artifact path/hash were checked using the frozen pure validators. The pinned imported BF16 helper source identity and capsule receipt match. Before/after worker inventories confirm byte stability; image ID and current worker source HEAD/clean status are retained.

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
| dynamic/M33 | 168 | 112 | 0 | 1024 |
| dynamic/M8192 | 168 | 112 | 0 | 1024 |

Resource values are compiler metadata, not measured timing or occupancy. `LOCAL=0` alone does not establish absence of stack spills. This archive makes no GPU numerics, graph, performance, quality or adoption claim.

`failed-prerequisites7/` preserves the original CPU7 static-M4 `UNSUP_EARLY_EXIT` failure (source `66809847c78ac5338965c11a1164dde5849e8515`); no contracts receipt existed. `failed-prerequisites8/` preserves CPU8's original FAIL result and FAIL contracts receipt (source `4e93663900b6ced9b22c6b8abc0a877260d50e47`): 19 lowerings completed, but test-module discovery failed before the expected suite ran. Neither prior attempt is relabeled PASS. Their exact local execution/stdout originals are preserved, and absence is recorded in the inventories. CPU8's bulky compiler artifacts remain in their original worker directory and are not duplicated here.

All logs/PTX/cubins use deterministic gzip with both original and stored hashes/sizes in `manifest.json`; raw JSON bytes are unchanged. Collection involved file reads and private `/tmp` writes only. No tests, compilation, fleet submission, GPU or service action was performed.
