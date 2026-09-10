# EP2×TP2 CPU11 evidence

CPU11 passed **217 CPU tests (0 failures/errors/skips) and 8 fresh CuTe lowerings** on frozen source `c1c1c478145ab332a4e9cdf0850c25c66a60cd5e`. The matrix contains 3 same-source EP4 baseline lowerings and 5 EP2×TP2 candidate lowerings. This is CPU compilation and contract evidence; it does not establish GPU numerics, throughput, or default adoption.

`originals/result.json` and `originals/contracts.json` preserve the original bytes. All 8 PTX, 8 cubins, and 8 resource logs are retained with deterministic gzip compression. `manifest.json` records both original and stored SHA-256 values; `SHA256SUMS` covers the published files. `fleet.log.gz` preserves the original CPU log.

The original collection verified 23 mounted sources and 79 contract sources against the frozen git revision and revalidated the frozen artifact/matrix validators without compilation or tests. Worker evidence bytes, sizes, timestamps, and inodes were unchanged across collection. CPU11 explicitly records `cuda_initialized=false`, isolated contract execution, and binding-runtime recheck. Packaging performed only local file-integrity, compression, and text-disclosure checks; it did not repeat the test suite or compilation.

Image identity recorded by the original execution: `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`. Capsule manifest SHA-256: `b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`. Collection did not re-inspect Docker. Raw commands, environment/container dumps, and queue metadata remain outside this published archive.

## Compiled resource scope

| Specialization | Count | Registers | Stack bytes | Static shared bytes | Constructor shared estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| M6 static (baseline global; hybrid local/global) | 3 | 123 | 0 | 1024 | 98304 |
| M32 static (baseline global; hybrid local/global) | 3 | 118 | 0 | 1024 | 101376 |
| M8192 dynamic (baseline/hybrid) | 2 | 168 | 112 | 1024 | Not recorded |

The constructor estimate is not an actual dynamic-shared allocation measurement. Actual dynamic/total shared bytes remain unreported. Stack size alone does not establish spill stores or loads. Full per-specialization receipts are in `verification.json` and the original resource logs.

## Preserved failed preparations

| Run | Frozen source | Recorded failure | Completed result rows / emitted artifacts |
| --- | --- | --- | --- |
| CPU8 | `44e86d8afaa0f7771966f80d9866e06fab89b0b4` | Baseline M32 resource gate incorrectly asserted `dynamic_shared + shared <= 101376` | 1 static row; 2 PTX, 2 cubins, 2 resource logs |
| CPU9 | `ea3529160e2b069f1925bda21bb91ce5b25aaa58` | Dynamic observer read nonexistent `_Pointer.element_type` | 2 static rows; 2 PTX, 2 cubins, 2 resource logs |
| CPU10 | `4a8b80d121c3599760bef864358ef8521bb34774` | Dynamic compile completed, then `json.dumps(key)` rejected `torch.dtype` before appending its result row | 2 static rows; 3 PTX, 3 cubins, 2 resource logs |

Each `failures/cpuN/` retains the original result/log and original artifact hash inventory. Their contract tests were not reached, so absent `contracts.json` is expected. Missing final CUDA recheck fields are not interpreted as `false`. Failure artifacts were not relabeled as passes. CPU9 collection observed a later clean worker HEAD (`4a8b80d…`); its bytes were instead bound to the recorded frozen `ea352916…` git objects. CPU10's two rejected collection/finalization attempts are retained in provenance; finalization used the same captured originals. Full failed PTX/cubin payloads remain in their original private archives.
