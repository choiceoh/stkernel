# CPU12 / CPU13 Q0 dual-warp evidence

Both normal CPU runs passed **239 tests (0 failures/errors/skips) and 9 fresh CuTe lowerings**. Each run contains 3 same-source EP4 baseline and 6 hybrid lowerings, including separate hybrid Q0 single-warp/dual-warp variants. This package establishes CPU contract, emitted-artifact and source evidence only; it makes no GPU correctness, speed, readiness or adoption claim.

| Run | Frozen source | Original result | Outcome |
| --- | --- | --- | --- |
| CPU12 | `b36dc7f1b19e471fa3280536e244f1711311eaea` | `cpu12/originals/result.json` | 239 tests / 9 lowerings PASS |
| CPU13 | `407271d9cf73f5db6192a8410c68f152e67301cb` | `cpu13/originals/result.json` | 239 tests / 9 lowerings PASS |

Each `cpuN/originals-manifest.json` maps all **29 original worker files**: unchanged result/contracts JSON plus 9 PTX, 9 cubins and 9 resource logs. The 27 artifact files are byte-identical between the two independently completed runs. Duplicate resource payloads also share hashes, leaving **23 unique gzip artifact payloads** under `artifacts/`. Decompression reproduces the exact original bytes, SHA and size; no source or receipt is substituted. Top-level `manifest.json` indexes both runs, compressed originals and shared payloads. `SHA256SUMS` covers the staged files.

Both collections verified **23 mounted and 84 contract source hashes** against their respective frozen git objects, identical before/after original bytes/size/mtime/inode and clean matching worker HEADs. Actual frozen `validate_compile_matrix`, `validate_artifacts` and capsule-runtime validators were re-run over stored files, without compiler/test/GPU imports. Those three unchanged validator sources are compressed once. The original image receipt is `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`; capsule SHA is `b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`. Collection did not re-inspect Docker. CUDA remained uninitialized and contracts ran in an isolated process.

## Scoped PTX audit

`audit/cpu12-q0-ptx.json` and `.md` preserve the original review. It observed route metadata loads after the common CTA publication barrier inside the persistent batch loop; no cross-batch hoist/reuse was observed. The post-Q0 20,635-line tail was identical after consistent register/label renaming. `audit/artifact-links.json` binds its control/candidate PTX and cubin to both runs' actual hashes. `audit/cpu13-q0-audit-reuse.json` records the actual-byte and receipt comparison. CPU13 reuses only that scoped audit because all four audited bytes, keys, specialization and resources match; it does not inherit a GPU result. The original reports' private path strings are historical; use the links file for stored payloads.

The dynamic rows report REG168 / STACK112 / LOCAL0 / static SHARED1024. Stack size alone does not measure spills; dynamic/total shared allocation remains null. PTX ordering is not final SASS or GPU race-freedom proof.

## Preserved preparation observations

- `observations/submit7/`: original local submission and fleet startup logs record `accepted=false`, rc3, PREPARE refusal **before a GPU hold** because relevant main changes were missing. This was not a GPU test.
- `observations/merge13/`: initial failure was the GLM model manifest/docs count; retry failure was a proof-marker source literal. The final child returned 0 and printed **7021 checks + 50 megakernel regressions**, but its wrapper receipt is `passed=false`, `coverage_complete=false` because torch/other checks were skipped. Parent-reported outer rc3 is labelled separately from the original child rc0. All three original JSON/log pairs are compressed intact. Their benign local Python test commands remain auditable; no field is redacted.
- CPU12's first uncompressed transfer timeout is retained under `cpu12/collection-attempt1.json`. The gzip retry required the same before inventory; it did not alter worker evidence. CPU13 used gzip initially. Existing CPU8/9/10 failure evidence stays in the prior `ep76_cpu11` archive and private originals.

CPU launch execution commands, environment/container dumps and queue metadata remain private. Benign local core-test commands are retained within the original core JSON receipts. Before/after source inventories and original private SHA inventories are preserved; entries absent from this staging were intentionally not selected. Packaging ran only local hashing, deterministic compression and disclosure checks. No original/private archive, tracked source, GPU, service or queue was changed. This directory is the durable repository copy of that completed packaging. `delivery.json` records the original staging inventory/README/manifest hashes and this repository destination; the original staging directory remains private and unchanged.
