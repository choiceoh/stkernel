# Onepass8 supplemental source/log observations

This is local read-only analysis of existing evidence, **not an additional GPU test or a performance acceptance**. The original onepass8 archive remains authoritative: B completed with its recorded quality rejection; A was refused by preflight before boot, so no matched B/A result exists. Nothing here changes cold/quality classification, subtracts delays, or proves v9 readiness.

## Contents and scope

- `jit/` preserves the original audit/readback and gzip-compressed exact prefix, through-after and delta bytes. The canonical record's290411-byte prefix plus70787-byte delta equals its361198-byte after boundary. All three hashes match. The original retained log was361353 bytes; the later155 bytes are outside the record and were never included in this audit.
- The monitor reported **seven JIT events**: six in the first2K request and one in fixed2K rep0. The fourth sequential POST segment, attributed to32K, contains no JIT/compile warning. That is **not** a global or per-request no-compilation guarantee: hook coverage/suppression and non-head compilers were not audited, and durations were absent. No numerical correction or cause of the32K TTFT is inferred.
- `gmu-original/` preserves both original review files, including the old “pinned KV source unavailable / ordering unverified” statement. They are not rewritten.
- `pinned-image/` contains the original receipt and compressed exact two image-source files. `gmu-code-order-followup.json` and `.md` resolve only the subsequently inspected ordering: override capacity is applied at2616–2617 **before** maximum-length admission at2628. This does not prove actual memory capacity or a future boot succeeds at GMU0.60; fixed1056 blocks do not shrink with that profiling budget.

`manifest.json` records original/stored SHA-256 and sizes. `verification.json` links the exact canonical B record already stored at `measurements/glm53_ep_tiled_20260909/ep76_onepass8/job/onepass.jsonl.gz`, avoiding another copy. The existing records and all input audit/source files were unchanged. `SHA256SUMS` covers every staged file except itself. Gzip decompression reproduces original bytes; no original was redacted. No raw Docker environment/command dumps were selected.

All work here was local hashing, source/log reading and compression. No new remote reads, tests, compilation, inference, GPU reservation, service action or repository edit. This directory is the durable repository supplement. `delivery.json` records the original private staging inventory/README/manifest hashes and this repository destination. The original staging files remain unchanged.
