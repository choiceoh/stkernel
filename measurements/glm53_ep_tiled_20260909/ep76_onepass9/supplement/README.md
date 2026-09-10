# v9 matched numeric and log audit supplement

This durable supplement preserves original read-only audits of v9 on frozen source `407271d9cf73f5db6192a8410c68f152e67301cb`, image `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`. The exact original two-row record is [../job/onepass.jsonl.gz](../job/onepass.jsonl.gz), raw SHA `039d91324a657eae023147143a5faedfc2110722b34e0174c55dba5011acf20a`. Nothing here changes the canonical quality/judge outcome or grants adoption.

## Stable paths

| Private source | Stored package path |
| --- | --- |
| `/tmp/glm53-ep76-onepass9-matched-audit/audit.json` and `.md` | `matched/audit.json`, `matched/audit.md` |
| `/tmp/glm53-ep76-onepass9-diagnosis/audit.json` and `.md` | `diagnosis/audit.json`, `diagnosis/audit.md` |
| Diagnosis `{B,A}-log-prefix.raw`, `-log-through-after.raw`, `-onepass-delta.raw` | Same six names under `diagnosis/`, with `.gz` appended |
| `/tmp/glm53-ep76-onepass9-{B,A}-ready-audit.json` | `ready/B-ready-audit.json`, `ready/A-ready-audit.json` |
| `/tmp/glm53-ep76-onepass9-A-ready-markers.json` | `ready/A-ready-markers.json` |

The original JSON still contains its historical `/tmp` paths; use this table and `manifest.json` to locate the durable copies. No original audit/receipt is rewritten or redacted. All six raw log files use deterministic gzip with mtime0. Original/stored SHA-256 and sizes are recorded separately, and decompression reproduces exact bytes. The parent archive's existing files, manifest and checksum inventory are unchanged; this subdirectory has its own manifest and `SHA256SUMS`.

## What the audits establish

The matched audit recomputes fixed1024×3 output B76.54038 → A72.50972 tok/s (−5.2661%) and fixed-window engine21.53830 →21.41215 step/s (−0.5857%). Candidate A misses76. Both canonical facts18/18 and Korean0/8 pass; B proof3/3 and A4/4 remain recorded. All eight request hashes match, while all eight output hashes differ. Full response text is unavailable to independently regenerate those hashes or rejudge facts. The original canonical judge remains incomplete/unresolved, n1/no floor; the audit is not a new statistical or kernel numerical acceptance.

All four saved ranks per arm are bound to source/image and the same actual GMU0.60/1056blocks/maxlen1048576/maxseq4/maxbatch8192. The original full-identity audit reports only Q0_DUAL_WARP0→1 differing in environment. Ready receipts preserve73 mounts, source14/loader15 pins, first-eligible-layer12cases/72candidate+72stock per rank, Q0 false/true and SF6 FINALIZED42. These are saved startup/selection proofs, not current-service health or independent distributed-sum validation. Ready launch markers may include profile calls; fresh execution and quality remain canonical row evidence. Memory fields refer to distinct startup times; missing device-free-at-ready stays null.

The exact canonical head-log prefix plus delta equals each after boundary. **Each arm has seven logged request-time JIT events**: six in initial2K and one gumbel warning in fixedrep0. The attributed32K segments (B delta48–90; A49–92) contain no compile-pattern message. Hook/suppression/all-rank coverage and JIT durations were not established: absence of a log message is not proof of no JIT, and no time is subtracted from TTFT. Single32K/128K observations do not establish repeatable prefill improvement.

Exact per-fixed accepted/drafted counters are absent. Whole-onepass acceptance and10-second windows crossing boundaries cannot replace them. A rep2's higher engine-window rate but lower output rate and323 vs285 SSE text chunks are consistent with changed generation progress, not proof of its cause; chunks are neither tokens nor engine steps. No native-kernel or speculative-acceptance cause is established, and the missed76 target is retained.

Packaging used local file/hash checks and compression only: no additional tests, remote access, inference, GPU/queue/service action, runtime source change or commit.
