# Bounded candidate failure diagnostics

Prepared after GPU v5 rejected concentrated6912. The original 4 PASS / 1 FAIL / 12 NOT_RUN result is unchanged. No diagnostic GPU job has been submitted.

The new capture retains the original numerical assertion and bounded row-level evidence; the existing fixture may be selected through the normal offline runner with `--diagnose-case concentrated6912`. A one-cell result is explicitly not full GPU acceptance.

`prepare-cpu17.py` is a reviewed preparer, not CPU evidence. It requires a clean published source with unchanged CPU16 kernel/composed bytes, the approved scheduler and at least 12 GiB of available host memory. It creates a fresh independent exact-revision source and submits only the normal CPU fleet command. No worker fallback or serving memory reclaim is used. A fresh complete CPU17 receipt is required before the diagnostic can run on GPU.

The initial public-serving snapshot had only 7720004 KiB of available RAM. Server admission must be checked live; this snapshot is not a new test result.

The eight new diagnostic tests and 17 wrapper tests passed without skips. Integration discovered146 tests:135 passed,11 skipped for missing host Torch/packaging,0 failures/errors. These local results do not replace the required pinned-image146-test/no-skip CPU17 gate. Source hashes and original/stored log hashes are in local-validation.json.

Read-only review confirmed the original comparison/AssertionError and candidate call count are unchanged, BF16 raw words use unsigned16-bit normalization and at most8rows/8columns are retained. It caught a stale CPU16 path assertion; that assertion now pins CPU17 and all17 wrapper tests passed again. The CPU17 preparer received independent review; no remote execution was part of that review.

At 2026-09-08T22:50:25.382301+09:00, the reviewed CPU17 preparer was executed once against published source `81b8ef91cd72f3f20a3cdf8014a8b25a51b357c1`. Admission was refused: host MemAvailable=7791744KiB, below12GiB. No CPU17 source, bundle transfer or job was created. The missing CPU17 receipt still prevents a diagnostic GPU submission. The admission JSON and exact command are preserved here.

The 23:05 KST follow-up still found no CPU17 source/job, with MemAvailable6867044KiB below12GiB. The CPU preparer was not rerun under the unchanged refusal condition; no new server workload was submitted.

Two follow-up tools are prepared, syntax-checked and reviewed, but have not been executed:

- `collect-cpu17.py --revision <actual CPU17 source SHA>` requires actual normal CPU17 completion,146 tests without skips,29 contract files,13 mounts and24 remap builds. It compares source/PTX/cubin/resources/remap bytes with CPU16, verifies immutable inputs before and after transfer and preserves mismatches in rejected staging. Original frozen Python source bytes are compressed rather than normalized.
- `prepare-diag-gpu1.py --revision <CPU17 receipt commit SHA>` requires a committed byte-identical actual CPU17 receipt and submits only `--diagnose-case concentrated6912` through the normal GPU fleet. The receipt commit must be the immediate single child of the actual CPU17 source, allowing an independent depth2 checkout. Commit the CPU17 archive once immediately after its source; do not interpose unrelated commits or bypass the ancestry guard. Diagnostic, performance and default-promotion scopes remain explicit; a diagnostic PASS cannot clear the earlier full-suite FAIL.

`followup-tools.json` records source hashes, syntax-check scope and the read-only scheduling snapshot. These tools are preparation, not additional CPU/GPU test results.
