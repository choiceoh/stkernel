# CPU11 no-device compilation

Source `695274d9d4c62041dacaa4fcd0861ce9e357bd4b` passed 63 pinned CPU tests with zero
failures, errors or skips, plus actual E72/I2048 CuTe and all 24 Triton remap
specializations. CUDA remained uninitialized. Completion: 2026-09-08T18:27:49.575500+09:00.

The normal fleet CPU lane ran on srv2/head using the immutable image recorded
in `submission.json`, runc, no network or CUDA devices, 4 GiB memory and two
CPUs. Host MemAvailable before launch was 96.11 GiB;
the existing 12 GiB guard stayed enforced and no serving memory was reclaimed.
`fleet.log` retains the original test and compiler output.

CuTe resources remain REG168 / STACK112 / SHARED1024. Its PTX shrank from
958390 to 955476 bytes and its cubin from 308384 to 307264 bytes. These are
compiler observations, not measured GPU latency. All 24 remap PTX and compiler
hashes match CPU10. Their whole cubin hashes differ; no cause is inferred
without a separate section inspection. See `compiler-inspection.json`.

The [allocator inspection](row-address-inspection.md) traces all seven
unroll/tail copies of the changed expression. Each goes from 14 PTX arithmetic
instructions to two (`shl` plus `add`), removing 84 static instructions in
total. The following nine address/store instructions match under register
renaming at every site. Counts exclude the row atomic and expert-base load;
they are not SASS instructions, per-request savings or a speedup percentage.
Every referenced instruction line was rechecked against the receipt-bound PTX.

The separate local compose and core logs are retained as `local-compose.log`
and `local-core.log`: 6795 core checks, 38 megakernel regressions and zero
fleet regressions. Their host Torch skips remain explicit; the pinned 63-test
result above has zero skips.

All 57 original files were re-read and hashed
on the remote host, then verified after local transfer. PTX and cubin files
are gzip-compressed here; `source-manifest.json` hashes their original bytes.
All 13 mounted-source and 18 contract-source hashes match the recorded frozen
commit. The full compiler receipt also passed `validate_compile_evidence`
against the clean remote source. Compiler artifacts and CPU contracts do not
establish GPU numerics, sanitizer success, latency or serving acceptance.

Verify the archive with `shasum -a 256 -c SHA256SUMS`.
