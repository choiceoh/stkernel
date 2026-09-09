# CPU10 no-device compilation

Source `a9d0d1e3e8161ee44eac02191021ea9512a3e2cc` passed 61 pinned CPU tests with zero
failures, errors or skips, plus actual E72/I2048 CuTe and all 24 Triton remap
specializations. CUDA remained uninitialized. Completion: 2026-09-08T18:19:31.585897+09:00.

The normal fleet CPU lane ran on srv2/head using the immutable image recorded
in `submission.json`, runc, no network or CUDA devices, 4 GiB memory and two
CPUs. Host MemAvailable before launch was 93.01 GiB;
the existing 12 GiB guard stayed enforced and no serving memory was reclaimed.
`fleet.log` retains the original test and compiler output.

REG 168 / STACK 112 /
SHARED 1024. The comparison with CPU9 is in
`compiler-inspection.json`. CuTe PTX byte equality:
True; whole CuTe cubin hash equality:
True. All 24 remap PTX hashes match
CPU9: True. Whole remap cubin hash equality
holds for 0/24; no cause is inferred for
binary differences without further section inspection.

All 57 original files were re-read and hashed
on the remote host, then verified after local transfer. PTX and cubin files
are gzip-compressed here; `source-manifest.json` hashes their original bytes.
All 13 mounted-source and 18 contract-source hashes match the recorded frozen
commit. The full compiler receipt also passed `validate_compile_evidence`
against the clean remote source. Compiler artifacts and CPU contracts do not
establish GPU numerics, sanitizer success, latency or serving acceptance.

Verify the archive with `shasum -a 256 -c SHA256SUMS`.
