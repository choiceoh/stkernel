# CPU12 no-device compilation

Source `0ae4c08f49c22122f499771ca6cefab1113b3476` passed 65 pinned CPU tests with zero
failures, errors or skips, plus actual E72/I2048 CuTe and all 24 Triton remap
specializations. CUDA remained uninitialized. Completion: 2026-09-08T19:02:23.578746+09:00.

The normal fleet CPU lane ran on srv2/head using the immutable image recorded
in `submission.json`, runc, no network or CUDA devices, 4 GiB memory and two
CPUs. Host MemAvailable before launch was 92.02 GiB;
the existing 12 GiB guard stayed enforced and no serving memory was reclaimed.
`fleet.log` retains the original test and compiler output.

REG 168 / STACK 112 /
SHARED 1024. The comparison with CPU11 is in
`compiler-inspection.json`. CuTe PTX byte equality:
False; whole CuTe cubin hash equality:
False. All 24 remap PTX hashes match
CPU11: True. Whole remap cubin hash equality
holds for 0/24; no cause is inferred for
binary differences without further section inspection.

The [scale-address inspection](scale-address-inspection.md) traces all ten
Q0 scale-store sites (seven equal-scale and three varied-scale compiler
copies). Each address dependency graph falls from 41 to nine PTX arithmetic
instructions, removing 320 instructions across the static artifact. Every
one of the 500 before/after instruction references and all shared-load/store
endpoints were independently rechecked against the receipt-bound PTX. The
input SF-column setup, physical-row load, payload store, pointer widening and
scale-byte store are excluded. These are not executed counts, SASS or speedup.

The separate local core gate passed 6795 checks and 38 megakernel regressions;
its original log retains host Torch skips and ran zero fleet regressions.
The pinned 65-test result above has zero skips. Initial insufficient head and
worker memory checks did not launch a compiler. The unchanged 12 GiB guard
later admitted the normal head CPU runner with 92.02 GiB available during a
boot transition, without reclaiming serving memory. The original submission
records that sequence and the same immutable image / 4 GiB / two-CPU limits.

All 57 original files were re-read and hashed
on the remote host, then verified after local transfer. PTX and cubin files
are gzip-compressed here; `source-manifest.json` hashes their original bytes.
All 13 mounted-source and 18 contract-source hashes match the recorded frozen
commit. The full compiler receipt also passed `validate_compile_evidence`
against the clean remote source. Compiler artifacts and CPU contracts do not
establish GPU numerics, sanitizer success, latency or serving acceptance.

Verify the archive with `shasum -a 256 -c SHA256SUMS`.
