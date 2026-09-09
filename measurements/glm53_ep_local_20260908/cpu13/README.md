# CPU13 no-device compilation

Source `18148116a7abb242741d5b112c8d735353d9fc71` passed 68 pinned CPU tests with zero
failures, errors or skips, plus actual E72/I2048 CuTe and all 24 Triton remap
specializations. CUDA remained uninitialized. Completion: 2026-09-08T19:34:23.620075+09:00.

The normal fleet CPU lane ran on srv2/head using the immutable image recorded
in `submission.json`, runc, no network or CUDA devices, 4 GiB memory and two
CPUs. Host MemAvailable before launch was 39.40 GiB;
the existing 12 GiB guard stayed enforced and no serving memory was reclaimed.
`fleet.log` retains the original test and compiler output.

REG168 / STACK112 / SHARED1024 remain unchanged from CPU12. CuTe PTX
shrinks from 942077 to 941539 bytes and cubin from 300864 to 298456 bytes.
Static local-memory loads/stores stay at four/five. All 24 remap PTX hashes
match CPU12; whole remap cubin hashes differ. No cause is inferred for those
binary differences without a separate section inspection.

Lane 0 compares each already-loaded transformed scale while publishing its
raw bits. The existing per-warp slot31 carries count in bits0–3 and equality
in bit4, without another shared store/load or slot. Consumers decode count
before the empty check, then reload the first raw scale and use the published
flag instead of scanning all selected scales. The comparison preserves
Float32 unordered-NaN and signed-zero semantics, including the first-NaN
single-route case. Varied-scale quantizer input loads remain unchanged.

The [PTX inspection](scale-state-inspection.md) finds seven consumer scale-load
sites reduced to one; the seven static Float32 comparisons move into the
producer. Count/state shared loads and stores remain one each. For a positive
count C the consumer executes C scale reads before and one after; C=0 reads
none, and C=1 saves none. These are shared warp broadcasts, not 32 independent
transactions. Moving comparisons into lane 0 can change scheduling, so this
is not a GPU speedup result. `verify_scale_state_inspection.py` independently
checks both original PTX/cubin hashes and every recorded excerpt reference;
its receipt is `scale-state-reference-verification.json`.

The [route-state contract](route-state-contract.json) binds the candidate and
pinned inherited source. The compressed original `stock-gated.py.gz` matches
the runtime guard's SHA; 37 excerpt lines were rechecked and the inherited
kernel has no route-cache reference after the override returns. This source
scope check is not GPU race verification. The actual-source CPU oracle uses
independent consumer namespaces for all four warps and 32 lanes, tests packed
empty state16 before scale reads, route filtering, raw bits, stale state and
canaries, and verifies the existing publication barrier's position.

Separate local tests passed seven scale-cache cases and the core gate's 6795
checks / 38 megakernel regressions. Its log preserves host Torch skips and ran
zero fleet regressions; the pinned 68-test receipt above has no skips. The
operational scheduler was an approved-main ancestor whose bench, launcher,
profile and test files exactly matched current approved main. Its checkout
was kept fixed for the other active holder; only unrelated probe files had
advanced upstream.

All 58 original files were re-read and hashed
on the remote host, then verified after local transfer. PTX and cubin files
are gzip-compressed here; `source-manifest.json` hashes their original bytes.
All 13 mounted-source and 18 contract-source hashes match the recorded frozen
commit. The full compiler receipt also passed `validate_compile_evidence`
against the clean remote source. Compiler artifacts and CPU contracts do not
establish GPU numerics, sanitizer success, latency or serving acceptance.

Verify the archive with `shasum -a 256 -c SHA256SUMS`.
