# CPU19: shared FC1 A and grouped scatter lowering passed

Normal fleet CPU session `epdecodecpu0909v19` completed on srv1 through the
srv2 scheduler on 2026-09-09 11:20:28–11:21:08 KST (39.56 seconds).
Frozen source was `cd293e19c146bd52c3419b064af209b57b2555d9` at
`/home/choiceoh/stkernel-ep-onepass-0909-19`. Payload, copy and outer codes
were all 0.

**112 tests passed, with zero failures, errors or skips.** Original evidence
records `verdict=PASS`, `phase=complete`, `cuda_initialized=false`, and
`binding_runtime_rechecked=true`. All 28 kernels compiled: three micro CuTe,
one dynamic-prefill CuTe and 24 Triton preparation variants. CPU18's failed
first lowering remains separately preserved; this run uses the corrected
shared-branch temporary initialization.

The M32/top8 candidate has all three receipt flags enabled: `scatter_fp32`,
`ep_direct_scatter`, and `shared_fc1_a`. Its cache key has the three matching
version tags. M64/top1/FP32 and M64/top8/BF16 retain their prior flags/keys;
both control PTX and cubin files are byte-identical to CPU17.

Actual shared-candidate PTX has four unrolled FC1 stages, each with exactly
six TMA instructions on a single matching stage barrier: A, SFA, gate B/SFB,
and up B/SFB. Each emits `expect_tx` for 27,648 bytes. First stage lines
4380–4435 targets shared offsets 2048,51200,18432,53248,34816,55296 on
barrier +16; the second uses the corresponding alternate buffers/barrier +24.
All four stages are checked. Consumer end-of-FC1 `bar.sync 2, 160` is at1571;
DMA's matching barrier4677 precedes its first FC2 TMA4706. Final task barriers
4227/4827 remain. There are32 static TMA sites:24 FC1 and8 FC2, compared with
40 in the previous separate gate/up design. These are compiled instruction
sites and declared transaction sizes, not measured DRAM traffic or speed.

Grouped scatter also lowered as intended. Each FC2 output block
(2839–3214,3215–3547,3548–3881,3882–4223) contains two ID loads, two weight
loads, and two row guards, reduced from CPU17's16 ID/16 weight loads and16
row guards. Its16 FP32 vector RED sites and common128-thread post barrier
remain. There are no shared output stores/BF16 shared loads/proxy fences in
these blocks. Metadata stays per lane, and the actual CuTe host validator
checks pair adjacency, full coverage and the two-row grouping.

The candidate PTX is217,006 bytes (CPU17 direct path:262,956 bytes).
Resource use remains `REG161 STACK0 SHARED1024 LOCAL0`; emitted PTX has no
local loads/stores. M64 controls remain REG222/220 with zero stack/local.
These observations establish emitted code structure and resource counts,
not GPU numerical correctness, runtime occupancy, or consumer speed.

`evidence.tar.gz` preserves61 original files:28 PTX,28 cubin,4 resource logs,
and the result. `result.json` duplicates the original bytes. `head/` preserves
submission, driver, exit and complete fleet log. Source hashes match the
submission and runtime receipts. Both frozen checkouts were clean/full with
no alternates before/after collection; all worker/head/local original hashes
matched. Image identity remains
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`;
the116-file capsule passed strict validation against manifest
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.
The12GiB admission guard,4GiB/2CPU limits and no-device runtime remained.

`verify.py` reproduces original-file/source/completion checks, exact cited PTX
instructions, resource counts and raw control equality against CPU17. Its
saved output is `verification.json`; observations are in `capture.json`.
Collection only read/copied existing evidence after GPU19 submission; it did
not submit tests, use GPUs or modify frozen sources. GPU and serving evidence
are separate. `SHA256SUMS` covers every archive file except itself.
