# SF6 direct-prefill onepass: A recorded; B running

The canonical A/B began at 2026-09-09 05:25:29 KST after 1.2 seconds in the
queue. This is an execution-status receipt, not a performance result.

- Session: `sf6-direct-0909v1`; ticket: `17888991282385029`.
- Launch: `be907d63be5a4736a1fc232fbae9fd6a`; supervisor PID: `2385029`.
- Frozen GPU source: `85e25370779c2b8a6c9aaefa275b1dc8f27d60d0`.
- Source checkout: `/home/choiceoh/stkernel-sf6-direct-0909`.
- Evidence: `/home/choiceoh/glm53-logs/SF6-DIRECT-sf6-direct-0909v1`.
- Observer PID: `2390658`; confirmed exact hold OWNED and READY with 63 targets,
  the frozen source and explicit `sf6_direct=true`; initial errors were empty.

A uses packed-only `t,r,sf6`; B uses raw `t,r`. Compact AR and inline RDMA are
both off in both arms. Both retain AR consumer PDL/MK PDL, SPEC_K=5,
KV_TOKENS=1100000 (actual override 665 required), standard 2K/32K/128K quality and
prefill, and exclusive fixed 3x2048 decode. Four-node 10 GiB memory guard remains.
Exact argv/spec and completion instructions are in `probes/sf6_direct_onepass.md`.

CPU core 71,159 checks plus 50 megakernel regressions and 80 preparation tests
passed before submission. Current main's KV-zero boundary fix is shared by
both arms. Earlier SM121 compile/installed-wrapper CPU receipts remain in the
separate direct-prefill implementation evidence folder.

Per-rank direct ownership finalization, GPU model execution, final quality,
actual memory and timing are pending. There is no canonical analyzer verdict
or final exit receipt yet. No default promotion or merge has been performed.
The old v5 failure is separate and supplies no acceptance evidence for this run.

The existing 10-minute heartbeat was updated to follow this exact ticket and
report completion/failure only. The admitted remote source remains frozen;
this status document is a later local documentation-only addition.

## Interim candidate result, 2026-09-09 05:40 KST

A completed its standard onepass and fixed 3x2048 decode. The individual record
and its 147 memory samples pass their CPU audit: 20.28170 steps/s,
49.30553 ms/step, 61.95479 output tok/s, quality 18/18 and Korean corruption 0.
Prefill TTFT: 2K cold 2.44010 s/warm 0.83828 s, 32K 10.88157 s, 128K 42.22854 s.
Minimum host MemAvailable was srv1/2/3/4: 15.54665/10.24701/11.76080/18.07024 GiB.
These are candidate-only measurements, not an A/B verdict.

The full runtime admission failed: all four ranks missed the required MHC
selftest PASS and capture. T16 reported FP32 post_mix/comb_mix differences
(maximum 1.78814e-7/2.38419e-7), while BF16 residual and layer_input were exact.
The consumer PDL test passed. No valid prepared proof was accepted, so the
observer marked A's runtime FAIL. The checker and the admitted source are
unchanged, and B continues through the original chain.

All four failed preparation receipts nevertheless directly record the expected
SF6 lifecycle: 42 packed layers, 3,604,414,464 packed bytes, one finalization
releasing 4,756,340,736 raw scale bytes, m6 serving, zero fallbacks and actual
KV 665. The release count is not a measured net memory saving versus B.
The preserved intermediate raw record, memory, receipts and logs are at
/tmp/sf6-direct-onepass-0909v1/interim-0540. Candidate numbers and marker fields
are retained in candidate_interim.json with comparison=null. Final canonical
acceptance remains unestablished; no default promotion or merge is warranted.
