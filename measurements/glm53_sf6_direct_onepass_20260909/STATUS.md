# SF6 direct-prefill onepass: running

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
