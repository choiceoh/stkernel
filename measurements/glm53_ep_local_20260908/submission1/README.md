Normal fleet GPU admission: eplocal0908v1, 2026-09-08 14:20:31 KST.
Preflight passed. At 14:21 KST the session was queued third, behind the AR/MHC
consumer and prefill step-cost jobs, with approximate fleet ETA 17:20 KST.
No GPU cell has run yet. Queue forecasts can change.

Frozen execution source is 98ce2cf07794e9361c2fdbc9b11b6bc7ffaffb07 at
/home/choiceoh/stkernel-ep-local-gpu-0908-1 on the head. Do not modify this copy.
Driver PID 614995, job /tmp/glm53-ep-local-gpu-0908-1.
Read exit.json and capture/completion.json before submitting any retry.
The source was rebased onto main 926239e, and every compiled overlay source
still matches CPU3's hashes. Local logic finished with 6783 checks, 38
megakernel regressions and 120 fleet regressions. The complete local log is
preserved; host tensor prerequisites may skip, so this is separate from the
actual pinned-image CuTe compilation evidence.

The runner pauses the incoming GLM containers under its normal boot holder,
uses a fresh head-only container for each component cell, and restores the
exact original IDs/config/source. It requires 16 GiB available memory and
128 GiB disk reserve on all nodes. The standard fleet supervisor retains its
public recovery responsibility. No production candidate or default is deployed.
Passing these component checks would still leave real TP4 transport, EP
serving quality/capacity/decode, and current-baseline direct TTFT unproven.
