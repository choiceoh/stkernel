# Bound one-shot transport resources to the four-slot protocol

Base: `de8bfff6` (main, after #944 and the CUDA 13.2.1 migration).
This change reduces requested RDMA queue capacity and host completion-tracking
storage. It makes no measured serving-speed, quality or GPU-memory claim.

## Resource change

| Per process, TP4 / two rails | Before | Requested after |
|---|---:|---:|
| Send WR capacity, three QPs | 3,072 | 24 (8 per QP) |
| Receive WR capacity, three QPs | 12 | 0 |
| CQ entries, two CQs | 8,192 | 24 (16 + 8) |
| Completion-count storage | 512 bytes (64 uint64) | 16 bytes (4 unsigned) |

For one rail the CQ request is 24 entries, replacing 4,096. These are requested
queue entries, not measured NIC bytes: providers may round up their allocation.
Boot checks returned CQ/SQ capacities against the required bound. The
[rdma-core QP contract](https://man7.org/linux/man-pages/man3/ibv_create_qp.3.html)
and [NVIDIA CQ documentation](https://networking-docs.nvidia.com/doca/archive/3-5-0/rdma-aware-networks-programming-guide)
describe actual capacities being at least the requested capacity.

## Why the bound includes failures

The GPU cannot publish a sequence more than four ahead of the all-peer ACK.
The proxy emits exactly two RDMA Write WRs per sequence per peer: payload then
flag. A peer's SQ therefore needs at most `2 * RING` entries. No receive WRs
are posted or consumed by these plain writes.

Only the flag normally requests a CQE, but payload WRs can also generate error
or flush completions. Each CQ is sized for both WRs across all four live slots
and every peer on that rail: `2 * RING * peers_on_rail`. Sharing the CQ between
send and receive does not add receive completions because no receive WRs are
posted. A fatal completion still stops the proxy through the existing health
path. Increasing the ring or adding WRs/receive operations would require
revisiting these capacities.

The counter for a sequence reaches three and resets before its ACK is
published. That ACK is required before reusing its slot. Completion counts can
therefore use the same four-slot index as the payload ring. Counts range from
zero through three, so an unsigned counter suffices. The increment/reset,
all-peer ACK rule and single ACK publication per bounded poll pass remain the
same. The hot path keeps the original counter operations.

## Validation

`test_engine_oneshot_proxy.py` compiles the actual production proxy and header
with mock verbs and UBSan. Four rail/inline builds cover every rank, burst
widths 1/2/4, CQ batches 1/16, three cross-QP completion orders and both drained
bursts and continuously refilled windows: **294,912 normal sequences** plus
post/CQ/WC failures. Per-QP ordering is preserved while one QP can finish later
sequences before another finishes an earlier one. Refill cases reuse a freed
slot while older sends remain live. Every ACK is checked against actually
delivered peer completions, and simulated SQ/CQ occupancy may never exceed the
new bound. Error cases enqueue both unsignaled payload and flag completions.

`test_engine_oneshot_setup.py` executes the production preparation/cleanup code
against mock resource providers for all four modes. Every rank's requested
CQ/SQ capacities are checked; provider rounding is accepted and capacities
below the bound fail with full cleanup. Acquisition/release fault injection,
late GID discovery and GPU-use lifetime guards remain covered.

The focused CPU suite passes 49 tests with no failures or skips. All four actual
Torch extension compile/load variants pass in the existing image
`sha256:83080fb01fb9aab3efb34a364d045e9f50ce142885e8a1cc96dde1ca3aa0c052`,
with CUDA hidden, no GPU/RDMA devices exposed, no network, two CPUs and one compile worker.
`cpu.log`, `compile.json` and `toolchain.log` retain the final results, source
hashes and actual nvcc version. The compile report's `cuda` field is the Torch
wheel's build version; `toolchain.log` identifies the CUDA 13.2 compiler.

GPU queue submissions/changes, model boots, RDMA initialization, service
restarts and image builds were not performed. Actual provider allocation,
RDMA/GPU correctness and matched consumer latency/step rate remain unmeasured.
