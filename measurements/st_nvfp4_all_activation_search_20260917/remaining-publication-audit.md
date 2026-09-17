# Remaining FC search / C=2 publication boundaries

Status: the follow-up [publication repair](publication-repair.md) implements
both ordering fixes and adds identical-input M8/M16 GPU checks. The text below
preserves the evidence and open questions at the time of the initial audit.

Follow-up review on 2026-09-18 after PR #1133. The static decode patch is
merged at `48a23b02eeef1d842322e2343d300c34ea811dd3`; its required CI check
has passed. This review adds evidence only, with no serving-code changes or
GPU launches.

## 1. Confirmed additional gap: dynamic prefill A/SFA publication

Both served SF6 prefill paths still perform the same unbridged transition:

```
FC1 scale search -> generic global stores of packed A and SFA
    -> resident grid barrier -> publish tasks -> resident grid barrier
    -> TMA loads of packed A and SFA, in the same kernel
```

The short producer is
`moe_dynamic_gated_sf6_q0.py:292-336`; the long producer is
`moe_dynamic_gated_sf6_prefill.py:289-333`. Both use the common
`moe_dynamic_gated_sf6.py` kernel, which returns from
`initialize_route_q0_and_publish` at line 712 and proceeds toward TMA
consumption without a global async-proxy fence. The word-scale load method
in `moe_dynamic_gated_sf6_words.py:138-141` issues the actual A/SFA copies.
There is no intervening CUDA kernel or stream synchronization boundary.

The [PTX proxy-fence contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
requires ordering between generic and async accesses to the same global
objects. The shared-memory and mbarrier-init fences do not supply this edge.
The same paired-operand risk documented in `c2-logical-audit.md` therefore
remains during prefill, independently of the static decode repair.

CPU-only compilation in the pinned serving image confirms the absence in
both intermediate and machine code:

| Dispatch M | Selected class | Activation radius | PTX global proxy fences | SASS global proxy fences |
|---|---|---:|---:|---:|
| 2304 | MoEGatedDynamicKernelSF6Q0Words | 1 | 0 | 0 |
| 32256 | MoEGatedDynamicKernelSF6Prefill | 1 | 0 | 0 |

Evidence: `audit_prefill_publication.py` and
`prefill-publication-native.json`. Both compile for PTX 9.3 / sm_121a. CUDA
remains uninitialized. All seven audited kernel/helper source hashes match
the current workspace. The campaign dispatcher snapshot predates unrelated
cache/guard fixes, but the six relevant selectors, dynamic compiler and
launcher match the current workspace by exact function-source hash.

The earlier serving receipts show why this remains relevant to C=2:
the second concurrent request uses `2304 x 14 + 1699` for its long prompt,
where C=1 and the first C=2 request use `32256 + 1699`. The 2641-token
portfolio prompt can split into `2304 + 337` under concurrent decode.
This changes how often and where the affected producer is executed.
Both prefill paths are affected; it is not a second-request-only defect.
The first C=2 request also had errors, so chunking alone is not a complete
explanation of the observed score loss.

The appropriate repair location is in each consuming kernel immediately
after the routing/packing publication returns, before its TMA loads. The
common SF6 kernel covers both compiled classes. Equivalent raw/generic
kernel owners also warrant a source-wide check; their native binaries are
not covered by these two receipts. Updating pinned parent source hashes
would be necessary for a serving-code repair.

## 2. Open protocol concern: the grid barrier's final arrival

Static v4 (`moe_static_kernel_v4.py:998-1018`) and dynamic gated
(`_moe_dynamic/gated.py:1023-1043`) use this protocol:

```
CTA sync; membar.gl
leader: old_epoch = ld.acquire(epoch)
leader: arrived = atom.relaxed.add(count, 1)
if last: st(count, 0); st.release(epoch, old_epoch + 1)
else: spin on ld.acquire(epoch)
CTA sync
```

The native static and prefill receipts emit `atom.global.add.s32`, whose
omitted semantic qualifier is relaxed. The epoch acquire occurs before
that counter read; the last arriving leader skips the epoch wait.

The unresolved edge is acquisition of *other CTAs' current-phase writes*
by the last leader before it releases the next epoch. A release signal
does not by itself acquire the values observed by an earlier relaxed RMW.
PTX's documented acquire patterns include an acquire RMW or a relaxed
read followed by an acquire fence. Neither is explicit on this branch.
The preceding SC fences must also be considered: their order is not
necessarily the same as counter arrival order, so the latest counter
arrival cannot simply be assumed to have executed the latest SC fence.
The old-epoch acquire and CTA-local barrier do not establish that missing
cross-CTA relationship on their own.

This is a protocol concern requiring resolution, not a reproduced GPU
failure. The prior audit treated generic grid publication as established;
that premise is now explicitly open. A targeted acquire RMW or a
GPU-scope acquire fence on the final-arrival path would express the
intended transfer before epoch release. Its necessity and scope should
be settled at the PTX contract, rather than inferred from a favorable
one-time SASS lowering or from score behavior. Relevant definitions are
the [release/acquire patterns and observation order](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-and-acquire-patterns).

## 3. Lower priority: native C=1/C=2 output equivalence

The existing address and lifetime audits narrow the remaining static
concern: 36,864 SFA coordinates match the actual CuTe layout, row ownership
is unique, and FC2 stage/route-cache lifetime checks passed. The C=2 direct
scatter still needs an identical-input comparison: run the same first
eight rows and routes in the M8 and M16 paths, with controlled extra rows,
then isolate packed A/SFA, FC1 results and FC2 contributions before the
final sum. Previous ss1/as1 comparisons used different width inputs and
do not close that specific boundary. This is an evidence gap, not another
identified scatter implementation defect.

The immediate confirmed repair target is prefill publication. The grid
barrier concern should be resolved separately. Consumer score recovery
and latency after either repair remain unmeasured; no fleet was started
for this review.
