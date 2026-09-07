# Local reuse sanitizer with resource evidence, 2026-09-08

The diagnostic remains incomplete and cannot admit serving. Memcheck again
stopped after 40/48 trials, before the 8192/concentrated case, with exit 15.
All 274,432 recorded row-trials per arm passed the original candidate/control
criterion in this run. This does not erase the earlier stock-control failures
or the int8gate2 candidate failure. Racecheck was not reached.

The retained container shows `OOMKilled=false`, cgroup `max/oom/oom_kill=0`,
and peak memory **2,909,007,872 bytes**, below its **17,179,869,184-byte** limit.
The sampler took 656 observations without errors. The container memory limit
did not cause the exit. These readings do not measure all GPU driver allocations
or rule out other host/device allocation limits; nvidia-smi reports memory N/A
on this platform.

The NVIDIA kernel journal records `Out of memory [NV_ERR_NO_MEMORY]` from
`_memdescAllocInternal` at **07:27:37**, two seconds before the container exited.
A broader query also finds the same driver allocation failure at **06:23:58**,
immediately before reusediag1's exit. The earlier narrow query did not match
the driver's wording. `job/kernel-allocation-events.json` preserves both records.
The precise allocation constraint remains unresolved; there is no evidence of a
disk-space failure here.

NVIDIA documents that Compute Sanitizer tracking can consume additional host
and device memory, and that allocation failure may terminate the application
with this generic message. This identifies a possible mechanism, not the exact
allocation that failed in these runs. See the [Compute Sanitizer memory-footprint
documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memory-footprint).
No kernel filter, blocking-launch override or error-reporting suppression is used.

The next diagnostic releases completed case input/output tensors and unused
allocator cache after the existing case-end synchronization and all retained-
output checks, before allocating the next case. It preserves wrapper/weights,
within-case execution, all 48 trials and numerical limits. It also records CUDA
free/total, allocator allocated/reserved and host MemAvailable observations.
This reduces probe resource retention without changing model runtime or claiming
to resolve the driver failure before a measured run.

Source `5952925c9832ea99d8244d7ea4a464ef752bc495` was frozen on all four nodes.
Normal session `moem64reuse20908` received GO at 07:23:22 KST; probe execution was
07:24:34–07:27:40. Exact incoming container/configuration/source recovery finished
before outer exit 1 at **07:30:28 KST**. The API/global-write/shared-race detector
controls passed with their expected exit 99. Raw logs, state, source, prefix-plan
and payload checks, driver records and recovery are preserved here. `analyze.py`
validates them while keeping numerical/serving acceptance false.
