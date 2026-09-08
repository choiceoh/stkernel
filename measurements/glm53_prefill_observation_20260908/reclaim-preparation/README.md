# CPU preparation for private host-memory reclamation

The second observation boot reached readiness but could not start PRIME under
the unchanged 12 GiB guard. This opt-in experiment records the pinned allocator's
active/total bytes and process/host memory before and after returning unused
host allocator pages. Model capacity, serving defaults and GPU allocator caches
are preserved. It does not attribute the earlier shortage to a leak or predict
the amount recoverable.

34 tests pass without skips in the pinned image on srv1, with runc, no network or
GPU, 4 GiB memory and two CPUs: 11 observer, 16 runner and seven host-reclaim tests.
The latter cover missing-API/capture refusal, synchronization/reclaim ordering,
no CUDA initialization in the API process, actual CPU tensor pointer/content
preservation after glibc trim, all-rank/source/counter validation, middleware
partial failure, and refusal before PRIME if reclaim evidence is invalid.
The real pinned image exports the private `torch._C._host_emptyCache` callable.
This is CPU validation, not a GPU pinned-tensor lifetime or serving-quality proof.

The first invocation incorrectly selected tests as package modules and failed
with three import errors before tests ran. It is retained as
`initial-command-error.*`. Explicit unittest discovery passed 33 tests; a
middleware failure-order test and analyzer receipt propagation were then added,
and the final 34-test run passed. Each source manifest identifies its exact files;
only `pinned-cpu.*` applies to the final source.

The next normal fleet submission must freeze this source on all four hosts and
use `--reclaim-host-memory`. It records before/after evidence, then attempts the
normal guarded PRIME and baseline/profile/routes collection. A returned-page
count or passing CPU test is not permission to weaken the memory guard. Failed
reclaim or insufficient headroom ends the experiment through exact-original
recovery; any partial result stays incomplete. TTFT after reclamation is labelled
in the completion and attribution artifacts and earns no speedup claim by itself.
