Failed before the first onepass request. Normal GO 01:34:31 KST; B1 boot completed 01:42:32; optional wrapper minimum12GiB refused head9.816GiB and worker.3 10.745GiB. No OOM occurred in this measurement: the client was never spawned. No tok/s or TTFT was measured. No candidate boot ran.

Outer supervisor finished with payload1/recovery0/return1 and release01:46:56. Its legacy recovery deferred to central idle recovery; recovery0 is not proof of restored public health. The source/image/KV capacity was not changed to lower memory use. Raw private Cmd/Env snapshots remain in the original job to avoid publishing credentials.

Next attempt uses the new approved canonical fleet chain/onepass policy, without the custom onepass_memory/fresh after-hook wrapper. The12GiB extra reserve was a runner choice; onepass_memory itself defaults10GiB, also greater than the failed head snapshot. Both facts are retained rather than claiming a solved allocator problem.
