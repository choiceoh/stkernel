# ST boot preparation and queue lease readiness

> 살아 있는 참조 — **ST 부팅 준비 랑데부와 큐 리스 경로가 지금 어떻게 동작하는지. 부팅 경로가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

Cold and partially warm pack caches can leave TP ranks several minutes apart.
On 2026-09-12, both `B-boot-failed-cold-packs` and
`B-boot-failed-warm-packs` under `st-prefill-3000-20260912-batch6`
failed at NCCL sequence 14: `RuntimeMemory.checkpoint("loaded")` queued a
120-second all-reduce while a peer was still generating weight packs.
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200` did not extend that collective's timeout.

`Comm.init` now connects a separate Gloo group before preparation starts.
`build` meets there before the arena-admission vote and after weight preparation.
The group has a 30-minute timeout, reports the boot phase and rank on failure,
checks phase agreement, and is destroyed after the final rendezvous. Both serving
groups retain their existing timeouts. A barrier timeout override on the serving
Gloo group alone is insufficient: a follower waiting for rank 0 still uses the
group's receive timeout. Establishing the group before loading also lets a lost
peer connection fail without waiting for a new group to form.

The pinned queue runner includes `launchers/lib/fleet-lease.sh` and
`engine/base/fleet_lease.py`. Occupancy still refuses on unreadable evidence;
the files allow it to read a free lease or actually request a holder's yield.
The helper preserves argument boundaries locally and over SSH. Its heartbeat
detaches its standard streams so capturing the PID cannot stall probe startup.

Validation: `tests/test_engine_boot_readiness.py` uses actual CPU Gloo processes
with a one-second serving control timeout and two-second rank skew, in both
directions, plus missing/dead peers and phase mismatch. The shell regression
suite exercises a newly pinned runner, local and simulated SSH execution,
multiword/metacharacter arguments, yield recording, and heartbeat PID capture.
These are startup/control-path checks; full GPU cold-boot and consumer speed or
quality acceptance remain separate evidence.

Deployment requires the new engine source for the next boot and a newly pinned
controller for new reservations. Already running supervisors retain their
immutable snapshots; a checkout update or `resume` does not upgrade them. Do not
rewrite an admitted runner directory in place or bypass its source approval.
