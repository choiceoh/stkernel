# Nullable Docker HostConfig comparison fix

The first normal observation capture failed before stopping the originals:
Docker changed `OomKillDisable: null` into `false` while creating every stopped
clone. Other requested Config/HostConfig fields agreed. The full payload remains
unchanged; only comparison/identity treats these two representations alike.
Explicit `true`, other resource fields (including `MemorySwappiness`), GPU
requests and bind changes still fail. Raw HostConfig hashes and raw OOM flag
representations are retained alongside the comparison identity.

Docker's [resource constraints documentation](https://docs.docker.com/engine/containers/resource_constraints/)
identifies enabled OOM killing as the default; disabling it is an explicit option.
The inspected head is Docker29.2.1/API1.53/cgroup2 and reports
`OomKillDisableSupported=false`. No OOM configuration is changed by this fix.

Sixteen runner/lifecycle/request tests pass in the pinned CPU-only image, without
skips (runc, network none, 4 GiB memory, 2 CPUs). The two new cases distinguish
null/false from true and reject resource/bind/GPU/policy changes. The inherited
main change in `bench/cpu_contracts.py` is an audit-hash constant; observer and
model kernels are unchanged. Prior 39-test evidence remains historical.

`tests/check_glm53_clone_config.py` also passed against each of the four running
originals. It creates a clone but never starts it, compares all 16 Config and
63 HostConfig fields, checks original identity, and deletes the exact clone ID.
All four report no semantic differences, originals unchanged and clones removed.
The originals' Mounts arrays can reorder between inspect calls, so identity sorts
by destination; it preserves every mount entry. The current diagnosis isolates
that order-sensitive false mismatch, but earlier fixture failures without saved
before/after mounts remain unattributed.

No direct TTFT, quality or model-routing observation was obtained by these checks.
