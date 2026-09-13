# Boot preparation uses the waiter's queue lease

`st-kda-amortize0913v2` was acknowledged, then paused before GO because its
receipt environment did not include `ST_LEASE_OWNER`. The boot supervisor added
that field to its waiter. Comparing environment key names showed it was the
only new non-bookkeeping field; the source and image were unchanged. No GPU
payload started.

Bind the boot lease owner and canonical `FLEET_LEASE_PATH` before preparation.
For an existing ticket, reconstruct those same two fields when reading the
supervisor's initial `/proc` environment. Official `edit` can then prepare a
fresh signed receipt, and `resume` retains the same ticket, age, PID, and source.
Environment authentication and actual lease admission remain enforced.

Validation: 82 relevant tests passed on srv2's CPU, including the actual shell
entrypoint before preparation, a legacy signed receipt that initially rejects
the waiter's environment, and edit/resume followed by successful validation in
that existing waiter's environment. No GPU or model boot is part of this check.

```sh
CUDA_VISIBLE_DEVICES= nice -n 19 python3 -m unittest \
  tests.test_fleet_onepass_integration tests.test_fleet_pause \
  tests.test_fleet_pending tests.test_fleet_prepare
```

Raw successful output: `cpu.log`.

CI follow-through: refreshed only the three changed fleet test audit pins after
reviewing their unchanged dependency closure. The full engine check also caught
an existing HTTP test's assumption that two completion choices occupy rows 0 and
1. A completed row may legitimately be reused. The test now checks each actual
admission's seed and retains response/index/token assertions, with explicit one-
and two-row cases. No serving code changed. Five focused audit/HTTP checks pass
on Linux (`cpu-ci-repair.log`).

Environment re-preparation also used to freeze a new main base, even though the
accepted source and target were unchanged. That rejected this waiting kernel
probe after unrelated prefill commits landed. Fresh preparation now retains a
prior authenticated approval only when source/input/executable/image/target and
command/spec identities match; it copies the already resolved source refs, never
fetching or advancing the approval base. CPU work still executes for the new
environment. A changed input gets no inherited approval, and a forged receipt
is rejected. Linux validation now covers 103 tests (`cpu-retain.log`).
