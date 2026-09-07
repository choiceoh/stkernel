# Fleet handoff and shared startup controls

The srv2 audit found a boot successor queued at 2026-09-08 05:38:56 KST while
`moedirect0908` then restored production from 05:39:31 to 05:45:19. The successor
started at 05:45:27. The 348-second restore is an observed cost, not a measured
post-change saving. Five startup campaigns spent 2,326 seconds on baseline
health waits and 2,229 seconds on PRIME health waits. Those builds differed;
these measurements do not authorize cross-build baseline reuse.

The implementation supervises the final restore/handoff once per boot job and
adds fixed-build startup campaigns. Two candidates use seven boots instead of
ten; three use nine instead of fifteen. Each candidate retains two boots and
all cache, runtime and quality gates. These are controlled boot-count savings,
not measured GPU turnaround gains.

## CPU validation

- At `24a0ad4`, macOS: 118 tests, passed, complete coverage.
- At `24a0ad4`, Linux srv4: 126 tests, passed, complete coverage.
- Final code `f2c9d36`: 16 Linux follow-up tests, passed, complete coverage;
  eight portable campaign/receipt tests also passed on macOS.
- Final follow-up covers the subsequent runtime image/module binding, serialized
  pinning, signal handling and review corrections. The earlier full suite was
  not relabelled as having run on a later revision.
- No reported gate skipped a test. The Linux integration suite skips on other
  operating systems; the CPU evidence runner treats skipped coverage as
  incomplete, so that cannot substitute for the Linux gate.

`cpu-macos.json`, `cpu-linux.json`, `review-linux.json` are the original machine
receipts. Their referenced detailed logs remained in the local build directory
and `/tmp/fleet-handoff.5FCpUo` on srv4. `validation.json` pins final source hashes.

The Linux integration executes the real shell queue against temporary files and
fake system commands. It verifies two waiters hand off with only the last one
restoring, cancellation before/after offer, receiver failure, failed restore
retaining debt, probe ordering, nested cleanup, and an unchanged pinned runner
when the source checkout advances. It does not call a real GPU, Docker daemon,
SSH host or production endpoint.

```bash
PYTHONPATH=bench:tests python3 bench/cpu_checks.py --suite fleet \
  --test tests/test_startup_campaign.py --test tests/test_startup_cache_receipts.py \
  --out build/handoff-cpu.json
# On Linux, additionally:
PYTHONPATH=bench:tests python3 bench/cpu_checks.py \
  --test tests/test_boot_supervisor_linux.py --out build/handoff-linux.json
```

## Rollout constraints

The srv2 common checkout was clean on `ab` at `6e8d1a9`, which contains unmerged
prefill work. Runtime adoption is prepared on a separate branch based on that
commit, preserving its overlay/profile/launcher tree. The approved production
checkout is configured separately through `fleet/production-repo`. Existing
jobs already queued under the old runner finish with their original controller;
new supervised submissions pin their own control scripts. Production acceptance
and measured live handoff savings remain separate from the CPU receipts here.
