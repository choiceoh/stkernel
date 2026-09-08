# Queued reservation editing — 2026-09-08

CPU evidence only. Linux fixture processes replace Docker, SSH, metrics and
production restore with local commands. No real fleet reservation, GPU boot,
measurement or production restore was performed by this validation.

- Fleet behavior suite: 127 passed, complete coverage, no skipped tests.
- Final real-process supervisor suite: 14 passed (12 initially, plus concurrent
  admission/edit and damaged-record recovery), complete coverage, no skips.
- Final edit unit suite: 7 passed; these also belong to the 127 fleet cases.
- Core gate: 71,087 checks, including 38 megakernel regressions, passed.
- Shell syntax and diff whitespace checks passed.

The full fleet run preceded the final recovery-record hardening; the affected
supervisor and edit suites were rerun after that change. Final controller and
test content identities are in `source-sha256.json`. Report log paths identify
the Linux CPU validation host's retained temporary checkout.

An initial core attempt encountered old macOS AppleDouble transport sidecars
(`._*.py`) in that temporary checkout. Only those identified sidecars were
removed, and the unchanged core gate passed on retry.

Covered behaviors: ticket/age/PID/neighbors preserved, literal argv and working
directory replacement, validation failure retaining the old command, actual GO
during preflight, concurrent editor revision conflict, started/legacy/dead/PID
reuse rejection, immutable structured experiment payloads, probe admission,
one final boot restore, cancellation/handoff recovery, and recovery despite a
damaged pending record. The suite checks correctness, not GPU turnaround time.
