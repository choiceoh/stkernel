# Reservation usability — 2026-09-08

CPU evidence only; fixtures replace Docker, SSH, serving and restore commands.
No GPU reservation, benchmark, baseline or production restart was performed.

- Fleet suite: 139 behavioral tests passed, no skips, complete coverage.
- Linux supervisor: 20 real-process cases passed, no skips, complete coverage.
- Core: 71,087 checks including 38 megakernel regressions passed.
- Shell syntax and diff whitespace checks passed.

The retained report includes commands, test counts and source log paths. Content
identities are in source-sha256.json. New cases cover joined/withdrawn/active job
filtering before the global limit, exact argv after GO, completed failure versus
successful payload/failed restore, stale PID/session isolation, corrupted records,
old stdout log discovery, bounded regular-file tailing, cancellation and per-ticket
log separation. An unread client pipe receives no reads until the supervisor
finishes, while 2 MiB of payload output is retained and final recovery completes.

`show` reads local reservation files only. `logs` reads a bounded regular-file
suffix. Process completion is not GPU numerical correctness or a performance
verdict. Older records without exit status remain unknown; old terminal output
is not reconstructed. No GPU turnaround claim is made by these CPU checks.
