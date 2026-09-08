# Fleet request workflow — 2026-09-08

Implemented detached raw runs, retry of saved experiments, ticket history/logs,
classification explanations, and preparation before queue admission. Preparation
binds source inputs, runs an optional bounded CPU gate once, and rechecks cheap
inputs before GO. Edits and failed preparation use the same revision/ownership
boundary, so cleanup cannot discard a replacement reservation.

`validation.json` combines the passing fleet 192, core 71,087 and final Linux
supervisor 25 checks. `cpu-initial.json` retains the initial Linux fixture failure;
`cpu-linux-final.json` records its correction and passing rerun. Remote refresh no
longer duplicates the initial preparation at queue startup. All validation is CPU
only, with no serving restart or GPU reservation for validation.

`early_failures.json` records the actual AR v6/v8 and EP binding v2 failure causes.
`ancestry-replay.json` replays the literal guards against the actual historical
checkouts, without fetching or modifying them: both fail before admission in
36.4/61.3 ms for the local guard checks. These are not measurements of end-to-end
turnaround improvement or total preparation including network fetch.

The stopped-container failure is fixed in followup PR #484, merged into the
unmerged EP experiment PR #478. `ep-stopped-cpu.json` records 39 passing lifecycle,
binding/local evidence, sanitizer and offline tests. It accepts uniformly stopped
persistent containers and restores their original identities/running states
without starting serving. The existing frozen queue payload is unchanged; the
next source-bound EP run needs fresh CPU evidence for the fixed source.
