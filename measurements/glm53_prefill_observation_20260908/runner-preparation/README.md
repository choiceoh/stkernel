# Observation runner CPU preparation

The pinned runtime ran 39 CPU tests: 11 observer, 14 runner/request/lifecycle,
6 existing pure-prefill attribution and 8 fleet handoff tests. Exit 0, no skips, no GPU access.
`pinned-cpu.json` records the exact files, command and image. These checks do
not establish live model hook coverage, trace compatibility or performance.

A separate Docker API fixture created two **stopped** containers with runtime
runc, host networking, 4 GiB memory and 2 CPUs. It copied Config and HostConfig
through the same `clone_payload`/`create` helpers. Neither container was started.
All created fixture containers were removed by exact ID in finally.

The first fixture attempt failed an assertion after creating both containers.
Its failure was not attributed; the raw log is retained. The second check
passed all 16 requested Config fields and 63 HostConfig fields; the original
Config, HostConfig, image, ID, state and mounts were unchanged. No top-level
inspect fields differed on the second check. This does not establish the cause
of the first failed assertion or validate a live model boot.

The fixture command used the pinned image and the exact static base64 launcher
shape, with `--host 0.0.0.0 --port 8000 --max-model-len 1048576` and profiler
configuration. Only diagnostic endpoint, extension and output binds changed.
No production container was stopped, modified or inspected by the fixture.
