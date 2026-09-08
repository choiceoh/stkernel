The normal fleet run glm53cpuvotemem0908v1 failed before its first memory sample.

Frozen execution source: 6e693f53a118bce9fc0a8b457943c3880a9fb59b.
The private PRIME server booted and its idle observer RPC passed, but the client
rejected /glm53/cpu-vote-memory as an unsupported endpoint. BASE0/CPU/BASE1,
model requests, TTFT and quality measurements never ran. This is not evidence
of a memory reduction or prefill improvement.

The client now uses a dedicated PrivateMemoryAPI subclass, adding only the
memory JSON endpoint and requiring an empty options body. The general observer
still rejects that endpoint. The new test executes the real client POST method
with a mocked HTTP transport, which catches the route-registration omission
missed by the previous whole-client mocks. Local tests: 13 total test outcomes,
including two environment skips (Linux process/torch and FastAPI prerequisites);
no new pinned-image or GPU retry has run.

capture/completion.json reports complete=false, performance_acceptance=false,
and restored_original=true. restore-comparison.json verifies exact original
ID, image, config, host config, mounts and overlays on all four nodes. This
original was the normalizer's 0d5ca6d deployment, not the later #469 merge.
Original recovery completed at 13:25:33 KST. The fleet supervisor subsequently
performed a separate approved-public-main refresh and released the session at
13:34:54 KST with exit code 1. Do not conflate that public refresh with the
payload's exact original restoration.

Files were transferred with source SHA-256 verification. Compressed logs retain
their raw contents. The experiment is closed as failed; no duplicate retry is
queued. Current prefill work proceeds on the independent expert-local branch.
