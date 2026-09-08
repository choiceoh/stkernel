# Reduced-KV v5: failed, no speed comparison

The exact reservation `decode-next-0909v5` / `17888900831650401` finished
with payload and supervisor return code 1. It ran on 2026-09-09 from
03:32:04 to 03:41:07 KST (543.1 seconds of payload execution). The run-log
tail reports candidate A onepass exit 3 and retained failure logs at
`/home/choiceoh/glm53-logs/fail-decode-next-0909v5A-034105`.

The passive observer acquired the correct reservation and announced source
`932b3fc49e4482519668efb92363ca6619863597` with 61 targets. It then recorded
`reservation terminated: failed`. The output directory listing contains A's
four-rank preparation artifacts and memory receipt, but no onepass record
or B preparation artifacts. There is no measured step, throughput, quality
or prefill comparison to report.

The original artifacts remain on srv2. This status preserves facts already
read from the official scheduler and log tail. Original memory/boot-log
contents have not been retrieved for diagnosis; actual KV allocation,
per-node memory and MHC/transport/SF6 mode verdicts remain unverified for
this attempt. No analyzer verdict is claimed without its required inputs.
No guard, GPU queue order, source code, default or serving process changed.
The idle controller owns recovery.

SF6 direct-prefill implementation in PR #501 is separate and was not part
of this v5 source. Its CPU proof does not supply the missing v5 GPU result.
