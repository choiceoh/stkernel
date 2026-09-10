# EP76 submission2: refused before GPU hold

Session `epdecode76onepass0910v2`, source `29daef8b95f3dd098ba24b2fae61b322d70ecb38`, was rejected during normal fleet preparation with `accepted=false`, `state=startup-failed`, return code3. The original log explicitly says `PREPARE REFUSED (no GPU hold)` because relevant FP8-dense startup memory-logging changes from main `8855a7c4` were absent. `disposition=started` and the recorded PID describe the startup process; they do not mean the GPU experiment began.

No GPU performance, quality or numerical result is recorded here. This submission failure is not replaced by a later v3 success or failure. The two original logs, exact submission descriptor and submission tool are preserved byte-for-byte, with hashes. The file named `summary.json` is a derived description, not a canonical onepass result.

Source `3fab1ce81a79936b3b76fa4d87b447d9f55bacba` is the subsequent retry preparation after the main merge and outer CPU register-layout receipt validation. Read-only git object comparisons confirm identical native kernel and PREP bytes across the two revisions. Fresh CPU4 source admission is still required; the earlier CPU3 pass is not relabeled as CPU4 or GPU evidence.

Collection was local only. This archive does not contain or fabricate a fleet ticket, GPU holder, serving boot, canary, onepass record or public recovery proof.
