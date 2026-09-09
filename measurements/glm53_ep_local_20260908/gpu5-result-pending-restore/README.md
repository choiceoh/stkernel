# Full GPU v5: numerical failure, public restore pending

Offline validation received GO at22:17:24 KST and ended at22:20:32 with FAIL.
The 24-specialization remap and balanced4096/6912/8192 MoE fixtures passed.
Concentrated6912 failed after input/routes/scales changed at the same
addresses: one row exceeded the fixed numerical limits. Maximum relative L2
was0.0118141882, normalized peak0.0406976752; stock controls passed. Initial
candidate and its nondefault-stream replay had passed. Runtime identity was
successfully rechecked after the failure. The tolerance is unchanged.

The suite stopped at this failure. Four remaining MoE fixtures and all eight
sanitizer cells were not run. This is not full GPU acceptance; successful
balanced timings do not establish TTFT or the incremental value of CPU16.
No silent retry or default promotion was performed.

The offline wrapper restored the exact four incoming stopped records:
before==stopped==restored, including configuration/source/mount/running state.
The normal fleet then started its separate public-serving restore and health
wait. At this snapshot the fleet job had no exit.json and retained its holder.
Do not conflate the wrapper's restored_original=true with completed fleet
public restoration. Final exit/release/service evidence must be collected
when that phase finishes. This directory is intentionally named pending-restore.

Every closed capture file was reread remotely and verified after transfer.
Logs use deterministic gzip; archive-manifest.json records original and stored
hashes. Frozen source is .../stkernel-ep-local-gpu-0908-5b atd53fd44f; CPU16's
original passing compiler receipt and all source/runtime checks remain in the
source and gpu5-queued archive. No frozen source or original result was edited.
