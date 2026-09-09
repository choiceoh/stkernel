# Canonical onepass retry with shared graph-profile control

Session `eplocalonepass0909v3a` was accepted into the normal fleet queue on 2026-09-09 at 02:22 KST, ticket `17888881731434836`. This is queued evidence, not GO or measurement completion. Run B1/A/B2 again; do not splice the previous B1 into a new pair.

Frozen source and runner remain `/home/choiceoh/stkernel-ep-onepass-0909-2` at `ac2f188392bc76f1194f8eee7be9b0dfbec346f5`. The new job is `/tmp/glm53-ep-onepass-0909-3a`. No kernel or canonical measurement code changed. The submission JSON records the full argument vector, and its output records the immutable launch/run log paths.

The only effective change versus onepass2 is common `VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=1`; already-effective `CG_MEM_PROFILE=0` is now explicit. The prior B1 four-node inspect snapshots confirm estimator application was 0 and skip-unused was 0. The existing skip gate only suppresses the discarded estimate for MRv2 / glm5_next with estimator application disabled. Model/MM profiling and actual graph capture/warmup remain. GMU 0.6329, graph mode/capture set, KV target/blocks, max length/sequences/batch and image are identical across all three arms and retained from onepass2. This is a candidate workaround for the observed startup crash, not a proven root-cause fix.

The original submission named v3 had an argument-position error: a note occupied the estimate-minutes argument. Preflight passed but queue registration failed before GPU admission. Its original plan/output are preserved here; normal fleet cancel withdrew that dead owned entry. v3a corrects the arguments and was accepted. There was no extra GPU workload or queue priority override.

Canonical onepass uses the standard prefix-cache and first-postboot/JIT semantics. Each 32K/128K combined request supplies one observation, even when the summary repeats it as cold and warm. No optional memory/fresh wrapper, custom GPU payload, after-hook or separate benchmark is added. The candidate numerical acceptance and 40% full-prefill goal remain unresolved.

Collection: local plans/output copied byte-for-byte; fleet show/status read via SSH from the frozen source; exact accepted launch log copied by SCP. SHA256SUMS covers every retained file except itself. Queue status is only the capture-time snapshot.
