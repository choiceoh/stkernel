# onepass14: A0 observed, comparison cancelled

Frozen source `c3696a76c6674ad38e01967fa587b8d31a6d121e`, session `eplocalonepass0909v14`, ticket `17889166573854513`, supervisor `3854513`.

A0 completed the unchanged canonical onepass workload: fixed decode 54.4293 / 57.5863 / 64.2312 tok/s, quality 18/18, Korean contamination 0/8 and knob proof 4/4. Full original prefill, TTFT, decode intervals and eight request records remain in `job/onepass.jsonl`; the pre-cancel bytes match it exactly. There is no matched baseline. The canonical verdict is incomplete / no baseline on this build. These observations do not establish a measured regression percentage or performance/default-adoption acceptance.

The owner stopped the remaining arms after the A0 result. B1 had started booting but produced no measurement record; A/B2/B3 did not run. Cancellation is pinned to the exact supervisor/session/ticket and original record hash. Payload and supervisor returned 143; the owned holder was absent at terminal capture. Recovery was deferred to the normal idle controller; this archive does not claim public restoration completed.

All four A0 containers were strictly captured before traffic finished. Original image, config, full mounts, start/creation identity and frozen mounted bytes were rechecked locally. Raw inspect/Env/Cmd remains in the private `/tmp` capture and is not copied here; the allowlisted identity records their hashes. The original observer chunk hashes and offsets were checked through closure. Streams include explicitly unassigned and later B1-boot bytes, which must not be counted as A0 activity.

The mandatory startup self-test returned successfully on every worker by inference from its exact fail-closed source, enabled flags, and the same worker's subsequent model-load and graph-completion markers. No full PASS JSON was published by the INFO-level logger. Individual numerical values, input-hash continuity, and the effective runtime logger level therefore remain unobserved. The original inference receipt contains an earlier, unapplied WARNING-level logging proposal; it is historical evidence only. A later source change uses explicit flushed stdout for future positive receipts and was not applied to this frozen run.

Logs, streams and frozen Python source use deterministic gzip without content normalization. `originals.json` preserves original/stored byte counts and SHA256; `SHA256SUMS` covers the archive. Collection performed only read-only remote observations and fresh local archive writes, with no GPU/test/queue/deployment action.
