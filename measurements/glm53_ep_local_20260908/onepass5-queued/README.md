# Onepass5 queued: remove EP decode regression before adoption

The default-promotion edits were withdrawn before commit, merge or deploy following the operator's clarification. EP/local/warmup/zero-weight-micro defaults remain off. Frozen source `e2a54cff881465c2bb7dbbd3f5ec39ca240c7f74` contains the short-only padded-micro wrapper correction; inherited kernel arithmetic and dispatcher geometry are unchanged.

Actual onepass4 graphs have 6/12/18/24 tokens with SPEC_K=5. The fixed fallback used 6/12/18/24 top-k=1 calls. The selected zero-weight lane now admits the missing short-only batch and uses 1/2/3/3 top-k=8 calls through existing eight-row staging. This is a call-count change, not measured speedup evidence.

Session `eplocalonepass0909v5`, ticket `17889057482943885`, runs the canonical B1/A/B2 chain. Only A enables EP, EP-local prefill, compact preparation and zero-weight micro. All arms share the unused-graph-profile skip, pinned image, original KV/capacity settings and exclusive loopback endpoint. The existing onepass fixed-decode options add three 1024-token requests per arm alongside standard 2K/32K/128K requests. The primary objective is direct output `decode_tokens`; per-request prefill tok/s and TTFT remain recorded.

Local checks: six actual-wrapper control-flow/staging/proof tests and three existing logic checks passed; compact/proof/overlay checks passed. Frozen Linux CPU evidence is `passed=true`, `coverage_complete=true`, 31 tests with no skips. The tests model routing/copying on CPU and do not establish device numerics. The historical GPU5 numerical failure and onepass4 decode regression remain unresolved until further evidence addresses them.

The submission receipt establishes an accepted reservation, not GPU execution or completion. `queue-at-collection.json` is a time-specific snapshot. The remote source stays frozen through the run; old archives and original rows are unchanged. No default promotion, merge or live deployment has occurred.
