# Onepass6 — decode regression remains; remaining B2 stopped by the user

**Do not adopt this candidate by default.** The repaired short EP lane completed onepass B1 and A, but fixed-length output decode fell from **76.899 to 62.489 tok/s (-18.74%)**. The user stopped unnecessary remaining measurement after seeing the gap. All three candidate fixed-length requests had already completed when that instruction arrived; B2 was still loading and was cancelled before producing a measurement row. This directory preserves completed B1/A evidence, not a successful complete B1/A/B2 bracket.

Source `e2a54cff881465c2bb7dbbd3f5ec39ca240c7f74`, session `eplocalonepass0909v6`, ticket `17889071393096545`. GO was **2026-09-09 07:38:59 KST**, B1 completed at 07:49:31, A at 07:57:31. The owner verified the reservation ticket, holder PID and supervisor command, then used the supervisor's SIGTERM handler at **07:59:34.995**. Its normal release completed at **07:59:35.601**, terminal state `cancelled`, payload/overall return code 143, supervisor absent. This was an operator stop, not a model crash. No B2 measurement or final bracket verdict exists. No replacement GPU run was submitted.

## Direct measurements

Each fixed-length request generated exactly 1024 tokens. The pooled output metric is `sum(completion_tokens - 1) / sum(decode_s)` over the three fixed requests: B1 `3069 / 39.909691993`, A `3069 / 49.112354542`. This excludes TTFT from decode and is distinct from engine steps/s.

| Measurement | B1 | A | A change |
|---|---:|---:|---:|
| Fixed decode pooled output tok/s | 76.899 | 62.489 | -18.74% |
| Fixed decode rep 0 tok/s | 84.722 | 65.123 | |
| Fixed decode rep 1 tok/s | 68.555 | 65.399 | |
| Fixed decode rep 2 tok/s | 79.225 | 57.598 | |
| Fixed decode window median steps/s | 19.831 | 16.871 | -14.93% |
| First 2K TTFT | 2.414 s | 2.370 s | |
| Second 2K TTFT | 0.856 s | 0.883 s | |
| Third 2K TTFT | 0.845 s | 0.884 s | |
| 32K prefill input tok/s | 3014.655 | 3396.416 | +12.66% |
| 32K TTFT | 10.796 s | 9.582 s | |
| 128K prefill input tok/s | 3059.468 | 3474.584 | +13.57% |
| 128K TTFT | 42.020 s | 37.000 s | |

Prefill tok/s uses each request's actual `prompt_tokens / ttft_s`; the first 2K request has 2121 tokens, not the summary's 2128. All eight request hashes, actual prompt lengths, requested output bounds and seeds match B1/A. The recorded source, overlay, harness, session, workload and endpoint also match. B1 is marked cold compile; no estimated JIT time is subtracted. Long-context ratios are descriptive single-request results, not an accepted prefill noise-floor verdict or evidence for the campaign's 40% target. Earlier onepass4 used different decode request lengths, so a precise cross-run decode speedup is not claimed.

## Quality and acceptance limits

- Both arms scored 18/18 facts, with no recorded external traffic or measurement evidence issues. A's four active proof markers passed 4/4. All fixed requests returned 1024 tokens.
- B1 failed the existing Korean gate: one of eight responses contained two CJK characters in `Halvorsen博士`. A passed with 0/8 dirty responses. The original B1 failure is retained without changing the scanner or removing its response.
- The original judge is `incomplete`: `baseline is incompatible or failed its gates`. It is not rewritten as a statistical pass or an accepted regression verdict. The observed decode gap is sufficient to reject default adoption and stop the remaining work at the user's direction.
- The prior concentrated6912 GPU numerical failure (normalized peak 0.0406977) remains unresolved. CPU contracts, successful graph execution and these text-quality checks do not replace that numerical gate.
- EP/local/warm/zero defaults remain off. PR478 remains draft; nothing was merged or deployed by this follow-up.

## Runtime and evidence

The existing zero-weight micro lane now executes the actual SPEC_K5 graph shapes: 6/12/18/24 tokens use 1/2/3/3 top-k8 calls. All four ranks completed graphs; the repaired short-only batch path actually ran. The intended reduction in calls did not eliminate the remaining decode slowdown.

Strict B1 and A snapshots succeeded on all four nodes. `B1-A-runtime.json` checks equal capacity, image, mounts and non-EP launch arguments; the only environment differences are the four intended EP controls (compact, compact warmup, EP local, zero-weight micro). `B1-identity/` and `A-identity/` contain independently checked allowlisted summaries, parser sources, logs and manifest hashes. Raw Docker inspect/Env/Cmd inputs remain private. There is no B2 strict snapshot.

- `records/onepass.jsonl` and `records/verdicts.jsonl` are exact remote originals; the records contain B1 and A only. `derived-comparison.json` keeps the formulas, per-request values and matching identity checks separately.
- `boot/` holds full saved B1/A head logs. `fleet/terminal-run.log.gz`, `terminal-capture.json` and `user-stop.json` preserve normal cancellation and source checks. The frozen full-history source remained clean before and after collection.
- `streams/` holds all eight passive stdout/stderr streams and original event bytes with every chunk offset/hash checked. They span B1, A and the interrupted B2 loading window, and remain UNASSIGNED unless separately attributed by container, worker prefix and log boundaries.
- The observer detected release, closed its stream subprocesses, emitted `observer_finished` at 07:59:41.768 and exited. `streams/closure.json` includes the PID-absent check. No process was signalled by the collector.
- Fleet release does not prove public serving recovery. The normal idle controller retains responsibility for recovery; no direct restart or recovery-policy bypass was performed.

The preceding shallow-checkout deployment refusal is preserved in [onepass5-failed](../onepass5-failed/README.md). Retry6 used identical source/workload in a verified full-history checkout. [onepass5-queued](../onepass5-queued/README.md) preserves the frozen Linux CPU31 complete result. `collect-evidence.py` is the read-only terminal collector; `originals.json` and `SHA256SUMS` preserve artifact provenance.
