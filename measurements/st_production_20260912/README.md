# ST GLM-5.3 production cutover — 2026-09-12

The GLM production backend now runs the standalone ST engine on all four GB10 nodes. The head's systemd supervisor started the pinned fleet, passed real generation health checks, and adopted the same containers after supervisor restarts. The former vLLM idle recovery timer and srv1's obsolete TP2 boot service are disabled.

## Deployed configuration

| Setting | Live value |
| --- | --- |
| Engine release | `5734b29fde84` |
| Release directory, all nodes | `/home/choiceoh/st-releases/5734b29fde84` |
| Image tag, all nodes | `st-engine:prod-5734b29fde84` |
| Rank order | srv2, srv1, srv3, srv4 |
| Engine source SHA-256, all ranks | `ae748cff7c85281f200def3a0fa987be83c21bf2777e637c25a91f61e83ab748` |
| Execution | `--production`: served lanes, captured decode, stock MoE/MLA, DFlash2 |
| Capacity configuration | KV 8.73 GiB per rank, 4 request rows, context ceiling 1,048,576 |
| Head API | `http://10.10.10.2:8000/v1` / `http://100.125.220.117:8000/v1` |
| Model | `glm-5.3-flash` |
| Deneb/Wormhole names | `glm-5.3-flash-local`, `glm-5.3-flash-local-low` |
| External API | `https://sensation-glider-unknotted.ngrok-free.dev/v1` |

Wormhole retains its existing model names and upstream address. Its low-thinking route produced ST request IDs `chatcmpl-20` and `chatcmpl-21`; the ST head's served counter advanced from 12 to 14. Its regular thinking route also returned reasoning and final content from ST. The external ngrok service previously forwarded to an unused srv1 loopback port; its systemd drop-in now forwards to srv2:8000.

`deployment.json` records container IDs, per-node image IDs, commands, mounts and rank settings. Image IDs differ between node-local builds; runtime source identities match each other and the local engine tree. Every `runtime-rank*.json` verifies the pinned ABI and absence of the vLLM package.

## Acceptance evidence

All 15 synthetic API cases passed:

| Evidence | Cases | Result |
| --- | ---: | --- |
| `direct-api.json` | 9 | Korean answer, ordinary SSE, thinking SSE, generated tool call, four concurrent HTTP requests, 9,391-token retrieval |
| `wormhole-low.json` | 2 | Korean answer and content SSE through the actual low-thinking application route |
| `wormhole-thinking.json` | 1 | Reasoning plus final answer through the regular application route |
| `public-api.json` | 2 | Korean answer and SSE through public HTTPS |
| `long-context.json` | 1 | 132,041 input tokens; recovered the secret at the beginning, `7349`, in 60.677 seconds |

The long input crosses many 6,912-token prefill chunks. This is a synthetic retrieval check, not a broad long-context quality score. Four simultaneous HTTP clients were checked for complete, correct responses; no throughput improvement over vLLM is claimed. The configured 1M ceiling was retained, but this cutover did not send a 1M-token request.

The checkpoint's tokenizer JSON declares right truncation at 2,048 tokens. The release clears truncation and padding before serving, and the acceptance checks confirm the full prompt reaches the model. `tokenizer-test.log` separately exercises a saved truncation rule in the actual native ST runtime. A 12,006-token checkpoint-tokenizer check also preserved the final marker.

`memory-summary.json` and the four raw memory ledgers report **159/159 successful checkpoints on each rank**. Peak reserved memory was 61.734 GiB and peak workspace was 5.731 GiB per rank, within the 12 GiB workspace ceiling. These are boot qualification peaks, not a sustained production memory profile.

`cpu-tests.log` records the full engine CPU discovery run, including the new fleet ownership and production-default tests. GPU/runtime-dependent tests are skipped locally; the tokenizer regression also passed separately inside the native ST image. The ownership tests execute real shell control flow against an isolated fake fleet and cover foreign locks, acquisition races, preparation failure, healthy adoption and unreachable nodes. The production config test advances the date to 2040 and verifies that experimental overrides are still refused.

## Operation and recovery

On srv2:

- `~/.config/st-glm53.env` pins the release, image, rank directory and release-local metadata; `ST_PRODUCTION=1` selects fixed defaults without experimental expiry dates.
- `~/.config/systemd/user/st-glm53.service.d/release.conf` pins the supervisor's executable to the same release.
- `st-glm53.service` is enabled and active; user linger is enabled. It checks real chat generation every 30 seconds and restarts all four ranks after three consecutive failures, with bounded launch backoff.
- `fleet-idle-recovery.timer` is disabled and inactive. On srv1, the obsolete root `vllm-tp2.service` is also disabled and inactive.
- `supervisor.log` records the initial fleet launch at 06:42 KST, API readiness at 06:44:49, and adoption after supervisor restarts. Container identity was unchanged during adoption. A worker crash was not deliberately injected.

The launcher now acquires the fleet lock atomically, releases its own lock on preparation failure, and refuses to stop a foreign owner. The supervisor waits for other ST jobs or unreachable nodes without consuming its retry budget. A failed image build propagates as a failure. The release directory separates production code from the mutable `~/st-engine` experiment tree.

Before cutover, the old engine tree, image tag and systemd configuration were backed up on every node under:

`/home/choiceoh/st-backups/20260912-f4d7-fb0c3021`

The directory name reflects the first prepared revision; the actual deployed revision is `5734b29fde84`. The head also contains `rollback-to-vllm.sh`, copied here for review. It stops ST and re-enables the existing approved vLLM idle controller, which waits for five idle minutes before restoring the approved checkout. The recovery checkout was confirmed present. Rollback was prepared but not executed. The corrected ngrok upstream remains valid for the restored vLLM head.

## Reproduction

Run `python3 probes/engine_production_check.py --output result.json` from a host that can reach the head. `--quick` checks basic chat and SSE. For the existing authenticated Wormhole route, run on srv4 with `--base http://127.0.0.1:18800 --model glm-5.3-flash-local-low --wormhole-config /home/choiceoh/.wormhole/config.json --quick`; the probe reads the token locally and never writes it to the report. `long-context-check.py` reproduces the larger synthetic input.
