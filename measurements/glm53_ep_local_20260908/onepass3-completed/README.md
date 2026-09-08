# Onepass3: measured B1/A, invalid normal verdict, B2 absent

Session `eplocalonepass0909v3a` (ticket `17888881731434836`) entered GO at 02:33:10 KST on 2026-09-09. B1 finished at 02:44:32 and A at 02:53:21. Normal chain then returned **invalid / UNPROVED, exit 4**, released the ticket, and did not run B2. The original verdict, proof fields and two metric rows are preserved without correction. **No default acceptance or complete-bracket performance verdict follows.**

| Context | B1 TTFT | A TTFT | B1 prompt tok/s | A prompt tok/s | Provisional change |
|---|---:|---:|---:|---:|---:|
| 32K | 10.8306 s | 31.4414 s | 3,004.91 | 1,035.10 | −65.55% |
| 128K | 41.6289 s | 36.6926 s | 3,088.21 | 3,503.67 | +13.45% |

Rates use each request's actual `prompt_tokens / ttft_s`; percentages are descriptive A/B1 ratios, with no B2 drift control. Each long context has one combined request: its repeated warm aggregate is not another observation. B1's first 2K TTFT was 2.4073 s and best warm 0.8494 s; A's three 2K TTFTs were **20.4710, 0.9552 and 5.1506 s**. Reporting only the fastest warm sample hides the third-request delay. First-request compilation/JIT is part of observed latency; its isolated contribution is not measured, and these rows omit `cold_compile`. Canonical prefix caching remained enabled.

Both arms returned quality **9/9**, Korean anomalies **0/5**, and no traffic issues. Per-request decode rates fell from roughly **69–84 to 23–28 output tok/s**. Exact requests, output hashes, counters and decode measurements remain in `onepass.jsonl`; `summary.json` contains derived rates. The unchanged normal judge objective was `decode_steps`, not prefill TTFT.

The normal proof registry does not recognize shared `VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=1`, leaving its proof null even though EP-local proof is true. The normal baseline lookup also requires `knobs == {}`, so the shared nondefault control makes `base:null`. This harness incompatibility explains the preserved `UNPROVED (proof 1/1)` result and abort before B2. Saved live evidence does not rewrite that verdict or make this an accepted run.

All four A containers were captured privately at 02:51:35–36 with stable IDs/start times before and after each log read. Every rank logged skipped-unused profiling at 02:48:08 and real graph capture completion at 02:49:03. During actual onepass requests, all four logged **E72/I2048/top8 T6912 EP-local LAUNCHED and MHC token shards selected at 02:50:52**. Boot-only T8192 markers are separate. Selection/launch is not numerical acceptance; the actual quality results above are retained independently.

Source remained clean `ac2f188392bc76f1194f8eee7be9b0dfbec346f5`, overlay `7c4c8ce88b69`, with the same pinned image and original capacity. Common skip-unused=1 and explicit estimator application off retained real graph capture, graph cap 32, GMU 0.6329, max length 1,048,576, sequences 4, batch 8,192 and the submitted KV controls. `submission.json`, launch/run logs and sanitized A configuration preserve the plan and identity evidence. Raw Docker inspect/Cmd/Env stays private under the remote job's `live-A` and local `/tmp/glm53-onepass3-live-A`; mount identities do not attest subsequently mutable file contents.

Collection only read existing files and normal fleet show/status; no requests, service/source/queue changes, tests, or resubmission occurred. `provenance.json` lists exact sources and commands. Named terminal files were read twice remotely; local live-A copies match their recorded hashes. Gzip has `mtime=0` and decompresses to exact original log bytes. `SHA256SUMS.json` covers every stored file except itself. Fleet status is a later snapshot; the ticket's release and central-controller recovery policy are separate from other work. Existing queued/historical archives remain unchanged.
