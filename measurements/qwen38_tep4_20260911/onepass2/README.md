# Qwen3.8-Flash-Next TEP=4 — first bracket (2026-09-11 16:54 KST, boot 9, `QWEN38-TEP4-SSD2`)

`python3 bench/onepass.py --name QWEN38-TEP4-SSD2` on srv2 against the stack of
commit `62c387f0` (profile defaults of boot 8: `PLE_SSD=1`, `KV_CACHE_MEMORY` 16 GiB,
`MAX_NUM_BATCHED=4096`, `GPU_MEM=0.55`, fastsafetensors, b12x guarded EP kernel,
one shared b12x workspace, PIECEWISE graphs, `SPEC_TOKENS=0`). Image `qwen38-fi618:local`.

- `onepass-2.log` — the printed report
- `record.jsonl` — the JSONL record onepass appended (`/home/choiceoh/q38-logs/bracket-onepass.jsonl` on srv2)

| ctx | tok | cold prefill tok/s | warm prefill tok/s | cold TTFT | warm TTFT | quality |
|---|---|---|---|---|---|---|
| 2,000 | 1,640 | 2,053 | 4,977 | 0.80 s | 0.33 s | o o o |
| 32,000 | 23,021 | 2,538 | (1 req) | 9.07 s | – | o o o |
| 128,000 | 91,271 | 2,713 | (1 req) | 33.65 s | – | o o o |

Decode: **17.9 step/s median** (n=101 one-second windows inside the answers; 2K n=31,
32K n=26, 128K n=44 — all 17.9), no speculation (raw acc 0.0 %), so 17.9 tok/s single
stream, flat across context. Quality 9/9, Korean corruption 0/5 responses (0 per
million chars in every class). 162 s total. Zero `NV_ERR_NO_MEMORY` lines on srv4
during the smoke requests before this run (the first boot without them).
