# GLM W4 SHA256 cache key fleet validation

Runtime source: `7132fd15306166f15ce783fe2580dad661f22801`; benchmark source: `7132fd15306166f15ce783fe2580dad661f22801`

PRIME creates startup artifacts and warms compilation. Timed order: BASE1 → FAST1 → FAST2 → BASE2. Timed boots use the same runtime/profile, warm rank and FP8 artifacts, PREFILL_WARMUP=0 and Korean onepass at 2K/32K. FAST enables SHA256 W4 keys; all arms retain FAST_IO=1. PRIME aliases historical MD5 packs before timing.

| Arm | Health (s) | Head model (s) | W4 attach (s) | W4 key / read / copy (s) | Profile (s) | Quality / corrupt | Raw acceptance |
|---|---:|---:|---:|---|---:|---|---:|
| PACKKEYPRIME | 541 | 331.6 | 14.548 | 13.178 / 0.776 / 0.717 | 106.7 | 6/6; 0/4 | 47.92% |
| PACKKEYBASE1 | 236 | 84.4 | 11.995 | 10.282 / 0.89 / 0.76 | 37.1 | 6/6; 0/4 | 51.77% |
| PACKKEYFAST1 | 224 | 77.3 | 3.864 | 2.478 / 0.438 / 0.732 | 37.0 | 6/6; 0/4 | 49.87% |
| PACKKEYFAST2 | 229 | 77.5 | 3.931 | 2.52 / 0.461 / 0.742 | 36.7 | 6/6; 0/4 | 45.65% |
| PACKKEYBASE2 | 232 | 85.3 | 11.985 | 10.327 / 0.809 / 0.772 | 36.2 | 6/6; 0/4 | 49.19% |

Two-sample head comparisons (PRIME excluded):

| Metric | BASE mean [range] | FAST mean [range] | Difference |
|---|---:|---:|---:|
| health_s | 234.000 [232.000, 236.000] | 226.500 [224.000, 229.000] | -7.500 |
| model_s | 84.850 [84.400, 85.300] | 77.400 [77.300, 77.500] | -7.450 |
| mk_attach_s | 11.990 [11.985, 11.995] | 3.897 [3.864, 3.931] | -8.093 |
| key_s | 10.305 [10.282, 10.327] | 2.499 [2.478, 2.520] | -7.806 |
| read_s | 0.850 [0.809, 0.890] | 0.450 [0.438, 0.461] | -0.400 |
| copy_s | 0.766 [0.760, 0.772] | 0.737 [0.732, 0.742] | -0.029 |
| profile_s | 36.650 [36.200, 37.100] | 36.850 [36.700, 37.000] | +0.200 |

All-node warm receipts:

| Arm | Node | Rank cache | FP8 hit/miss/error | W4 fast/legacy hits | SHA hits / MD5 fallbacks / aliases / errors | Copy disarms |
|---|---|---|---|---|---|---|
| PACKKEYPRIME | srv1 | saved 184.238s | 0/244/0 | 254/0 | 0/254/254/0 | [0, 0] |
| PACKKEYPRIME | srv2 | saved 56.812s | 0/244/0 | 254/0 | 0/254/254/0 | [0, 0] |
| PACKKEYPRIME | srv3 | saved 55.305s | 0/244/0 | 254/0 | 0/254/254/0 | [0, 0] |
| PACKKEYPRIME | srv4 | saved 58.639s | 0/244/0 | 254/0 | 0/254/254/0 | [0, 0] |
| PACKKEYBASE1 | srv1 | hit 51.535s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |
| PACKKEYBASE1 | srv2 | hit 48.104s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |
| PACKKEYBASE1 | srv3 | hit 47.542s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |
| PACKKEYBASE1 | srv4 | hit 47.214s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |
| PACKKEYFAST1 | srv1 | hit 52.476s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYFAST1 | srv2 | hit 46.041s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYFAST1 | srv3 | hit 48.204s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYFAST1 | srv4 | hit 47.336s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYFAST2 | srv1 | hit 50.921s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYFAST2 | srv2 | hit 45.444s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYFAST2 | srv3 | hit 48.804s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYFAST2 | srv4 | hit 47.794s | 244/0/0 | 254/0 | 254/0/0/0 | [0, 0] |
| PACKKEYBASE2 | srv1 | hit 51.654s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |
| PACKKEYBASE2 | srv2 | hit 46.645s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |
| PACKKEYBASE2 | srv3 | hit 50.065s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |
| PACKKEYBASE2 | srv4 | hit 48.768s | 244/0/0 | 254/0 | 0/0/0/0 | [0, 0] |

Prompt/output comparison with BASE1 (artifact bytes and generated responses are separate checks):

- PACKKEYPRIME: `{'baseline': 'PACKKEYBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`
- PACKKEYBASE1: `{'baseline': 'PACKKEYBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 4}`
- PACKKEYFAST1: `{'baseline': 'PACKKEYBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`
- PACKKEYFAST2: `{'baseline': 'PACKKEYBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`
- PACKKEYBASE2: `{'baseline': 'PACKKEYBASE1', 'n': 4, 'matching_prompts': 4, 'exact_responses': 0}`

Host samples every 10 seconds; sampled OS minima, not CUDA peak memory:

| Node | Samples | Min available RAM (GiB) | Min free disk (GiB) | Net swap growth (MiB) |
|---|---:|---:|---:|---:|
| srv1 | 177 | 14.18 | 289.48 | 0.0 |
| srv2 | 177 | 7.50 | 872.26 | 0.0 |
| srv3 | 177 | 9.59 | 1704.17 | 0.0 |
| srv4 | 177 | 12.13 | 2052.68 | 0.0 |

Trial exit: `0` (null means ongoing).

See report.json, pack-key-gpu.json and the adjacent raw logs, environment snapshots and exact response files. Host stage times include synchronization. Two warm samples per arm do not establish full-context quality or broad throughput performance.
