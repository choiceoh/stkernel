# GLM single mapped rank-cache restore trial

Runtime: `50223132e14a9a59fe07b03f2db7fc53de4579cf`

PRIME is excluded. Timed order: BASE1, FAST1, FAST2, BASE2. Same runtime/profile, warm artifacts, PREFILL_WARMUP=0, Korean onepass at 2K/32K. Only rank-cache prefetch changes. Hash worker times overlap other phases.

| Arm | Health (s) | Node | Prefetch | Restore (s) | Hash work / wait (s) | Copy / discard (s) |
|---|---:|---|---:|---:|---|---|
| RANKMMAP3BASE1 | 225 | srv1 | 0 | 45.402 | 38.112 / 38.115 | 2.892 / 4.363 |
| RANKMMAP3BASE1 | 225 | srv2 | 0 | 44.984 | 41.25 / 41.253 | 2.85 / 0.851 |
| RANKMMAP3BASE1 | 225 | srv3 | 0 | 37.609 | 33.91 / 33.913 | 2.83 / 0.836 |
| RANKMMAP3BASE1 | 225 | srv4 | 0 | 39.084 | 35.507 / 35.511 | 2.582 / 0.959 |
| RANKMMAP3FAST1 | 234 | srv1 | 1 | 39.347 | 37.733 / 30.437 | 2.887 / 5.954 |
| RANKMMAP3FAST1 | 234 | srv2 | 1 | 46.275 | 45.335 / 42.576 | 2.788 / 0.839 |
| RANKMMAP3FAST1 | 234 | srv3 | 1 | 36.853 | 35.918 / 33.09 | 2.8 / 0.895 |
| RANKMMAP3FAST1 | 234 | srv4 | 1 | 37.405 | 36.67 / 33.869 | 2.561 / 0.909 |
| RANKMMAP3FAST2 | 228 | srv1 | 1 | 39.199 | 37.652 / 30.856 | 2.931 / 5.345 |
| RANKMMAP3FAST2 | 228 | srv2 | 1 | 45.64 | 44.732 / 41.959 | 2.804 / 0.809 |
| RANKMMAP3FAST2 | 228 | srv3 | 1 | 38.541 | 37.588 / 34.764 | 2.841 / 0.865 |
| RANKMMAP3FAST2 | 228 | srv4 | 1 | 39.291 | 38.547 / 35.754 | 2.571 / 0.901 |
| RANKMMAP3BASE2 | 220 | srv1 | 0 | 45.974 | 38.544 / 38.548 | 2.883 / 4.511 |
| RANKMMAP3BASE2 | 220 | srv2 | 0 | 43.892 | 40.152 / 40.155 | 2.858 / 0.848 |
| RANKMMAP3BASE2 | 220 | srv3 | 0 | 44.029 | 40.257 / 40.26 | 2.888 / 0.851 |
| RANKMMAP3BASE2 | 220 | srv4 | 0 | 44.3 | 40.822 / 40.825 | 2.558 / 0.884 |

| Arm | Quality | Corrupt | Raw acceptance |
|---|---|---|---|
| RANKMMAP3PRIME | 6/6 | 0/4 | 0.48633879781420764 |
| RANKMMAP3BASE1 | 6/6 | 0/4 | 0.5027818448023426 |
| RANKMMAP3FAST1 | 6/6 | 0/4 | 0.46206896551724136 |
| RANKMMAP3FAST2 | 6/6 | 0/4 | 0.494356005788712 |
| RANKMMAP3BASE2 | 6/6 | 0/4 | 0.4923520923520924 |

Exit: `0`. See report.json for both GPU probes and the adjacent raw SHA-256 inventory for the retained node/response/resource files. Two warm samples per path are not a broad throughput or full-context quality verdict.
