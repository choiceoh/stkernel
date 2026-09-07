# GLM rank-cache prefetch trial

Runtime: `e224b9155f30e2a9e496fb22f3ace00bce6f0659`

PRIME is excluded. Timed order: BASE1, FAST1, FAST2, BASE2. Same runtime/profile, warm artifacts, PREFILL_WARMUP=0, Korean onepass at 2K/32K. Only rank-cache prefetch changes. Hash worker times overlap other phases.

| Arm | Health (s) | Node | Prefetch | Restore (s) | Hash work / wait (s) | Copy / discard (s) |
|---|---:|---|---:|---:|---|---|
| RANKPREFBASE1 | 229 | srv1 | 0 | 49.323 | 42.07 / 42.073 | 2.92 / 4.298 |
| RANKPREFBASE1 | 229 | srv2 | 0 | 43.871 | 40.171 / 40.174 | 2.799 / 0.867 |
| RANKPREFBASE1 | 229 | srv3 | 0 | 43.932 | 40.196 / 40.2 | 2.878 / 0.831 |
| RANKPREFBASE1 | 229 | srv4 | 0 | 43.193 | 39.677 / 39.68 | 2.592 / 0.896 |
| RANKPREFFAST1 | 232 | srv1 | 1 | 32.312 | 53.198 / 22.94 | 2.763 / 6.54 |
| RANKPREFFAST1 | 232 | srv2 | 1 | 55.738 | 101.365 / 50.604 | 2.786 / 2.285 |
| RANKPREFFAST1 | 232 | srv3 | 1 | 42.51 | 74.579 / 37.344 | 2.785 / 2.318 |
| RANKPREFFAST1 | 232 | srv4 | 1 | 39.117 | 69.054 / 33.762 | 2.605 / 2.693 |
| RANKPREFFAST2 | 234 | srv1 | 1 | 32.227 | 53.053 / 23.133 | 2.771 / 6.252 |
| RANKPREFFAST2 | 234 | srv2 | 1 | 55.838 | 101.591 / 50.812 | 2.77 / 2.193 |
| RANKPREFFAST2 | 234 | srv3 | 1 | 43.089 | 76.051 / 37.815 | 2.791 / 2.413 |
| RANKPREFFAST2 | 234 | srv4 | 1 | 39.554 | 69.779 / 34.185 | 2.594 / 2.712 |
| RANKPREFBASE2 | 238 | srv1 | 0 | 49.309 | 41.972 / 41.976 | 2.941 / 4.36 |
| RANKPREFBASE2 | 238 | srv2 | 0 | 43.819 | 40.076 / 40.079 | 2.836 / 0.872 |
| RANKPREFBASE2 | 238 | srv3 | 0 | 46.25 | 42.457 / 42.461 | 2.937 / 0.822 |
| RANKPREFBASE2 | 238 | srv4 | 0 | 45.19 | 41.726 / 41.73 | 2.563 / 0.865 |

| Arm | Quality | Corrupt | Raw acceptance |
|---|---|---|---|
| RANKPREFPRIME | 6/6 | 0/4 | 0.4737967914438503 |
| RANKPREFBASE1 | 6/6 | 0/4 | 0.5344563552833078 |
| RANKPREFFAST1 | 6/6 | 0/4 | 0.5270967741935484 |
| RANKPREFFAST2 | 6/6 | 0/4 | 0.5048458149779735 |
| RANKPREFBASE2 | 6/6 | 0/4 | 0.4986822840409956 |

Exit: `0`. See report.json, gpu.json and raw node/response/resource files. Two warm samples per path are not a broad throughput or full-context quality verdict.
