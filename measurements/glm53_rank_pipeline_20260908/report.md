# Rank checkpoint pipeline startup

Measured source: `82a97dbf5d05286ca0b9670d38463ad84953988d`. PRIME excluded; B/A/A/B. CPU readiness vote fixed at 0; loopback API; background prefill bench off; required model/MM/graph warmup retained.

| Arm | Health s | Head model s | Slowest rank restore s |
|---|---:|---:|---:|
| RANKPIPEPRIME | 407 | 265.0 | None |
| RANKPIPEBASE1 | 216 | 80.0 | 50.892529008036945 |
| RANKPIPEFAST1 | 238 | 97.3 | 66.42528636398492 |
| RANKPIPEFAST2 | 234 | 94.6 | 65.55840246396838 |
| RANKPIPEBASE2 | 219 | 80.0 | 50.911419017997105 |

Serial mapped_hash_s includes page faults. Reader and DMA timers overlap; they must not be summed as sequential wall time.

| Arm | Node | Restore s | Mapped hash s | Read s | Hash s | Host copy s | Copy wait s | Vote s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RANKPIPEBASE1 | srv1 | 50.893 | 43.418 | 0.000 | 0.000 | 1.422 | 1.260 | 1.162 |
| RANKPIPEBASE1 | srv2 | 44.560 | 40.941 | 0.000 | 0.000 | 1.291 | 1.254 | 1.919 |
| RANKPIPEBASE1 | srv3 | 39.441 | 35.526 | 0.000 | 0.000 | 1.346 | 1.349 | 2.300 |
| RANKPIPEBASE1 | srv4 | 39.315 | 35.811 | 0.000 | 0.000 | 1.099 | 1.248 | 1.775 |
| RANKPIPEFAST1 | srv1 | 65.655 | 0.000 | 39.112 | 22.258 | 0.000 | 0.063 | 1.222 |
| RANKPIPEFAST1 | srv2 | 66.425 | 0.000 | 43.064 | 22.102 | 0.000 | 0.056 | 1.313 |
| RANKPIPEFAST1 | srv3 | 53.963 | 0.000 | 30.536 | 22.089 | 0.000 | 0.069 | 1.599 |
| RANKPIPEFAST1 | srv4 | 54.324 | 0.000 | 30.756 | 22.307 | 0.000 | 0.070 | 1.302 |
| RANKPIPEFAST2 | srv1 | 65.558 | 0.000 | 39.014 | 22.192 | 0.000 | 0.062 | 1.322 |
| RANKPIPEFAST2 | srv2 | 64.756 | 0.000 | 41.426 | 22.058 | 0.000 | 0.056 | 1.153 |
| RANKPIPEFAST2 | srv3 | 56.314 | 0.000 | 32.613 | 22.315 | 0.000 | 0.065 | 1.443 |
| RANKPIPEFAST2 | srv4 | 54.902 | 0.000 | 31.444 | 22.159 | 0.000 | 0.072 | 1.946 |
| RANKPIPEBASE2 | srv1 | 50.911 | 43.517 | 0.000 | 0.000 | 1.423 | 1.261 | 1.409 |
| RANKPIPEBASE2 | srv2 | 44.749 | 41.073 | 0.000 | 0.000 | 1.297 | 1.253 | 1.430 |
| RANKPIPEBASE2 | srv3 | 45.033 | 41.239 | 0.000 | 0.000 | 1.366 | 1.255 | 1.529 |
| RANKPIPEBASE2 | srv4 | 42.455 | 38.964 | 0.000 | 0.000 | 1.065 | 1.247 | 1.190 |

Comparison: `{"BASE": {"health_wall_s": {"samples": [216, 219], "mean": 217.5}, "head_load_model_s": {"samples": [80.0, 80.0], "mean": 80.0}, "slowest_rank_restore_s": {"samples": [50.892529008036945, 50.911419017997105], "mean": 50.901974013017025}}, "FAST": {"health_wall_s": {"samples": [238, 234], "mean": 236}, "head_load_model_s": {"samples": [97.3, 94.6], "mean": 95.94999999999999}, "slowest_rank_restore_s": {"samples": [66.42528636398492, 65.55840246396838], "mean": 65.99184441397665}}}`

Verification: `True`
