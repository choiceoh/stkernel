# Full-row INT8 diagnostic, 2026-09-08

INT8 eliminated the candidate and stock-control threshold failures in this fixed 72-trial TP4 diagnostic. This is evidence to proceed to the full acceptance gate, not serving or speed acceptance. Both INT8 and M64 remain default-off.

| Rows / distribution | Row-trials per arm | FP8 candidate / control failures | INT8 candidate / control failures |
| --- | ---: | ---: | ---: |
| 4096 / skew | 98,304 | 10 / 9 | 0 / 0 |
| 6144 / balanced | 147,456 | 1 / 0 | 0 / 0 |
| 8192 / balanced | 196,608 | 83 / 8 | 0 / 0 |

Each case uses three seeds and eight alternating trials. All 442,368 row-trials per arm are evaluated across all four ranks, with the unchanged per-row L2 0.02 / peak 0.04 / 3x same-row repeat limits. Actual unquantized partials also have zero candidate and control failures. Repeated row-trials are not independent samples. FP8 failure counts should not be interpreted as a trend relative to prior differently instrumented runs.

The 1,152 all-row packet checks have zero differing bytes, including scales and padding. The first trial of every seed uses an independent CPU recipe (144 checks); all others use the same tensor recipe on GPU. The actual INT8 reduction outputs match a native FP32 reduction of independently decoded packets bit for bit after BF16 storage. All four arms preserve their source and gathered input. A further 128 CPU-reference codec cases cover ties, signed bytes, extreme values, zero blocks and odd-row padding; eight short controls prove unchanged BF16 behavior below 4096.

For the candidate, the median of per-rank/per-trial median quantization L2 errors is about 2.67% with FP8 and 1.20% with INT8 in all three cases, relative to each arm's own unquantized FP32 sum. This quantization comparison is distinct from the candidate-versus-stock limits above. Full row maxima are retained in `summary.json`.

Frozen source: `5c80aef92b7b562c0e2577ae17d9a34936f43605`. Normal fleet session `moem64int810908` received GO at 04:59:36 KST. The diagnostic ran 05:00:55–05:02:32; the outer worker exited 0 at 05:05:10 after exact incoming recovery. Container IDs, images, configuration hashes, mounts, overlay hashes, manifest and port match the incoming snapshot on all four nodes. This verifies restoration of the incoming public configuration, not promotion of the candidate.

Reproduce the summary with `python3 measurements/glm53_moe_m64_20260908/int8diag1/analyze.py`. Frozen source, all four rank logs, CPU checks, compile logs, request and lifecycle evidence are included. Log bytes are losslessly gzip-compressed. No full-model quality or TTFT is measured here. Changed-input reuse, capture, sanitizers and the full serving bracket remain separate required work.
