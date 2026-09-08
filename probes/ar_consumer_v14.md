# AR consumer serving measurement, 2026-09-08

The first matched pair measured **0.637 ms/step lower candidate latency
(1.41%)**. This is one completed boot per mode. The second baseline failed
during a response after host earlyoom terminated the head worker; it cannot
establish repeatability. PR #473 remains draft and the profile default is 0.

| Measurement | A1 candidate | B1 default |
| --- | ---: | ---: |
| Fixed 2048-token pooled step/s | 22.454384 | 22.137807 |
| Fixed pooled ms/step | 44.534732 | 45.171593 |
| Fixed intervals | 73 | 79 |
| Existing decode-window median step/s | 22.813208 | 21.851566 |
| Fixed response median tok/s | 77.373059 | 69.805059 |
| Quality checks | 18/18 | 18/18 |
| Text heuristic flagged responses | 0/8 | 1/8 |
| 2K warm TTFT, seconds | 0.882327 | 0.851397 |
| 32K TTFT, seconds | 10.963622 | 10.807867 |
| 128K TTFT, seconds | 41.855192 | 41.657994 |

Each arm completed three exact 2048-token responses after the same prefill
and quality ladder. The primary rate is total steps divided by total elapsed
time across the fixed-response intervals; it is not the median of per-window
rates. Throughput increased 1.430%, equivalent to the 1.410% latency decrease.
All eight ordered request payload hashes match across A1/B1. Generated text
can differ; the B1 text heuristic found two mixed-CJK occurrences in one
response. Both arms passed all 18 quality checks.

The fixed output rates were 73.549324 / 77.373059 / 79.336058 tok/s for A1,
and 69.421314 / 76.898252 / 69.805059 for B1. These are user-visible output
measurements, not the kernel speedup: whole-onepass speculative acceptance
also differed (48.5% versus 46.7%). Cold compilation was reported for A1.
There is no claim of a prefill improvement or a stable 10.8% decode gain.

Both completed arms used source `f3e82e4317fc6215135839c5a02ebf6b14f8119d`,
overlay `29851eaa8973`, and image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Four-rank before/after source, image, boot and knob checks passed. Every
candidate rank proved actual `T=6 bf16=True vec4=True` capture. All eight
SSE request/output hash pairs matched each arm's recorded requests, and the
exclusive-traffic audit reported no issues. The performance comparison
changes only `VLLM_GLM53_AR_CONSUMER_PDL` at the profile level; normal dynamic
boot memory sizing remains in effect.

## Incomplete B2 and host memory

B2 became ready at 20:59:46 KST. It completed the prefill/quality ladder and
the first fixed response, then failed in fixed response 2 at 1139 generated
tokens. It has no completed record or after-traffic runtime proof. Partial
rates are excluded.

At 21:02:06 KST, earlyoom recorded 6143 MiB available out of 122566 MiB
(5.01%) and sent SIGTERM to host PID 3815363, `VLLM::Worker_TP`. The engine
reported that worker's death ten seconds later. The retained kernel journal
contains no messages for 21:01:30–21:03:00. The later Python formatting error
for `decode_tok_s=None` is a consequence of the interrupted stream, not the
primary cause. The journal proves a host-memory termination; it does not
identify why that boot used more RAM or attribute the pressure to the kernel.

During the final part of A1, available memory was about 8.0 GiB; B1 ended
around 6.6 GiB; B2 reached the earlyoom threshold. Memory returned to about
95 GiB after the stopped serving. The fleet payload and supervisor exited
with code 1, then handed off normally to `attr5-0908` at 21:03:09. No other
holder was interrupted and no memory protection was disabled.

Only B2 needs a retry. `--baseline-only` retains the complete workload and
all admission/correctness/runtime gates while avoiding another A1/B1 boot.
The retry adds read-only host-memory receipts to diagnose any recurrence.
Matching source, profile, image and ordered requests must be checked before
combining the new baseline with this pair.

## Retained evidence

- srv2: `/home/choiceoh/glm53-logs/ARCONSUMER-arconsumer0908v14/`.
- Local: `runs/arconsumer0908/v14/`, including A1/B1 independently recomputed
  summaries, raw records, SSE channels, all-rank proof, boot logs, earlyoom
  and kernel journals, host sysstat/container events, and fleet exit state.
- GPU correctness: `ARCONSUMER-arconsumer0908v13/gpu/`, all 15 stages passed;
  copied into v14 after verifying the complete source-bound evidence.
- Deployment CPU release gate: `runs/arconsumer0908/v14/cpu/`, passed before
  GPU reservation, including approved-main recovery preparation.

Recompute each completed arm with
`python3 runs/arconsumer0908/report_serving.py runs/arconsumer0908/v14 NAME`.
The ignored local reports supplement the retained remote raw evidence.
