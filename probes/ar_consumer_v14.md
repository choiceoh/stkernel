# AR consumer serving measurement, 2026-09-08

The completed A1/B1/B2 comparison measured **0.544 ms/step lower candidate
latency (1.21%)** against the two pooled baselines. The candidate took
44.535 ms/step; B1 and B2 took 45.172 and 44.988 ms/step. The gain remains
positive against each baseline (1.41% and 1.01%). This covers one candidate
boot and two baseline boots; it does not establish long-run production
performance. After reviewing these results, the operator requested adoption
and an end to further measurements. `VLLM_GLM53_AR_CONSUMER_PDL=1` is now
the profile default; the measured mode remains unchanged. Setting it to `0`
selects the previous path, and the comparison scripts explicitly retain
that baseline. The follow-up measurement automation is paused.

B2 here means the completed `arconsumer0908v17B2` retry. The interrupted
v14 B2 attempt and the retries without serving records are excluded.

| Measurement | A1 candidate | B1 default | B2 default retry |
| --- | ---: | ---: | ---: |
| Fixed 2048-token pooled step/s | 22.454384 | 22.137807 | 22.228061 |
| Fixed pooled ms/step | 44.534732 | 45.171593 | 44.988180 |
| Fixed intervals | 73 | 79 | 80 |
| Existing decode-window median step/s | 22.813208 | 21.851566 | 21.856178 |
| Fixed response median tok/s | 77.373059 | 69.805059 | 73.153069 |
| Quality checks | 18/18 | 18/18 | 18/18 |
| Text heuristic flagged responses | 0/8 | 1/8 | 1/8 |
| 2K cold TTFT, seconds | 2.367944 | 1.996371 | 2.366920 |
| 2K warm TTFT, seconds | 0.882327 | 0.851397 | 0.852495 |
| 32K TTFT, seconds | 10.963622 | 10.807867 | 10.840874 |
| 128K TTFT, seconds | 41.855192 | 41.657994 | 41.746885 |

Each arm completed three exact 2048-token responses after the same prefill
and quality ladder. The primary rate is total steps divided by total elapsed
time across the fixed-response intervals; it is not the median of per-window
rates. Pooling B1/B2 gives 3552 steps over 160.121006 seconds in 159 intervals:
22.183223 step/s or 45.079112 ms/step. A1 saves 0.544380 ms/step, equivalent
to 1.207610% lower latency or 1.222371% higher step throughput. Against B1
alone it saves 0.636861 ms/step; against B2 alone it saves 0.453447 ms/step.
All eight ordered request payload hashes match across A1/B1/B2. Generated
text can differ: B1 had two mixed-CJK occurrences and B2 had four, each in
one response. All three arms passed all 18 quality checks.

The candidate's three per-response step rates were 22.503781, 22.400681 and
22.456742. B1's were 22.134375, 22.013855 and 22.251537; B2's were
22.224976, 22.226805 and 22.232711. These response groups share their boot
and are not independent boot repetitions. The one-second intervals are not
independent samples for a significance claim.

The fixed output rates were 73.549324 / 77.373059 / 79.336058 tok/s for A1,
69.421314 / 76.898252 / 69.805059 for B1, and 67.415558 / 73.178048 /
73.153069 for B2. Whole-onepass speculative acceptance also differed
(48.5%, 46.7%, 44.7%), so the output-token difference cannot be attributed
solely to the kernel. A1 reported cold compilation; A1/B2 saved rank caches,
whereas B1 loaded warm rank caches. There is no claim of a prefill gain.

A1/B1 used source `f3e82e4317fc6215135839c5a02ebf6b14f8119d` and overlay
`29851eaa8973`. B2 used `47b0659be395b7a707d14e0081c2d2fc402eec1b` and
overlay `ca75a479821b`, after incorporating current main and runner fixes.
The overlay stamps include source metadata and therefore differ. A direct
Git comparison found byte-identical `overlay/`, `build/`, `profiles/`,
`bench/onepass.py`, `probes/input_reuse_channels.py` and
`launchers/start-glm53-nvfp4-tp4.sh` across those two revisions. The mounted
four-source hashes and all checked runtime knobs match across all three
arms, except for the intended consumer flag. All boots selected GMU 0.6329
and pinned 2,000,000 KV tokens (1056 blocks). All three used image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Four-rank before/after source, image, boot and knob checks passed. Every
candidate rank proved actual `T=6 bf16=True vec4=True` capture. All eight
SSE request/output hash pairs matched each arm's recorded requests, and each
exclusive-traffic audit reported no issues. Normal dynamic boot memory
sizing remains in effect.

## Excluded v14 B2 and retry history

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

`--baseline-only` was added to retry B2 with the complete workload and
all admission/correctness/runtime gates without another A1/B1 boot. It adds
read-only host-memory receipts to diagnose any recurrence. Serving source,
profile, image, actual boot budgets and ordered requests were verified
before combining the completed v17 baseline with the original pair.

The first baseline-only retry, `arconsumer0908v15` at `d980197a`, passed its
complete deployment CPU gate and reused all 15 GPU correctness stages. It
was admitted at 21:25:48 KST but stopped before the B2 boot: main #493 landed
between its initial source check and deployment's refresh. The fallback source
check also exposed an argparse invocation bug (`require-base --repo REPO REF`).
Deployment now passes the positional ref before `--repo`; a regression runs
that exact invocation and proves it accepts documentation-only advancement
while still rejecting missing upstream runtime code. The retry incorporates
main #493 instead of weakening the current-main requirement. v15 produced no
serving record; its supervisor restored approved main and handed off to
`attr6-0908` at 21:30:37 KST.

`arconsumer0908v16` stopped during CPU preparation, before reserving GPUs.
The new source-guard regression changed `tests/test_fleet_source.py`, but its
fleet cache audit digest had not been renewed. The conservative full-tree
fallback then failed the test expecting the reviewed fleet scope. AST review
confirmed that existing tests were unchanged; the only additions were the
CLI regression and its `shlex` import. Its launcher/helper reads are covered
by the existing `launchers/` and `bench/` dependency prefixes. The digest is
updated after that review. All fleet/startup/logic registry hashes and fleet
test inventory match, and 14 focused source/cache tests pass. The complete
deployment CPU gate was rerun before v17 admission.

## Completed v17 B2 and fleet handoff

`arconsumer0908v17` retained ticket `17888721744133545`. Fleet paused it
before admission when main #494 advanced. After incorporating that main
revision, the final source `47b0659b` passed the complete deployment CPU gate
(71,123 reported checks, including 275 fleet tests, 50 MK regressions and
the separate chat/tokenizer gate). Counts from different runners are not
additive coverage claims. The pinned approved-main recovery receipt and
all 15 source-bound GPU stages were retained; no additional GPU correctness
run was needed for the unchanged serving code.

The supervisor admitted revision 4 at 22:05:34 KST; B2 completed its eight
requests at 22:16:49. The four-node before/after runtime proof, all eight SSE
hash pairs, three exact 2048-token responses and exclusive-traffic audit
passed. Before requests, available host RAM was 14.743 / 8.395 / 11.788 /
15.224 GiB on srv1/2/3/4; after-request samples are retained. There was no
worker death or OOM. Memory protection and request/boot budgets were unchanged.

The campaign stopped its loopback serving after measurement. Payload and
supervisor both exited 0. Fleet recorded `handoff-accepted` by the next
queued job, `eplocal0908v5`, at 22:17:24 KST and then admitted that job.
This proves orderly transfer of serving/recovery responsibility, not public
service readiness during the successor's work. Reservation order was preserved.

## Retained evidence

- srv2: `/home/choiceoh/glm53-logs/ARCONSUMER-arconsumer0908v14/` and
  `/home/choiceoh/glm53-logs/ARCONSUMER-arconsumer0908v17/`.
- Local: `runs/arconsumer0908/v14/`, including A1/B1 independently recomputed
  summaries, raw records, SSE channels, all-rank proof, boot logs, earlyoom
  and kernel journals, host sysstat/container events, and fleet exit state.
- Local B2: `runs/arconsumer0908/v17/`, including `B2-summary.json`,
  `final-comparison.json`, raw records, SSE channels, all-rank proof,
  host-memory/boot logs, `fleet-final.json`, `fleet-lifecycle.jsonl`,
  `fleet-run-final.log` and `stop-experiment.log`.
- GPU correctness: `ARCONSUMER-arconsumer0908v13/gpu/`, all 15 stages passed;
  copied into v14 after verifying the complete source-bound evidence.
- Deployment CPU release gate: `runs/arconsumer0908/v14/cpu/`, passed before
  GPU reservation, including approved-main recovery preparation.
- Final v17 CPU release gate: `runs/arconsumer0908/v17/cpu-main494/`,
  source `47b0659b`, evidence key
  `41f7b534c243a26b9bfc82fb1131f5bd9a00be9594595ddff9ac0a2ce7f7b555`.

Recompute each completed arm with
`python3 runs/arconsumer0908/report_serving.py runs/arconsumer0908/v14 NAME`
for A1/B1, or use `runs/arconsumer0908/v17 arconsumer0908v17B2` for B2.
The ignored local reports supplement the retained remote raw evidence.
