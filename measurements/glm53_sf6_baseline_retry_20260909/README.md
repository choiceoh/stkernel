# Baseline-only SF6 retest: completed

The isolated baseline completed at **07:25:25 KST on 2026-09-09** with
supervisor and payload return code **0**. The individual canonical record,
exclusive traffic, memory guard and retained baseline runtime observation pass.
This is a valid baseline-only measurement. The original A/B remains INVALID;
this retest does not establish an SF6 speedup because the MHC runtime path also
changed between the original candidate and the new baseline.

| Metric | Original SF6 candidate A | Clean baseline retry B |
|---|---:|---:|
| Fixed pooled step/s | 20.281701533 | 20.206613283 |
| ms/step | 49.305527860 | 49.488748361 |
| Window median step/s | 19.880270979 | 19.873337126 |
| Fixed output tok/s | 61.954793538 | 62.843273357 |
| Fixed window count | 92 | 91 |
| 2K cold TTFT, s | 2.440095250 | 2.431041133 |
| 2K warm TTFT, s | 0.838278340 | 0.888750849 |
| 32K TTFT, s | 10.881572252 | 10.847696893 |
| 128K TTFT, s | 42.228542329 | 41.445911947 |

The raw candidate-minus-baseline step difference is +0.371602%, or
**0.183220501 ms/step lower** (-0.370227%). Window median differs by only
+0.034890%; output throughput is -1.413803%. These are descriptive cross-run
arithmetic, not a matched A/B performance verdict or statistical significance.
32K and 128K each use one combined request, so there is no separate warm sample.
Both runs report cold_compile=true, but that flag does not make their actual
runtime paths identical or establish a causal cold-prefill difference.

## Completed baseline evidence

There were exactly 8 completed requests from 0 to 8, no request left running
or waiting, and all 202 traffic samples had at most 1 running and 0 waiting.
Three seeded fixed requests generated exactly 2048 tokens each. All 8 ordered
request SHA256 values match old A. Quality is 18/18 and Korean corruption is 0/8.
The pooled baseline is 1851 steps / 91.603673216 seconds across 91 windows.
Per-repeat pooled rates are 20.287275, 20.198576 and 20.131317 step/s.

All 148 memory samples pass the four-node 10 GiB guard. Minimum MemAvailable
is 12.684128 GiB on srv2, 18.048424 on srv1, 14.172050 on srv3, and 18.629414
on srv4. This is a guard result, not a matched estimate of SF6 memory savings.

The observer retained before/after four-rank reports and original logs with
sealed receipts. Collection is COMPLETE and runtime validation is PASS.
All ranks show actual KV 665, GMU 0.6429, expected image/knobs and source hashes,
SF6 markers zero, OSAR PDL PASS, MHC consumer PASS and T=6 capture. The head's
actual serving argv and client use 127.0.0.1:18000. Prepared observation was
collected under the exact reservation; final observation is bound to the same
boot and the original completed record. No independent SSE transcript or
separate GPU numerical/sanitizer probe was added by this canonical path.

## Why the old candidate is only a reference

Old A and the contaminated old B both failed the common MHC T16 selftest and
had no fused AR-consumer MHC capture. New B passes this selftest on all four
ranks and captures the fused consumer at T=6 (bf16=True, vec4=True). Therefore
the actual MHC execution path differs despite identical source bytes. No MHC
code fix was applied between these runs; the reason for the changed selftest
outcome is unresolved. OSAR consumer PDL is separate and passes in both runs.
The prep-fused preimage check DISARMs to stock for the same utils.py drift in
both runs, so that warning is common rather than a new baseline-only change.

The API bind/port, git revision and serving argv/script identity also differ.
The strict two-arm A/B checker is unchanged and cannot accept this pair.
Neither the earlier contaminated 17.273478 step/s baseline nor its apparent
large difference is reused as a valid baseline. No default promotion or PR
merge follows from this result.

## Execution and source identity

Session `sf6-base-0909v3`, ticket `17889056112929421`, launch
`368b2fcbeabe41208ce22c53b07b7a61` ran from 07:13:32 to 07:25:25 KST (payload
712.6 seconds). Supervisor PID 2929421 / start 40799859 owned the reservation.
The sole canonical `bench/ab-lever.sh` arm was `sf6-base-0909v3B`, static `t,r`
with no SF6 lever. Exact serving revision:
`b7b06b70a534fdfae756f7410ac40305763f6c3a` in
`/home/choiceoh/stkernel-sf6-direct-0909`. It remains frozen after this run;
subsequent local commits only retain evidence and documentation.

Current-main ancestry was merged normally. All 63 serving overlays plus the
profile, benchmark scripts and runtime Reader are byte-identical to old A's
`85e25370779c2b8a6c9aaefa275b1dc8f27d60d0`; see `overlay-equivalence.json`.
Preparation used origin/main ancestry, clean checkout and 92 CPU tests PASS.
Immutable image is `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
Settings were SPEC_K=5, KV target1100000/hybrid187/maxlen1048576, AR consumer
PDL=1, MK PDL=1, compact AR=0, inline RDMA=0. Standard 2K/32K/128K plus exclusive
fixed 3x2048 ran once. Queue order and admission guards were preserved.

The isolated loopback port avoids existing port 8000 traffic, while the
exclusive counter checks still protect against arbitrary local clients.
Passive observer PID2930391 used required --revision b7b06b70... and helper
SHA256 `f4b5cfd560e37892b6793f8450d3b53b775492690b7348652a366f35d961b2a6`.
It performs no inference, GPU probe or serving lifecycle action. Central idle
controller owns subsequent serving recovery. Completion monitoring is closed.

## Retained artifacts and reproduction

`v3/final/` contains the original JSONL record and memory samples, all eight
rank reports/logs, sealed before/after receipts, observer state, original
record binding, expected source hashes, run log, terminal reservation and
actual campaign.exit=0, individual validation, raw comparison and independent
audits. `SHA256SUMS.json` binds the retained final snapshot. Initial inputs,
source and launch/admission receipts remain in `v3/`. Complete local raw copies
also remain in `/tmp/sf6-base-onepass-0909v3/{final,inputs}`.
Remote inputs: `/home/choiceoh/glm53-logs/sf6-base-0909v3-inputs`.
Remote results: `/home/choiceoh/glm53-logs/SF6-BASE-sf6-base-0909v3`.

Reproduce individual validation by importing
`probes/analyze_decode_next_onepass.py`, reading the sole record with
`read_jsonl`, calling `validate_record(record, canonical=True)`, and calling
`validate_memory(read_jsonl(memory_path))`. Runtime proof is re-evaluated with
`decode_next_runtime_proof.validate_report` and `compare_snapshots`; original
logs are hashed and reparsed, receipt artifact hashes and record SHA verified.
The arithmetic in `analysis.json` is explicitly a separate-run reference;
do not feed a fabricated two-arm receipt to the full canonical analyzer.

## Earlier preboot failure

`sf6-base-0909v2`, ticket17889043822842777, was admitted behind `mb9221-0909`
at 06:56:37 and ended at 06:56:40 with returncode2 before any model boot or
inference. The source-base guard required main#503, which tracked a composed
KV-zero file already byte-identical in this branch. The guard was preserved,
main was merged, and v3 received a new normal ticket. The v2 terminal status,
raw log, exact inputs, observer failure and actual exit remain unchanged in
`v2-preboot-failure/`. No v2 speed measurement exists.
