# GLM-5.3 C=1 MoE tile bundle — 2026-09-08

The corrected bundle completed one matched A-B serving campaign. Observed
pooled decode step throughput rose **1.60%** and fixed-2K output throughput
rose **3.08%**. This is one boot per arm, not an independently repeated win.
The profile default is now `t,r` by operator request (PR #461, 2026-09-08).
Set the knob to `t` to roll back. The serving figures below retain their
original one-boot-per-arm limitations; promotion adds no new GPU results.

## Changes kept together

| Geometry / work | t | t,r for M<=8 |
|---|---:|---:|
| Padded M rows | 32 | 16 |
| FC1 N/K | 64/512 | 128/256 |
| FC1 halves per 128-column intermediate | 2 | 1 |
| FC1 gate/up stages consumed | 32 | 32 |
| FC1 TMA A/B/SFA/SFB bytes per stage | 8/16/4/4 KiB | 2/16/2/2 KiB |
| FC2 N/K | 128/128 | 256/128 |
| FC2 output tiles per H=4096 item | 32 | 16 |
| Compiled shared memory | 101,376 B | 98,304 B |

These are geometry/requested-transfer counts, not measured DRAM savings.
Weight storage and the per-128-column BF16 rounding boundary stay unchanged.
M>8 uses the original t geometry. The FC1 scale fragment restores separate
N/K axes for the M16 warp arrangement.

## Numerical correction and validation

The initial bundle (`96ea286`) passed M1/U8 but failed M2/U8: maximum error
3.5 against limit 0.1875. FP4 stores had applied the byte swizzle to nibble
offsets. The correction converts the outer offset to bytes before applying
`byte ^ ((byte >> 3) & 0x30)`. Compilation now checks all 1,024 packed bytes
against the consumer mapping, not only uniqueness. See `ADDRESS_DIAGNOSIS.md`.

- Exact serving image, device-free sm_121a compile: corrected M2 and M6,
  max_rows=640, PASS (`cpu/corrected/`). Larger-M fallback was compiled earlier.
- Full Linux CPU gate: 6,742 checks, 38 megakernel regressions and 115 fleet
  regressions PASS. Seven diagnostic-parser tests reject real memory faults,
  other APIs, mismatched counts and errors after compilation.
- Corrected GPU fixture (`52079ce`): 13 shapes, 130 baseline/candidate
  comparisons against independent stock; mutated graph replays and exact-zero
  route weights PASS. M2/U8 replay 0 error is 0.046875, limit 0.125.
  The tolerance formula is unchanged; measured stock noise sets its value.
- Memcheck retained 34 CUDA `cuGetProcAddress_v2` lookup errors, all before
  the first kernel compilation. No device memory diagnostic was reported.
  A strict parser admits only that complete, counted pre-kernel diagnostic;
  the full raw log and exit 77 remain in `gpu-corrected/`.
- Racecheck M6/U40: zero hazards, errors and warnings, exit 0.
  Source-identical numerical/memcheck evidence was reused before racecheck;
  the source comparison and resource receipt are in `gpu-corrected/continuation/`.

## Serving comparison

Source `1f265b1`, exact image recorded in `source.json`, TP4, C=1, SPEC_K=5.
While queued, main promoted CTA=4 in PR #465; both arms use that approved
default with input reuse enabled. Only MoE `t,r` versus `t` differs.
Three matched, fixed 2,048-token decode requests run per arm at a private
endpoint, plus the 2K/32K/128K prefill and quality ladder.

| Metric | Baseline B (t) | Candidate A (t,r) | Change |
|---|---:|---:|---:|
| Pooled fixed-window step/s | 21.727898 | 22.076208 | +1.603% |
| ms/step from pooled rate | 46.023779 | 45.297634 | -0.726145 ms |
| Window median step/s | 21.825397 | 21.850820 | +0.116% |
| Fixed-2K output tok/s | 70.612170 | 72.787236 | +3.080% |
| All-request speculative acceptance | 45.0904% | 46.7101% | +1.6197 pp |
| Fixed windows | 81 | 77 | — |
| Quality | 18/18 | 18/18 | pass |
| Korean corruption | 0/8 | 0/8 | pass |

Pooled step/s is total steps divided by total sampled seconds. Median window
rate is also retained; counters are sampled about once per second. The three
per-request pooled rates all favor A, but they share a boot and are not three
independent boot trials. Output tok/s also reflects speculative acceptance;
the acceptance figure above covers all requests, not only the fixed requests.
The standard judge uses window median and records +0.1% with no baseline
floor yet. This does not establish a large or independently stable gain.

| Prefill throughput (tok/s) | B | A | Sampling |
|---|---:|---:|---|
| 2K warm | 2486.73 | 2506.92 | minimum TTFT of repeated requests |
| 32K | 2409.55 | 2977.30 | one request per boot |
| 128K | 2816.04 | 3065.92 | one request per boot |

The long-prefill figures are single observations. Larger-M MoE geometry is
unchanged, so these differences do not establish a prefill benefit attributable
to this decode bundle.

`serving/summary.json` is independently regenerated from `records.raw.jsonl`,
all-rank before/after source and active-M6 proofs, exact request hashes, and
separate SSE channels. `serving/per-request.json` retains the three requests.
The same image, source, runtime controls, prompt lengths and request hashes
matched; no external requests entered the private measurement endpoint.

## Lifecycle and reproduction

The original speed-only waiter was cancelled when the user requested fixing
the numerical error too. Session `moereformfix20908` held the fleet. After the
API-only sanitizer interruption, the owned supervisor and final restore were
paused before any restore boot; continuation `MOEREFORMFIXCONT0908` reused
unchanged evidence, ran racecheck and both serving arms under that same hold.
The paused supervisor/restore were resumed after A/B and an approved-source
republish. No third measurement boot or per-feature speed sweep was added.

Entry point: `probes/run_moe_reform_onepass.sh` under canonical `fleet.sh run
--gpu`, using a clean current-main candidate and dedicated approved restore
checkout. It runs numerical/graph/sanitizer checks, A then B, the standard
judge and `probes/analyze_moe_reform_onepass.py`. Final public recovery remains
owned by the fleet supervisor. Original failures and continuation receipts
are preserved rather than overwritten.

Public recovery was verified while this session still owned the fleet:
HTTP 200, all four ranks on approved main and profile defaults (CTA=4, MoE=t).
See `serving/restore-proof.json` and `restore-proof-ownership.json`.
The supervisor finished recovery with exit 0 and released the fleet. The
initial payload exit 1 is retained for its API-only diagnostic interruption;
the completed A-B continuation exited 0. No paused owner processes remain.

Default promotion validation: 6,742 CPU checks, 38 megakernel regressions and
115 fleet regressions pass (`cpu/promotion-logic.log`). Kernel source hashes
match the measured candidate. The profile declaration is checked exactly;
the A-B runner explicitly pins old baseline `t`, and empty lever knobs read
the current profile default. No additional GPU run was performed.
