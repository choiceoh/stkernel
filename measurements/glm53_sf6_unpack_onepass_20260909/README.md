# SF6 unpack onepass, 2026-09-09

The subsequently requested independent scalar baseline is retained in
[`../glm53_sf6_unpack_baseline_20260909/README.md`](../glm53_sf6_unpack_baseline_20260909/README.md).
It measured 20.19704 step/s versus this candidate's 20.21523, and passed the
Korean gate. The original campaign and files described below remain unchanged.

The vector candidate completed all eight canonical requests. The Korean gate
failed on the last fixed-decode answer, so the chain stopped before the scalar
baseline. This is an **INVALID A/B comparison**, not evidence of a speedup or
slowdown caused by the unpack change. PR #507 remains a draft and the serving
profile retains scalar unpack (`VLLM_GLM53_SF6_UNPACK_U8X4=0`).

## Actual candidate results

| Metric | Vector candidate |
|---|---:|
| Fixed-decode pooled step/s | 20.21523095 |
| ms/step, reciprocal of pooled step/s | 49.46765151 |
| One-second window median step/s | 19.87315354 |
| Retained fixed-decode windows | 88 |
| Fixed-decode output tok/s, pooled | 64.56809422 |
| Retrieval quality | 18/18 |
| Korean gate | FAIL: 1/8 answers, two CJK characters |
| Fixed-decode completion lengths | 2048, 2048, 2048 |
| Exclusive traffic | Eight completed requests; no unrelated traffic |

The step rate is 1,791 engine steps divided by 88.59656385798007 seconds of
fully enclosed fixed-decode windows. Output tok/s is 6,144 completion tokens
divided by the sum of the three request decode durations. These are different
measurements; speculative acceptance affects output tok/s.

| Prefill context | TTFT | Prompt tok/s | Notes |
|---|---:|---:|---|
| 2K first | 2.418303 s | 879.956 | First boot on new source; cold compile flagged |
| 2K warm | 0.875224 s | 2431.377 | Canonical minimum of warm TTFT samples |
| 32K | 10.787132 s | 3017.021 | One combined request |
| 128K | 41.321352 s | 3111.200 | One combined request |

The third fixed-decode answer contained `Halvorsen博士`. The retained scanner
excerpt is in `final/campaign.log`; `records.raw.jsonl` retains the output hash
and the two-character `cjk_mixed` count, not the complete generated text. The
scanner's rules and thresholds were unchanged. There is no matched baseline
output to attribute this failure to SF6 unpack arithmetic or dismiss it as an
existing model behavior.

## Runtime and ownership evidence

All four ranks passed the candidate's before/after runtime receipts: the
committed 63-file overlay, immutable image, private port 18000, KV blocks=665,
actual vector static/dynamic artifacts, 42 packed-only layers, and release of
4,756,340,736 raw-scale bytes per rank were verified. No raw-scale fallback
was observed. Each rank retained 3,604,414,464 packed bytes.

The MHC consumer self-test failed identically across the candidate's four
ranks and the consumer was not captured. Ordinary MHC and the OSAR consumer
remained armed. These results do not establish performance with the combined
MHC consumer enabled. The scalar arm never booted, so matched MHC state across
arms is unproven.

The actual M6 vector artifact ends in `f0c54eb04301a8a6`, and the dynamic
artifact ends in `e0c4370075719b32`. CPU compilation independently produced
these keys; scalar compilation produced `8b9695796f84cd45` and
`02112345f14079ae` respectively. Mode separation also covers the in-memory,
disk, and vLLM compile caches.

## Run identity and terminal state

- Source: `c85a1e75a7187ad69baefb30ef143f92f13430e8`.
- Accepted main ancestor: `fbeca7515fc58fe59fe181eb7c81b2d6fdaff93d`.
- Image: `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
- Session: `sf6-unpack-0909v1`, ticket `17889114423497508`.
- Supervisor: PID `3497508`, process start token `41382941`, revision 1.
- GO: 2026-09-09 08:55:28 KST, after the preceding reservation released.
- Candidate finished: 09:08:34 KST. Supervisor and payload returned **4**.
- The passive observer retained candidate before/after evidence, then recorded
  the failed reservation. No baseline record was fabricated.
- Reservation release completed normally; recovery remains with the central
  idle controller. No manual recovery boot or extra measurement was run.

The exact two-arm argv and preparation spec are under `probes/` and described
in `probes/sf6_unpack_onepass.md`. Each arm was configured for SPEC_K=5,
KV_TOKENS=1100000, KV_HYBRID_BLOCKS=187, 2K/32K/128K retrieval and prefill,
three fixed 2048-token decodes, and exclusive C=1. Only unpack mode differed.

## Retained evidence

- `final/`: original remote artifacts, including failed campaign exit, raw
  candidate record, quality verdict, memory samples, all-rank runtime receipts
  and logs, and the original analyzer result.
- `remote-final.sha256.json`: hashes verified against all 52 copied remote
  files before local analysis.
- `candidate-raw-metrics.json`: explicit candidate-only calculation retaining
  the failed quality result and null comparison.
- `analysis.json`: offline re-analysis that retains the candidate's raw
  metrics even though the second arm is absent. It keeps the Korean failure,
  missing-baseline errors and null comparison; its 31 focused tests pass.
- `admission/`: queued reservation and frozen deployment preparation.
- `local-cpu/`: core 71,163 checks plus 50 megakernel regressions, sensitivity
  and focused runtime checks; the final onepass proof suites passed 80 tests.
- `cpu-compiler/`: scalar and vector production-image CPU compiler stages;
  both passed with no CUDA context. Selected-stage coverage is not full-suite
  coverage.

Re-analyze the retained campaign without inference:

```sh
python3 probes/analyze_decode_next_onepass.py \
  measurements/glm53_sf6_unpack_onepass_20260909/final \
  --candidate sf6-unpack-0909v1A --baseline sf6-unpack-0909v1B \
  --canonical --sf6-unpack
```

The expected exit code is 1 and comparison remains null. The original
`final/summary.json` is kept unchanged, including its missing-baseline errors.
The analyzer was subsequently corrected to keep partial-campaign metrics;
that reporting change does not alter the measured source or original files.
The original arithmetic and SASS evidence remains separately in
`measurements/glm53_sf6_unpack_20260909/`.
