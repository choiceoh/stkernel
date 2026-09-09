# Scalar SF6 unpack baseline, 2026-09-09

The requested scalar baseline completed eight exclusive onepass requests,
retrieval quality 18/18, and the Korean gate with zero dirty answers out of
eight. Its observed step rate is effectively the same as the earlier vector
candidate. This run does not establish a useful speedup from four-byte unpack;
the profile remains scalar (`VLLM_GLM53_SF6_UNPACK_U8X4=0`) and PR #507 is a draft.

| Metric | Scalar baseline B | Earlier vector candidate A |
|---|---:|---:|
| Fixed-decode pooled step/s | 20.19703962 | 20.21523095 |
| ms/step | 49.51220668 | 49.46765151 |
| Window median step/s | 19.86883103 | 19.87315354 |
| Fixed-decode output tok/s | 65.79953963 | 64.53656683 |
| Fixed-decode windows | 87 | 88 |
| Retrieval quality | 18/18 | 18/18 |
| Korean dirty answers | 0/8, PASS | 1/8, FAIL |
| Supervisor / payload exit | 0 / 0 | 4 / 4 |

Raw arithmetic gives the candidate +0.09007% step/s, or 0.04456 ms saved per
step. This is one boot per arm in separate reservations. The timing does not
demonstrate a repeatable gain. Output tok/s is 1.919% lower for the candidate;
all eight output hashes differ, so output timing and quality differences
cannot be attributed solely to unpack arithmetic.

Step/s is the sum of fully enclosed fixed-decode engine steps divided by the
sum of their window durations: B is 1,770 / 87.63660581992008 seconds, A is
1,791 / 88.59656385798007 seconds. Output tok/s uses the canonical first-token
exclusion, `sum(completion_tokens - 1) / sum(decode_s)`. Each arm completed
three 2,048-token fixed requests. The earlier candidate README's 64.56809
includes the first token; this table consistently excludes it for both arms.

| Prefill context | Baseline TTFT | Candidate TTFT | Baseline prompt tok/s | Candidate prompt tok/s |
|---|---:|---:|---:|---:|
| 2K first | 2.383928 s | 2.418303 s | 892.644 | 879.956 |
| 2K warm | 0.851758 s | 0.875224 s | 2498.363 | 2431.377 |
| 32K | 10.843488 s | 10.787132 s | 3001.340 | 3017.021 |
| 128K | 41.401995 s | 41.321352 s | 3105.140 | 3111.200 |

Both records flag cold compilation. The 2K warm value is the canonical minimum
of two warm samples; 32K and 128K each have one combined request. These are raw
measurements, without a cold-start speedup claim.

## Quality and runtime

The candidate's last fixed-decode answer contained `Halvorsen博士` and failed
the unchanged Korean scanner. The baseline passed and its corresponding output
hash differs. The candidate failure therefore cannot be dismissed as reproduced
baseline behavior. Full generated text was not retained; the raw record keeps
hashes and scanner counts, and the campaign log retains the candidate excerpt.

All four baseline ranks passed source, image, environment, packed-only ownership,
artifact and before/after checks. Both arms release 4,756,340,736 raw-scale bytes
per rank, retain 3,604,414,464 packed bytes, finalize 42 packed-only layers and
allocate 665 KV blocks. The scalar M6 and dynamic artifact suffixes are
`8b9695796f84cd45` and `02112345f14079ae`; candidate suffixes are
`f0c54eb04301a8a6` and `e0c4370075719b32`.

Both arms have identical explicit MHC-consumer self-test failure on all four
ranks, with no consumer capture. Ordinary MHC and the OSAR consumer remain
armed. These runs say nothing about performance with the fused MHC consumer
active. Exclusive-traffic checks passed for all eight requests in each arm.

## Source and reservation identity

- A source: `c85a1e75a7187ad69baefb30ef143f92f13430e8`.
- B source: `c03da265a893c61e759eb56e2e43f77bac74a2cc`.
- B accepted main: `d1264d29e60133238f82054d17eb5cb15d95c335`.
- Image: `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
- B session: `sf6-unpack-base-0909v2`, ticket `1788927156325060`.
- Supervisor: PID 325060, start token 42954526, reservation revision 1.
- B GO: 2026-09-09 13:24:00 KST; chain finished 13:36:19 KST.
- Observer collection COMPLETE and runtime PASS; reservation released normally.

The first baseline preparation was refused before any GPU hold because main
had advanced. B includes that main merge. The 63 served source-file hashes,
launcher and canonical workload code are identical to A. The profile's new
video default is overridden with `MM_LIMIT={"image":4,"video":1}` in this
experiment only; the actual CLI matches A on all four ranks. The requested
workload, SPEC_K=5 and KV targets remain unchanged.

The record overlay stamps differ (A `bf371c1a59e6`, B `6bc0090c4d9c`). They
are the first 12 hex digits of the deployed manifest SHA, which includes its
`source_commit` header. Reconstructing the manifests from Git exactly reproduces
both stamps; only that provenance header differs. The 63 file rows and the
compile content hash are identical, as retained in `stamp-provenance.json`.
The offline diagnostic retains the stamp mismatch as an error; its 63 served
file hashes and across-arm runtime checks pass. Different source revisions,
reservations, stamps and the candidate quality failure remain explicit;
`valid=false` and `comparison=null` are never promoted to an adoption verdict.

## Retained evidence and reproduction

- `final/`: 28 original remote files, including both baseline collection phases,
  all-rank logs, raw record, memory samples and actual terminal receipts.
- `remote-final.sha256.json`: every copied file verified against remote bytes.
- `source-equivalence.json`: exact unchanged code scope and video override.
- `stamp-provenance.json`: reconstructed deployment stamps and identical
  serving content, including the reason the stamp check remains unequal.
- `admission/`: initial refusal without a GPU hold and v2 preparation/reservation.
- `diagnostic.json`: raw arithmetic, quality failures, output-hash differences,
  runtime checks, stamp mismatch, and null comparison.
- `local-cpu/`: core 71,167 checks plus 50 megakernel regressions, observer and
  exact baseline argv gates. The final offline diagnostic, observer and argv
  suites passed 24 tests.

The original candidate files in `../glm53_sf6_unpack_onepass_20260909/final/`
remain unchanged. Recompute without inference:

```sh
python3 -B measurements/glm53_sf6_unpack_baseline_20260909/analyze_baseline.py \
  measurements/glm53_sf6_unpack_onepass_20260909/final \
  measurements/glm53_sf6_unpack_baseline_20260909/final
```

The expected exit is 1 because the source stamp differs. No extra GPU run,
manual recovery boot, default change or merge followed this measurement.
