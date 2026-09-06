# M8 exact fastpath and fixed-window follow-up

Status: CPU gates passed; GPU probe queued. No new speedup has been measured.

The prior candidate remains retained and inconclusive. This follow-up adds
an opt-in reduction/scale-load path without changing its arithmetic result,
then increases the serving evidence beyond the prior 5/6 short-context windows.

## Candidate

- `VLLM_GLM53_MK_FP8_PACK2=1`
- `VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2`
- `VLLM_GLM53_MK_M8_FASTPATH=1`

The third flag replaces five shuffle/max pairs with a full-warp unsigned
maximum over nonnegative FP32 bit patterns. Each lane also reads just its
two FP4 scale bytes. The RQ=2/4 and low-rank paths retain their original code.
All three profile defaults remain 0. Implementation commit: `0421328`.

CPU validation: `tests/test_logic.py` passed 6,663 checks, including 30
megakernel and 20 fleet regressions. `tests/test_onepass_measurements.py`
passed eight contamination/window/timing fixtures. Both composed overlays
match their source, Python compilation and `git diff --check` passed.

## GPU gate and measurement protocol

`probes/mk_fp8_pack_bench.py --fastpath --reps 18` builds baseline, previous
M8 and new M8 from the same source. It checks FP8 conversion bytes, special
FP32/BF16 activation maxima, the independent GEMM oracle, SMLP2 and repeated
CUDA graphs. New M8 must be bit-identical to previous M8 on every tested
shape/replay. Twelve geometries have separate cold-weight and warm timings,
with all six execution orders balanced. The container is pinned to image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.

The serving [runner](runner.sh) uses four independent boots in B/A/A/B order
on one deployed build. Every arm runs the existing Korean 2K/32K/128K
onepass ladder plus three 2K-document responses with `min_tokens=max_tokens`
set to 2,048. Seeds, prompts, SPEC_K=5 and all other settings match.

Primary engine rate is total steps divided by total elapsed time in complete
fixed-response windows, with one-second edge margins. Report request tok/s
and TPOT alongside it. Each boot must have at least 20 eligible fixed windows.
The quality requirement becomes 18/18 retrieval checks and 0/8 corrupted
responses. Extra completed requests, observed concurrency/queueing, absent
counters, counter reset or incorrect output length invalidate the run.

The [analyzer](analyze.py) requires all four valid boots and candidate proof
3/3 before producing the comparison. It reports each boot, reversed pairs
and between-boot spread. Multiple windows within a boot are correlated and
are not counted as independent boot replicates. These fixed-output records
must not be pooled with the older variable-length workload.

## Queue issue observed before GPU execution

The canonical fleet accepted session `m8nextprobe` at 06:37 KST. At 06:38,
`prefillserve` yielded, but `legacy_busy()` continued counting its paused
`bash bench/chain.sh` parent as active benchmark traffic. Fleet was FREE,
serving healthy and idle, yet the probe could not acquire its turn. This is
a scheduler blockage before kernel execution, not a failed GPU test.
No hold or process was forcibly removed.
