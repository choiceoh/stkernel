# C1 ordered K products and joined queries — PR #946

The final default-on candidate is `ec5c671842a2c69e7ae1cbb15068324d0ad1a781`,
starting from `ee88028c` (#939) on a new branch. No commits were added to the
merged #939 branch. Native CUDA SHA-256:
`1a04724addbcf5b21801b77e594bd84c41b5547eb69aea4aadfb5ca1eaa611f6`.

Eight warps calculate independent K-block MMA products using the existing
`cp.async.cg` W4 reader. The epilogue replays the original per-slice FMA chain
and slice additions, preserving activation quantization and BF16 rounding.
C1 DSA queries share one grid with independent output ranges and the existing
three-slice reduction. The model-owned calls enable both changes by default.
The comparison control is a per-call probe argument, not a process-wide knob.
KDA input and wider batches retain their existing paths. No extra resident
weight representation is allocated.

## Same-build GPU result

`st-forward-ordered0914` passed 16 zero-tolerance numerical/replay groups in
91.74 seconds including setup/compile, without a model boot. Changed and
strided inputs, poisoned outputs, private scratch, independently rebound
direct destinations, both replay orders, mixed query widths and C4 followed
by C1 are covered. C1 K7 has eight token rows in this probe.

Warm captured B/A/A/B means, in microseconds:

| Component | Existing | Default candidate | Time change |
|---|---:|---:|---:|
| Dense MLP gate/up | 76.73 | 30.27 | -60.5% |
| MLA output | 27.45 | 22.98 | -16.3% |
| MLA output, direct destination | 27.03 | 24.97 | -7.6% |
| KDA output | 17.93 | 14.45 | -19.4% |
| KDA output, direct destination | 16.63 | 14.89 | -10.5% |
| Dense MLP down | 22.24 | 17.61 | -20.8% |
| Dense MLP down, direct destination | 20.91 | 18.80 | -10.1% |
| DSA query pair | 24.33 | 17.96 | -26.2% |
| DSA query pair after C4 | 24.62 | 18.02 | -26.8% |

KDA input (28.79 to 27.46 us) and C4 queries (81.52 to 81.42 us) use unchanged
kernels; their differences are not claimed as improvements. Evicted timing
is inconclusive, with large variation in unchanged controls and regressions
in several changed paths. In particular, evicted MLP gate/up is 124.50 to
235.92 us, and direct KDA output is 160.80 to 204.32 us. These results remain
in `timing-summary.json` and the raw events; warm gains do not establish
whole-model decode gains or the 24 step/s target.

Runtime: srv4 NVIDIA GB10, Torch 2.13.0+cu130, CUDA 13.0, pinned image
`sha256:062bb8e5d4c4ef658c8b57987e99c5e84e2b149c73c0ac721ea25263b258c93e`.
Real BF16 tensors come from
`/home/choiceoh/models/st-glm53-hybrid-gptq-v1/rank3of4.safetensors`.
Both arms use identical RTN W4 packs, not the consumer GPTQ pack. Direct
outputs use device memory destinations; this probe does not exercise NIC
transport or measure acceptance.

The focused CPU run had 21 passes and 12 GPU skips. Final production-flag
native compile/load passed with CUDA hidden: ordered kernels use 65–70
registers and the joined query kernel uses 76, with zero stack/local spill.
`compile-ordered.json` has the matching CUDA hash and complete resources.
The engine CI for executable head `ec5c6718` passed:
https://github.com/choiceoh/stkernel/actions/runs/34819727976

## Consumer reservation

The official replacement accepted `st-forward-pipeline-onepass0914`, ticket
`178937245629330`, frozen at `ec5c6718`. It replaces the cancelled
`st-forward24-onepass0914c` and retains its enqueue timestamp `1789367678`.
The reservation is queued; no consumer result is available at this write.
One candidate boot observes K7, 2K/32K/128K, C1 twice and C4 once. Individual
responses are bounded to 2048 tokens; combined responses to 6144 with a 4096
reasoning budget. The 35-minute estimate describes work, not queue waiting.
This is bounded performance observation, with answer grading excluded from
the user's performance objective. No foreign lease/container was stopped.
The queue receipt is retained alongside the GPU receipts. PR875 Oracle
comparisons and unfilled profile templates are supporting source evidence;
no consumer estimate is inferred from them. The operator requested immediate
default-on merge after the warm GPU result; consumer acceptance, tok/s and
step/s remain pending.

## Rejected implementations retained as evidence

The initial direct-register candidate `e4972867` passed 13 exact groups but
regressed in warm KDA input (28.93 to 77.67 us), direct KDA output (16.63 to
20.16 us) and direct MLP down (21.27 to 24.86 us). The cg/unroll repair
`0616bf1c` passed 16 groups but still regressed in KDA input (29.82 to 36.79
us), MLP gate/up (75.13 to 79.39 us) and direct KDA output (16.65 to 20.71
us). Both register implementations were removed from the final source.
Their receipts, logs and compile records remain here for audit.
