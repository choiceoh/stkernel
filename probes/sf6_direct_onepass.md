# SF6 direct-prefill canonical onepass

`sf6_direct_onepass_v1.json` is the exact argv for `sf6-direct-0909v1`.
Submit it through `fleet.sh run --gpu --detach --prepare probes/sf6_direct_prepare.json`
from a clean, composed candidate checkout, setting `REPO` to that checkout.
The payload is the canonical `bench/chain.sh`; it adds no independent GPU probe
or inference request. Keep the reservation's original ticket and queue order.

- A: `VLLM_GLM53_B12X_STATIC_V2=t,r,sf6`; decode and prefill read packed scales.
- B: `VLLM_GLM53_B12X_STATIC_V2=t,r`; original scales.
- Both: compact AR=0, inline RDMA=0, AR consumer PDL=1, MK PDL=1,
  SPEC_K=5, KV_TOKENS=1100000, KV_HYBRID_BLOCKS=187, MAX_LEN=1048576.
- One serving boot per arm; identical immutable production image and source.
  Standard 2K/32K/128K onepass quality/prefill plus exclusive fixed 3x2048 decode.
  Existing four-node 10 GiB host-memory guard remains enforced.

Start `observe_decode_next_onepass.py --sf6-direct` only after the exact ticket,
session and supervisor PID own the official boot hold. A file-only waiter may
wait for that ownership; starting the observer early can block admission under
the legacy GPU process classifier. Supply the same `--session`, `--ticket`,
`--candidate sf6-direct-0909v1A`, `--baseline sf6-direct-0909v1B` and output path
`/home/choiceoh/glm53-logs/SF6-DIRECT-sf6-direct-0909v1`.
The observer adds no inference requests and retains all four ranks' source,
image, configuration, startup markers and logs before/after the completed record.

The SF6-direct validation variant is explicitly bound in every runtime report,
observer state and phase receipt. It requires direct ownership finalisation of
42 layers and 4,756,340,736 original scale bytes per rank, in addition to the
existing SF6 serving and AR/MHC startup checks. These are expected storage
geometry values, not a measured reduction of host or GPU memory. Actual memory
samples and the final source-matched A/B determine the memory and timing result.

After the exact supervisor finishes, retain its real final return code as
`campaign.exit`; never infer success from an existing partial record. Run:

```sh
python3 probes/analyze_decode_next_onepass.py EVIDENCE_DIRECTORY \
  --candidate sf6-direct-0909v1A --baseline sf6-direct-0909v1B \
  --canonical --sf6-direct
```

The primary is pooled decode steps/s and reciprocal ms/step. Report output
throughput, window median, prefill and quality alongside it. One boot per arm
cannot establish statistical significance. Dedicated GPU numerical/sanitizer
checks and independent SSE recording are not performed by this canonical path.
This experiment does not change defaults or authorize a merge. Earlier v5
failure artifacts remain separate; they are not evidence for this implementation.
