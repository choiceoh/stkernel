# SF6 unpack: canonical two-arm onepass

`sf6_unpack_onepass_v1.json` contains the exact canonical argv. Submit from
the clean, composed candidate checkout with `REPO` set to that checkout:

```text
fleet.sh run --gpu --detach --prepare probes/sf6_unpack_prepare.json
  sf6-unpack-0909v1 30 "SF6 scalar versus four-byte unpack onepass" -- <argv>
```

The payload is `bench/chain.sh`, with A setting
`VLLM_GLM53_SF6_UNPACK_U8X4=1` and B using an empty override. Both retain
`t,r,sf6` and packed-only scale ownership. The submitted profile explicitly
sets unpack=0, matching the previous scalar implementation and allowing the
canonical baseline detector to recognize B. No third baseline boot is needed.

Both arms use the same committed source and immutable production image,
SPEC_K=5, KV_TOKENS=1100000, KV_HYBRID_BLOCKS=187, MAX_LEN=1048576,
compact AR=0, inline proxy=0, AR consumer PDL=1 and MK PDL=1. Serving and
the client use loopback port 18000. Existing host-memory gates remain active.

The unchanged onepass workload is 2K/32K/128K retrieval quality and prefill,
plus three fixed 2048-token decode requests with exclusive C=1 checks.
`combine_min_ctx=32000` gives three 2K prompts, one combined 32K prompt,
one combined 128K prompt and three fixed decode requests: eight total.

Start `observe_decode_next_onepass.py --sf6-unpack` only after the exact
reservation ticket and supervisor own the boot hold. Until then, read only
pending/holder/process files. Bind candidate, baseline, session, ticket,
port 18000 and the frozen checkout to the observer. It adds no inference
requests and retains four-rank source, environment, boot, log and serving
marker evidence before and after the completed records.

The unpack variant requires both arms to finalize 42 packed-only layers per
rank and to serve the selected unpack mode from distinct compiled artifacts.
Actual MHC state must match on every rank. Explicit self-test failure with
no capture is reported as matched fallback only when both arms prove it;
missing markers or mixed admission cannot establish a comparison. Existing
SF6-direct-versus-raw validation remains unchanged.

After the supervisor finishes, preserve its actual return code as
`campaign.exit` and run:

```sh
python3 probes/analyze_decode_next_onepass.py EVIDENCE_DIRECTORY \
  --candidate sf6-unpack-0909v1A --baseline sf6-unpack-0909v1B \
  --canonical --sf6-unpack
```

Report pooled step/s, ms/step, window median, output tok/s, prefill, quality,
exclusive traffic and MHC admission together. One boot per arm cannot
establish statistical significance or replace separate CUDA race checks.
Serving recovery remains with the central idle controller.

## Independent scalar follow-up

After A completed but failed the Korean gate, the operator requested the
missing scalar measurement. `sf6_unpack_baseline_onepass_v2.json` supplies
one empty defaults arm on source `c03da265`, without another vector or recovery
boot. Its explicit video capacity retains A's `image:4,video:1` condition after
the required main merge changed the profile default. The original v1 payload
and no-GPU-hold refusal are retained separately.

`observe_sf6_baseline.py --sf6-unpack` waits using file-only reservation checks
until the bound ticket and supervisor own the hold, then captures the four-rank
scalar runtime before and after the completed record. It issues no inference.
The defaults arm's supervisor exit is not itself a quality verdict; inspect
the raw record's quality and Korean scanner results independently.

Results and the offline diagnostic are under
`measurements/glm53_sf6_unpack_baseline_20260909/`. The independent follow-up
does not repair the original failed two-arm campaign or authorize adoption.
