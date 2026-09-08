# GLM53 decode transport and lossless scale candidates

This follow-up implements three opt-in candidates above the adopted AR/MHC
consumer path. `VLLM_GLM53_AR_CONSUMER_PDL=1` and `t,r` remain the defaults.
The operator resumed GPU onepass testing on 2026-09-09.
CPU checks and compiler results establish host contracts and buildability; device numerics, transport
ordering under RDMA, sanitizer results and step-speed gains remain separate.

| Candidate | Selection | Removed work |
| --- | --- | --- |
| Compact AR | `VLLM_GLM53_AR_COMPACT_CTA=1` | Empty CTAs in small AR consumer launches |
| Inline RDMA completion | `VLLM_GLM53_AR_PROXY_INLINE=1` | Repeated work-request setup and the NIC's separate read of an 8-byte flag |
| Lossless MoE scales | `VLLM_GLM53_B12X_STATIC_V2=t,r,sf6` | Eligible FC1/FC2 scale stages shrink from 2048 to 1552 bytes (24.22%) |

## Transport contracts

Compact calls use 12 CTAs for at most 32,768 elements, including T=6 and
T=8 decode. Larger direct C++ calls retain 48 CTAs. Each compact CTA contributes
four publication tickets; each ordinary CTA contributes one. The enabled
process uses one sequence-based completion rule for both geometries, including
counter wrap. The default-off build preserves the original publication code.
Every writer still performs its own system fence before the CTA barrier and
ticket. The last publisher writes the byte count before the transmit sequence.
PDL release and reduced-input dependency waits retain their ordering.

Inline mode owns a fixed work-request pair for each peer and ring slot. The
payload stays a non-inline RDMA write, followed on the same RC queue pair by
the inline, signaled completion flag. Source bytes are copied during
`ibv_post_send`; request objects are neither copied nor moved after setup.
The actual queue-pair capacity must support at least eight inline bytes.
CQ completion and all-peer ACK still determine safe ring reuse. All ranks
agree on consumer, compact and inline modes before launching collectives.

The transport header is deployed alongside the CUDA source, included in the
extension cache identity and retained in source evidence. CUDA graph cache
factors include both new flags. Actual capture and successful inline posting
have distinct proof markers; configuration alone does not prove execution.

## Scale storage and fallback

`sf6` targets the current reform's FC1 and FC2 layouts, rather than the old
`q` probe's 4 KiB FC1 stage. The original scales remain available to prefill
and larger batches. A layer whose scale codes cannot be represented exactly
uses the uncompressed reform path; no clamping or scale rounding is allowed.
The packed buffers are owned alongside the weight views and retained for
captured launches. Legacy `q` remains probe-only and cannot be combined with
`r`. Packing reduces scale traffic, not all expert-weight traffic; no speedup
percentage is inferred from the old `t` experiments.

For E=288, hidden=4096, intermediate=512, the two packed planes add 81.844 MiB
per layer, or about 3.44 GiB per rank if all 43 layers are eligible. This is
additional to the retained original scales. Packing uses bounded chunks;
KV sizing, GMU and memory guards are unchanged.

## CPU reproduction

On srv2, from a clean, composed checkout:

```bash
REPO="$PWD" bash /home/choiceoh/stkernel/bench/fleet.sh run --cpu \
  decode-next-cpu 15 'Decode transport and SF6 device-free checks' -- \
  python3 probes/run_decode_transport_sf_cpu.py --out /absolute/fresh/evidence
```

The six overlay-publication tests run on the host with rsync; the serving
image has no rsync and does not run those host deployment tests.
The runner uses the immutable serving image with `--runtime=runc`, no GPU
visibility or network, two CPU cores and bounded memory/swap. It runs the
CPU contracts, compiles all four transport flag combinations, and compiles
the MoE baseline/candidate at M=2/6/8/16 plus tiled prefill. It also compiles
the existing u/v/t/q ABI and the actual 2048/4096-byte expansion helper.
`--stage` selects
an individual failed stage for a focused retry. Every invocation needs a
fresh output directory; receipts bind the source commit and log hashes.

GPU validation must exercise the new transport
flags in all four ranks and the full `t,r,sf6` kernel, including fallback,
changed inputs and retained graph lifetimes. The existing baseline-only AR
GPU runner does not establish correctness of enabled follow-up flags.
Prior #473 GPU receipts cannot validate changed transport code or its header.

## Canonical onepass campaign

PR #500 changed fleet admission while this request was waiting. The active test
uses the current canonical `bench/chain.sh` directly, with one all-three
candidate arm and one empty-knob profile baseline. The earlier standalone GPU
harness remains available as source, but is not admitted or run by this campaign.
Dedicated GPU numerical and sanitizer coverage therefore remains unmeasured.
The active GPU evidence is the model's startup checks and onepass quality gates.
No wrapper whitelist, custom lever, boot-only leg or additional GPU hook is added.

Both boots use the same frozen overlay and immutable serving image, SPEC_K=5,
the standard 2K/32K/128K quality/prefill workload and fixed 3 x 2048 decode.
`ONEPASS_MEMORY_DIR` selects the canonical four-host 10 GiB guard. The original
KV/GMU remain unchanged, and the standard default baseline remains available
for normal serving. Production recovery belongs to the central idle controller.

`observe_decode_next_onepass.py` is a separate passive observer. It sends no
completion requests and never launches/stops GPU work. It captures the existing
four-rank runtime proof and boot logs before and after each arm, plus a hash of
the original onepass record. `analyze_decode_next_onepass.py --canonical`
requires both observed boots, ordered request hashes, exact mode/source/image
identity, full quality, Korean and exclusive-traffic coverage, memory evidence,
and the final supervisor exit. The canonical client records output hashes;
independent raw SSE capture and standalone GPU numerical proof are explicitly
not claimed. The analyzer's original stricter mode remains unchanged.

The primary rate is `decode.fixed_pooled_step_s`; its reciprocal is ms/step.
Window medians and prefill are reported separately. One boot per arm is a bounded
comparison, not a repeatability or significance claim. Compile-cold-prefill
values are not comparable across different compile states. This test does not
promote defaults or merge the optimization.

Use `probes/decode_next_prepare.json` with the official `fleet.sh run --gpu`
command. Pass the fixed workload, output directory and immutable image as literal
`env NAME=value` arguments to the canonical chain; its two arm arguments are
`NAMEA=VLLM_GLM53_AR_COMPACT_CTA=1 VLLM_GLM53_AR_PROXY_INLINE=1 VLLM_GLM53_B12X_STATIC_V2=t,r,sf6`
and `NAMEB=`. Start the passive observer before allowing the reservation to run.
The earlier `run_decode_next_campaign.sh`/`decode_next_lever.sh` are historical
harnesses and are not an alternate admission path under onepass-only policy.

## Recorded CPU validation: 2026-09-09

All 11 required CPU stages passed for production code `e8e20130`.
The [summary](evidence/decode-next-cpu/summary.json) selects successful stages,
verifies their log hashes and records the component source paths compared
between commits. Original failed/partial reports are retained unchanged.
No failed stage supplies successful coverage, and no transport compiler was
repeated after the scale-only correction.

| Gate | Result |
| --- | --- |
| Serving-image core and focused tests | 71,123 core assertions plus 50 megakernel regression cases; 37 focused test cases passed, zero skips |
| Native transport | All four compact/inline combinations compiled; actual extension modes and CUDA-not-initialized checked |
| Serving MoE at max_rows=640 | `t,r` and `t,r,sf6` compiled for M=2/6/8/16; actual FC1/FC2 byte maps passed for the eligible shapes |
| Compatibility | `u`, `v`, `t`, probe-only `t,q`, and tiled prefill compiled |
| Production expansion helper | 2048/4096-byte methods compiled, plus sf6 M=1/2/6/8 and raw M=16 fallback at max_rows=128 |

The immutable image is Torch 2.13.0+cu130 / CUDA 13.0. Transport ran on srv1,
and final scale gates ran on srv2, all through the official CPU lane using
device-free, network-free containers and fresh caches. A retry on srv1 was
declined by the 12 GiB memory guard before creating a container; it moved to
the node with available memory without changing that guard.

The first actual CuTe check caught an FC2 ordering error: shared memory needs
`[K64][row128][512B]` order. Commit `09d68b64` fixes packing and adds the exact
coordinate regression. The final probe-only fix resolves annotations at
definition time (`e8e20130`). Device execution and step speed remain unmeasured.
