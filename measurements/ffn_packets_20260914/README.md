# Packet-native FFN, PR #895

**Current scope (2026-09-14): packet FFN only.** Compact KDA and mixed FFNs have
been removed from this PR's execution code; ordinary KDA/cache/graph and static
decode implementations match integrated main `6522564a`. See the
[scope decision](../../bench/ST_GB10_PACKET_ONLY_20260914.md).
Earlier CPU/compiler records below identify their own frozen revisions.

The P follow-up is implemented behind `STK_prefill_ffn_packets=1`, default off. It requires native eager TP4, chunk-ordered prefill, H4096/E288/I512/top-8, tiled SF6 M128 and **8192 < real rows <= 32768**. Short prefill, decode, unsupported packs and calibration observers use the existing FFN. Production still rejects experiment overrides.

`PacketBatch` owns the existing four rank-ordered FP8-v3 packets. One control-group vote per eligible prefill agrees all layers before transport; one data all-gather per FFN is retained. The router, routed expert producer and shared gate/up consume that owner on the current stream. The expert uses the same cached workspace and four-row 32 KiB shared stage as the ordinary M128 body. No whole BF16 FFN input is allocated. The packet source has a distinct compilation identity and pins its inherited producer's source and unchanged arithmetic AST.

All readers preserve FP8 → FP32 scale multiplication → BF16 rounding. Router tile shape/K summation and top-8 selection stay the same. Expert group-16 quantization remains per expert, including unequal input scales. Shared group-128 quantization crops real rows **before** the GEMM. The existing activation, down projection and reduction order remain intact.

At 32,256 rows, the removed BF16 intermediate is 252 MiB/rank; the received packet is 126.24609375 MiB and shared Q/S remains 129.9375 MiB. These are declared buffer sizes. The conservative workspace ceiling is unchanged; the memory ledger must establish an actual engine peak. Extra packet reads and inverse quantization can offset a memory saving.

## Validation

The implementation was rebased on `70038de0`, retaining main's fused MoE decode output and covered-query prefill refinement. Its prior drafter lazy import regression is repaired: public aliases bind the actual common-lane functions, and the single-row proposal imports its walk selector. Optional compact mode remains compatible with lightweight test models.

- Eleven new CPU cases cover geometry, rank agreement, observer fallback, one full all-gather, three-reader ownership, real-row shared GEMM shape, real forward/auxiliary ordering and the inherited producer AST. The frontend evidence test also permutes physical route rows while preserving expert/token identity.
- The pinned Linux image passed **221 tests**, with **13 CUDA-dependent skips** (234 discovered in 25 isolated modules), after rebasing to `70038de0`. This includes the earlier CI regression modules and main's covered-query/absorb-tile changes. Exact commands, module results and source fingerprints are retained in [cpu.json](cpu.json).
- `probes/engine_ffn_packets_compile.py` builds the actual baseline/packet router and CuTe persistent MoE, plus shared quantization: **five SM121 variants**, without a GPU or CUDA context. [compile.json](compile.json) identifies their sources and compiler runtime. Compilation is not GPU correctness or latency evidence.
- `probes/engine_ffn_packets_check.py` is a canonical, byte-pinned 8 GiB fleet probe. It loads the actual L3 rank weights, tests 8193/8194/8195/9216/32768 rows, requires exact router/routes/shared Q/S/output and exact expert FP4/SFA by `(expert, token)`, then checks BF16 atomic-scatter output with fixed 2% relative-max / 0.4% relative-RMS component limits and records baseline variance. A separate unequal per-expert scale fixture uses identical synthetic scales in both arms. Every correctness cell must pass without skips before B/A/A/B timings begin for that cell.

The GPU probe starts from received packets and ends at the FFN sum, on one GB10. It excludes pack/NIC/all-gather/reduce-scatter, uses synthetic activations and a shared FP8 pack prepared from actual rank weights, and does **not** measure serving TTFT, tok/s, model acceptance or calibration-store quality. Actual 32K/128K C=1/C=4 onepass and same-build decode/quality gates remain required before adoption. No GPU result or performance improvement is claimed here.

## Reproduction

Pinned image: `sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc` (Torch 2.13.0+cu130, Triton 3.7.1).

Offline compiler, with no GPU exposed and artifacts outside the source checkout:

```sh
CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=. \
  python3 probes/engine_ffn_packets_compile.py --output /out/compile
```

GPU qualification uses the existing fleet queue and an immutable source checkout:

```sh
bash bench/fleet.sh run --gpu --fleet --detach st-ffn-packets0914v2 10 \
  'PR895 same-weight packet FFN qualification' -- \
  bash probes/run_engine_probe.sh probes/engine_ffn_packets_check.py \
    --ranks /path/to/exact/st-ranks --samples 8 --output /cache/ffn-packets0914v2.json
```

Both earlier compact-KDA reservations retain their original source checkouts. This separate P implementation changes neither reservation into evidence for the new FFN path. M (decode-priority expert sharing) and I (execution image) remain follow-up designs.

The GPU reservation `st-ffn-packets0914v2`, ticket `17893163563368864`, was accepted at source `4e24cc3d` in the separate checkout above. [admission.json](admission.json) is a queued-state receipt, not a completed numerical or performance result. The first submission obtained no hold because main advanced to `70038de0`; rebasing retained both covered-query and FFN execution proofs before resubmission.

That reservation has now finished with a **720-minute queue timeout before GPU
execution**. It supplies no numerical or latency result. The current probe
records the failing row/phase and component/expert byte differences, and retains
both whole-FFN device and synchronized wall timings in each B/A/A/B arm.
