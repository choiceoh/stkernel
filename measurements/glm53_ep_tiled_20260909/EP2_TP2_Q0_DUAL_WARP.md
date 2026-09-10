# EP2/TP2 dual-warp Q0 candidate

The preceding same-source onepass reached 78.93348 decode tok/s with EP2/TP2,
but lost 5.21% / 3.26% prefill throughput versus EP4 at 32K / 128K. The next
experiment keeps the EP2/TP2 weight layout and changes only its dynamic Q0
producer. These previous numbers motivate the work; they do not substitute
for the next same-source EP2/TP2 baseline.

`VLLM_GLM53_EP_HYBRID_Q0_DUAL_WARP=1` is opt-in and requires the exact
E144/I1024 tiled SF6 owner with `VLLM_GLM53_EP_HYBRID_TP2=1`. Both flags remain
off in the public profile. The static decode kernels, weight allocation,
FP4 quantization functions, SF6 layout, task publisher and FP32 output scatter
are retained. The previously rejected decode optimization stays off.

The M128/N128 Q0 staging area fits four H4096 BF16 tokens. The baseline uses
one math warp per token and leaves four of its eight math warps idle during
this phase. The candidate uses two warps per token, each processing four of
the token's eight 32-block SF iterations. Only the even warp's lane zero
allocates routes. A CTA barrier outside all token/warp guards publishes its
metadata to both consumers, including partial batches; the asynchronous
input-copy wait and subsequent batch/final fences remain separate.

This adds no shared allocation, global buffer, input copy or kernel launch.
It does add one CTA synchronization per four-token batch. The quantized-row
fanout and its bytes do not decrease: two warps share the work. A speedup is
not implied by the doubled active warp count. Register pressure, shared-memory
instruction order and the synchronization cost still need actual lowering
and GPU evidence.

The owner seals the selected mode before relayout, gives it a separate
workspace/prewarm identity and supplies an explicit bool to the dynamic
dispatcher. The candidate appends `glm53_ep2tp2_q0_dual_warp_v1` to the existing
21-field hybrid dynamic cache key; native keys remain unchanged. Startup
schema2 retains all original numerical/graph/input/scale checks and binds the
candidate's 22-field dynamic keys. A dedicated serving marker is emitted only
after the actual prefill call outside canary/capture.

The no-device CPU gate retains the previous eight lowerings and adds one
E144/I1024 candidate lowering, with explicit false/true selectors for the
hybrid dynamic pair. It registers 239 tests. Local macOS cannot import the 18
Torch-dependent tests; the full image gate is required. Compiling and host
contracts do not establish GPU numerical correctness or performance.

The planned canonical onepass uses SF6/K5/PREP1/OPT0 and HYBRID1 in both arms;
only A sets Q0_DUAL_WARP1. It retains fixed1024 decode ×3, all output/request
hashes, factual/Korean checks and 2K/32K/128K prefill tok/s and TTFT. Public
defaults and the earlier canonical verdicts do not change while this remains
unvalidated.
