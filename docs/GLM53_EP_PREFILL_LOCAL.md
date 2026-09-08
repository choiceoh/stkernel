# Full-token expert-local prefill

Experimental `VLLM_GLM53_EP_PREFILL_LOCAL=1` plus `ENABLE_EP=1` changes the
MoE execution geometry from 288 experts with I512 TP shards per GPU to 72
complete I2048 experts per GPU. Attention retains TP4. The exact EP4/DP1,
non-transformed b12x runner still has one late TP sum, so MHC token sharding
can continue through the existing all-gather/reduce-scatter path.

The new M128 producer consumes original [T,4096] activations and remapped
[T,8] routes. Remote sentinel IDs and zero weights are excluded before any
expert-indexed address. Each warp compacts at most eight local routes into
its existing shared cache. It reads/quantizes original token blocks and
scatters to the original output rows. The inherited FC1/Q1/FC2 code and task
publication protocol are pinned to the image's gated-source SHA-256.

This removes the existing EP prefill path's GPU nonzero/host count boundary,
expanded pair_x/pair_out, pair-list chunking and external index_add. It does
not remove the model's arithmetic, TP communication or the router remap.
Input/scale/row scratch must still cover worst-case concentrated routing;
no assumed balanced distribution reduces buffer capacity. Extra padding,
wide-I task splitting and complete serving memory remain measurement topics.

The candidate is restricted to eager SM121 E72/H4096/I2048/top8 M128,
SwiGLU-OAI (1,0,10), row-major NVFP4 and 4096..16384 executed rows.
Unsupported shapes retain existing EP paths. Eligible calls reject inherited
source drift or incompatible forced backends before submitting invalid IDs
to a stock kernel. A distinct cache suffix separates candidate artifacts.
`ENABLE_EP` and the candidate both remain off by default. EP also changes
the decode layout, which must pass its own existing-quality/latency checks.

The historical September 7 profile assigns 28–32% of occupied prefill time
to MoE. It is source context, not a current critical-path fraction or a
forecast. Global useful FLOPs remain unchanged by TP-to-EP repartitioning;
75% fewer experts per rank is not a 75% speedup. The 1.40x direct prefill
throughput objective remains open.

CPU admission/cache/SP/wrapper tests are implemented, as is an immutable-image,
no-device CuTe compiler runner. GPU correctness and sanitizer checks must
compare full-token output with the existing E72 compact path using identical
weights, balanced/concentrated/empty-local routes, odd tails and changed
inputs. TP4 wire numerics and current-capacity fresh 2K/32K/128K TTFT, output
quality, decode and memory checks follow before any default recommendation.

A preceding two-stripe prototype was withdrawn before GPU submission after
finding preserved failures on branch `codex/glm53-prefill-moe-overlap`
(`021131d`). It is saved only on local branch
`codex/glm53-prefill-moe-pipeline` at `15b7bd0`. Do not queue it.
