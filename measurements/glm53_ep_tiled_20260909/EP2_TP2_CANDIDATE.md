# Routed EP2 / TP2 candidate

This candidate changes only the main GLM routed-expert layout. It is opt-in via
`VLLM_GLM53_EP_HYBRID_TP2=1`; the profile remains `0`. Rejected decode option v5
must remain `VLLM_GLM53_EP_DECODE_OPT=0` in both arms.

Physical ranks 0/1 hold experts 0–143, and ranks 2/3 hold experts 144–287.
Rank parity selects intermediate channels 0–1023 or 1024–2047. The existing
compressed-tensors loader allocates E144/I1024 instead of E72/I2048. Attention
and shared experts retain physical TP4, as does the existing final output sum.
There is no added TP2 collective. SF6 and DFlash K5 remain enabled.

The motivation is less rank imbalance for a short request: idealized native
work changes from `16 * max(U0, U1, U2, U3)` to
`8 * max(U0 + U1, U2 + U3)`, where each U is the number of active experts in an
original rank. This is a structural bound, not a measured routing distribution
or speedup prediction. Pairing also duplicates routing/input preparation, and
can worsen prefill. Changed BF16 partial-sum grouping can affect output and
speculative acceptance. A rank-local stock oracle does not validate the full
distributed sum.

Per-rank routed weight storage remains 864 MiB and SF6 storage remains
81.84375 MiB. The shared static Q0 workspace increases by 40.5 MiB; the
8192-token dynamic Q0 padding bound adds 20.25 MiB. These figures exclude other
metadata and allocator effects and are not a complete device-memory forecast.

The loader accepts only the observed compressed-tensors NVFP4 B12X runtime,
with fifteen exact runtime source pins and an immutable per-rank identity.
Before allocation it validates the actual selected quantization method. The
initial input-scale tensors are local E-by-2/E tensors; they are not global
288-expert ModelOpt tensors. Actual-source symbolic checks show that the local
input maximum can change while current B12X Q0/FC1/FC2 alpha normalization
remains one. B12X defers input quantization until its owner consumes BF16.
These checks do not execute FP8 casts, CUDA arithmetic, or distributed sums.

The new portable host suite has 34 passing tests: geometry 10, owner 9, remap 3,
proof 4 and actual-loader contracts 8. A local attempt to run all 217 tests
could not import the two existing torch-dependent suites because the macOS
Python environment has no torch; the full suite belongs to the normal
no-device image CPU gate. No GPU result is implied by these host checks.

Acceptance requires fresh same-source baseline and candidate CPU lowerings,
the full CPU suite, four-rank startup canaries, and the canonical direct
onepass B/A. Both arms use SF6/K5/PREP1 and decode option 0; only hybrid routing
changes. Retain all fixed-1024 decode repetitions, output hashes, Korean and
factual-quality gates, plus 2K/32K/128K prefill tok/s and TTFT. The decode goal
is at least 76 pooled tok/s across three repetitions while preserving prefill
and quality. No default or performance acceptance has been established.
