Matched source407271d9, GMU0.60, B9/A9 only. Original two-row prefixSHA039d91324a657eae023147143a5faedfc2110722b34e0174c55dba5011acf20a.

| Metric | B9 | A9 | Change |
|---|---:|---:|---:|
| Fixed1024×3 pooled decode tok/s | 76.54037923 | 72.50971775 | -5.2661% |
| Fixed-window pooled engine step/s | 21.53829841 | 21.41214855 | -0.5857% |
| 2000 context prefill tok/s | 2698.85698 | 2713.23515 | +0.5328% |
| 32000 context prefill tok/s | 2630.90837 | 2927.53536 | +11.2747% |
| 128000 context prefill tok/s | 3150.83416 | 3125.64537 | -0.7994% |

A fails the absolute76tok/s target. Both canonical fact gates18/18 and Korean0/8, with all channel gated counts0; A proof4/4 and B3/3. All8 recorded request hashes match; output hashes are compared individually in audit.json. Full request/SSE text is unavailable here, so those content hashes and factual correctness are not independently regenerated. The original row/file hashes and all rate arithmetic are recomputed.

All four ranks have identical Cmd/Entrypoint/Image/mounted-source identities; full environment dictionaries differ only at VLLM_GLM53_EP_HYBRID_Q0_DUAL_WARP0→1. Both have actual GMU0.60/1056blocks/maxlen1048576/maxseq4/maxbatch8192. Source-bound bench/proof.py independently accepts all four A ready logs' hybrid and dual-warp composites;12cases/72candidate phases perrank and selected actual artifact/launch markers are preserved. Ready markers may include profile calls; completed canonical rows also report their execution proofs.

32K and128K each have one combined request per arm. Their cold_s/warm_s fields repeat one TTFT and cannot support a variance, consistent-gain or no-JIT claim. Fixed tok/s uses3069post-first tokens over summed decode seconds; engine step/s uses summed valid fixed-window steps/seconds. No olderGMU.6329 baseline is mixed in. No HTTP/GPU/test/source change was made.
