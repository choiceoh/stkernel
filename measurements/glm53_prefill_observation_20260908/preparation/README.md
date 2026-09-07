# CPU preparation for serving attribution

Eleven tests run in the pinned image without GPU access (`--runtime runc`,
`--network none`, 4 GiB RAM, two CPU cores). The tests include actual PyTorch CPU
BF16 tensors; none are skipped in `pinned-cpu.log.gz`. `pinned-cpu.json` retains
the command, exit and exact source hashes. `image-api.json` separately checks
the image's worker-extension interface and callable B12x run signature.

This is implementation preparation, not direct TTFT, live routing evidence,
numerical model acceptance or a speedup. No GPU work is submitted. Private boot,
profiler-off request collection and all-rank trace transfer remain to be wired.
See `docs/GLM53_PREFILL_OBSERVATION.md`.
