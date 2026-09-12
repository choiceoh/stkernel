# Standalone ST runtime

> 살아 있는 참조 — **독립 ST 런타임 이미지를 어떻게 짓는지. 빌드가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

Build from the pinned local fleet image with `bash engine/runtime/build.sh`.
The build performs no network downloads. `build.sh` checks the seed image ID
against `dependencies.json`, then binds that ID to the Dockerfile build.
`ST_IMAGE` selects the output tag; it does not select different dependencies.

The result owns the `engine` source and imports DeepGEMM directly. vLLM is
removed, including its overlay files. FlashInfer remains a library dependency
for CuTe support and JIT utilities; ST's own b12x package supplies the MoE API
and kernels.

Each image contains `/opt/st/runtime-manifest.json`. The build-time verifier
checks Python, CUDA and package versions, absence of vLLM, and all 879 hashes
in the extracted DeepGEMM provenance. It records the engine source tree's
SHA256 and individual file hashes. At deployment, run:

```bash
docker run --rm --gpus all --entrypoint python3 st-engine:glm53 \
  -m engine.runtime.verify --gpu
```

When mounting source at `/repo`, set both `-w /repo` and `-e PYTHONPATH=/repo`.
The verifier then identifies the mounted source, which can differ from the
manifest embedded in the image. Keep the returned manifest with the image ID
and validation results. The image's default command prints boot usage.

GLM requires rank files with metadata
`weight_layout=st-glm53-b12x-up-gate-v1`. Old gate/up rank files are refused
before device allocation. Each node needs its own rank, DFlash2 weights and a
metadata directory containing config, tokenizer and generation configuration.
`--ckpt-meta` and `--drafter-dir` reach all boot modes. The fleet order is
srv2, srv1, srv3, srv4 for ranks 0, 1, 2, 3.

Full-model admission releases clean pages of the selected rank/draft files,
then requires the arena plus a **12 GiB workspace ceiling** and **4 GiB OS
reserve** in immediately free host/device memory. Every rank must pass before
any rank allocates its arena. These are byte limits, not measured workspace
claims. KV remains an explicit budget. The native PyTorch allocator is capped
at its existing reserved bytes plus the arena and workspace ceiling. Its
fraction API only enforces that byte limit; it never chooses KV capacity.
Direct CUDA allocations such as NCCL are outside that allocator, so boot also
checks physical free memory.

Before HTTP admission, preparation exercises the largest legal prefill at the
start and end of KV capacity, then every target decode, DFlash proposal/context
update and greedy/stochastic sampling graph. A per-phase ledger records current
and peak allocated/reserved bytes and immediately free memory. Every TP rank
must pass each checkpoint. `--dump-dir/memory-rankN.json` contains the limits
and measured peaks; `ready` is set only after all preparation passes. Unseen
prefill tail shapes remain under the allocator ceiling. Other processes can
still consume the physical reserve after preparation.

Graphs are released before NCCL shutdown, and cleanup restores the previous
allocator limit. The loader releases consumed file-cache ranges after blocking
uploads. It never invokes a machine-wide cache flush. Small allocator tests
do not qualify the full-model workspace budget.

The need for the guard was exposed by the first full-model validation boot:
rank 0 failed the approximately 55.4 GiB arena allocation, and srv1/srv4 became
unreachable over SSH. Their exact failure state was not retrievable at the
time of that record. Access subsequently recovered; the new byte budget has
not been used to restart the production fleet. NVIDIA documents UMA memory reporting and buffer-cache
reclamation in its [Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
This source must not be described as a qualified full-model release until
successful boot and the complete quality/load gates are done.

See `measurements/st_engine_completion_20260911/README.md` for completed
numerical checks and the explicit remaining gates.
