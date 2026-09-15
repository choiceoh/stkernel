# Standalone ST runtime

> 살아 있는 참조 — **독립 ST 런타임 이미지를 어떻게 짓는지. 빌드가 바뀌면 여기도 바뀐다.** 여기가 틀리면 그건 버그다.

The default image uses CUDA **13.2.1**, PyTorch **2.13.0+cu132** and matching
ARM64 torchvision/torchcodec wheels. `cuda132.lock.json` pins every upgraded
wheel by URL, version and SHA256; the original fleet image remains the
immutable bootstrap source for the patched FlashInfer/CuTe/DeepGEMM libraries.

Prepare the runtime seed on one ARM64 node, then distribute that **same image
ID** to the other fleet nodes with `docker save` / `docker load`:

```bash
bash engine/runtime/build-seed.sh
# Downloads are verified before the GPU-free, --network none Docker build.
# ST_RUNTIME_ARTIFACTS selects the reusable download/build context directory.
```

`build-seed.sh` does not install anything on the host or start a GPU service.
The seed removes the old toolkit, installs the CUDA 13.2.1 compute SDK and
libraries, and exposes normal SDK linker names for NVIDIA's Python wheels.
It explicitly installs NVIDIA's SHA256-locked ARM64 cuSPARSELt wheel.
The unused cu130 torchaudio package is removed. DeepGEMM's extension and all
879 JIT headers keep their recorded provenance; its JIT uses the new CUDA_HOME.

CuTe DSL 4.6.2 requires **nvdisasm 13.3.73**, which is retained as a diagnostic
tool. It does not compile or execute kernels. NVCC, PTXAS, NVRTC, NVVM and
nvJitLink are **13.2.78**; CUDA's independently versioned libraries follow the
13.2.1 toolkit manifest. Triton's regular and Blackwell assembler paths both
select that PTXAS, and TileLang/DeepGEMM select the same CUDA_HOME. The existing
FlashInfer metadata asks for CuTe 4.7.0 while the fleet's patched library and
ST kernels pin 4.6.2. The official ARM64 cuSPARSELt wheel also declares the
nonstandard `manylinux2014_sbsa` tag inside its WHEEL metadata. `pip check`
reports both discrepancies; ST's actual compile/import checks are recorded
separately. The installed cuSPARSELt shared library is an AArch64 ELF binary.

Build the engine offline with `bash engine/runtime/build.sh`. It reads the
accepted seed ID from `dependencies.json`, refuses a different image, and binds
that immutable ID to the build. `ST_IMAGE` selects the output tag, not different
dependencies. A newly rebuilt seed must be validated and its ID updated in the
manifest before use; an arbitrary tag is not enough.

## The x86_64 check image

There is a second, unrelated image, and the distinction matters: `st-engine:glm53-sm120-x86`
is for **checks on an RTX 5050**, never for measurements, and it shares no layer with the
above. It cannot: the seed's parent `glm53:v13-b12x-it` is a locally built ARM64 vLLM image
with no registry digest and `ai.vllm.build.commit=unknown`, `promote_deep_gemm.py` lifts a
compiled extension out of it byte-for-byte, and `install_cuda132.py` refuses a non-aarch64
host. So the x86_64 side inherits nothing and names every package from an index:

```bash
python3 engine/runtime/make_x86_64_lock.py engine/runtime/cuda132.x86_64.lock.json
bash engine/runtime/build-x86_64.sh
```

`cuda132.x86_64.lock.json` holds 42 SHA256-pinned wheels; the cu132 index publishes no
digest, so the three torch wheels are fetched and hashed by the generator. Its `deviations`
field records the three entries that cannot match this lock -- `nvidia-cudla` dropped
(Tegra-only), `flashinfer-python` at the published `0.6.18.post1` rather than the fleet's
unpublished dev build, `tilelang` at `0.1.14`, the nearest version with an x86_64 wheel --
and the absence of DeepGEMM, which no x86_64 build can reproduce. Everything the vLLM
parent supplied on ARM64 and a base image does not -- 31 packages, 59 wheels, read from
every locked wheel's `Requires-Dist` -- is resolved into `closure.json` at fetch time, so
the build still runs `--network none`. Torch's own requirements are not the whole of it:
`flashinfer` needs `tvm_ffi`, and a closure built from torch alone yields an image that
installs cleanly and cannot import flashinfer. The image carries `TORCH_CUDA_ARCH_LIST=12.0`
and `CUTE_DSL_ARCH=sm_120`, no `engine` source, and no `runtime-manifest.json`: `verify.py`
describes the ARM64 runtime and does not apply to it. `engine/kernels/b12x`
(`@supported_compute_capability([120, 121])`, torch and flashinfer only) is what it can run;
every native lane is `-gencode arch=compute_121a` and `engine/kernels/cells.py` refuses a
device that is not a GB10. See `bench/OST_97X_LANE.md`.

**Validated once, on an RTX 5050 (2026-09-15).** The image builds, `torch 2.13.0+cu132`
survives installation with `cuda True` and capability 12.0, flashinfer imports and a bf16
matmul runs on the device -- `tools/ost-97x-selftest.sh` walks that whole chain from a
controller. Two things had to be fixed to get there and both are in the scripts above: a
closure resolved from torch's requirements alone yields an image that cannot import
flashinfer (`tvm_ffi`), and resolving it with pip's resolver ON lets pip replace the pinned
`torch-2.13.0+cu132` with a PyPI cu12 build. The build moves ~3 GB and writes ~18 GB; run it
when the box can spare that.

## The runtime manifest

The result owns the `engine` source and imports DeepGEMM directly. vLLM and its
overlay files are absent. Each image contains `/opt/st/runtime-manifest.json`.
The verifier checks Python, every locked package, actual compiler selection,
loaded CUDA runtime/NVRTC versions and library paths, the package lock, all
DeepGEMM hashes and the complete engine source identity. Production/served GLM
boots repeat the check before allocating the model.

```bash
docker run --rm --runtime=runc -e CUDA_VISIBLE_DEVICES= \
  --entrypoint python3 st-engine:glm53 -m engine.runtime.verify
```

For an authorized GPU check, omit `--runtime=runc` and `CUDA_VISIBLE_DEVICES=`,
then add `--gpus all` and the verifier's `--gpu` argument. CUDA 13.x minor
compatibility permits native cubins on R580+, but newer PTX requires a driver
that understands that PTX version. This migration targets native SM121a cubins;
GPU execution and the four-node consumer measurement remain separate evidence.
The host driver/toolkit and running containers are not changed by either build.

Native extension cache keys include the selected NVCC/PTXAS pair. CuTe's ST
source identity includes the runtime lock. Image and launcher defaults use
`/cache/cu132/` for CUDA, Triton, TileLang, DeepGEMM, FlashInfer and native build
artifacts so the migration cannot reuse the previous mixed-toolchain cache.

When mounting source at `/repo`, set both `-w /repo` and `-e PYTHONPATH=/repo`.
The verifier identifies the mounted source, which can differ from the embedded
manifest. Retain the returned manifest with the image ID and validation results.
The default image command prints boot usage.

GLM requires rank files with metadata
`weight_layout=st-glm53-b12x-up-gate-v1`. Old gate/up rank files are refused
before device allocation. Each node needs its own rank, DFlash2 weights and a
metadata directory containing config, tokenizer and generation configuration.
`--ckpt-meta` and `--drafter-dir` reach all boot modes. The fleet order is
srv2, srv1, srv3, srv4 for ranks 0, 1, 2, 3.

Full-model admission releases clean pages of the selected rank/draft files,
then requires the arena plus the **workspace ceiling** (`budget.WORKSPACE_GIB`,
12 GiB), the **OS reserve** (the box's SIGTERM line plus 1 GiB) and the prefix
tier's host cache in immediately free host/device memory. Every rank must pass
before any rank allocates its arena. The ceiling is a byte limit set from the
measured peaks: 9.67 GiB reserved by the largest prefill chunk at the end of the
served context (2026-09-14). A shape that spends more raises it with `--workspace-gib`
(`ST_WORKSPACE_GIB` in the launcher). KV remains an explicit budget. The native PyTorch allocator is capped
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
