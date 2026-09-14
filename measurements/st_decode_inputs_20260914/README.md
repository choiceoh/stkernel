# Draft input and keyed-draw fusion

The normal CUDA draft path now prepares anchor/mask IDs and absolute positions in one kernel. It covers batched proposal, one-row greedy proposal and one-row sampled proposal. IDs and positions occupy disjoint slices of one allocation. Device positions are read on replay; a one-row host integer position becomes a compile-time scalar without a host-to-device tensor copy.

`base.draws.step_block` now computes its existing SplitMix64 recipe in one CUDA kernel: seed, admission nonce, generation count, draft/verify/fresh purpose and position keep the same meaning. Arithmetic wraps at 64 bits; the upper 53 bits convert to FP64, scale by 2^-53, then round to FP32 exactly as the reference specifies. The original CPU tensor recipe and independent Python integer oracle are retained. No generator state, seed change or new draw consumption is introduced.

The batched sampled walk's probability buffer uses `empty` because every position writes every row/candidate before returning it to verification. A poisoned-allocation test checks that coverage. Candidate-support cloning remains, as do softmax, selector arithmetic and verification.

All three changes are active by default. K=7, model precision and cache/state ownership are unchanged. No persistent workspace is added.

## Work inventory, not measured serving speed

For K=7, both C1 and C4, the retained CPU reference dispatches:

| Path | Tensor operations producing storage | Candidate CUDA launches |
| --- | ---: | ---: |
| Anchor/mask IDs plus positions | 4 | 1 |
| Keyed step draw block | 53 | 1 |

The inventory excludes view/alias operations. It is a CPU dispatch count, not a measurement of GPU launches, DRAM traffic or latency. The sampled probability buffer also avoids one zero-fill operation. Greedy proposal benefits from input preparation; sampled/verification callers of `step_block` additionally use the draw fusion. Do not credit all of those operations to a purely greedy step. See [work-count.json](work-count.json).

## Validation

- Existing ST image `sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`, runc, no network, 2 CPUs/4 GiB, GPUs hidden: **48 tests, 39 passed and 9 GPU-only skips**. This includes the tiny real drafter's greedy/sample/verification comparisons and the poisoned probability buffer. [CPU log](cpu-tests.log).
- Actual Triton kernels in CPU interpreter: **43 exact comparisons**. Input tests cover C1–C4, strided inputs, scalar/device positions, K=0/1/5/7/31, 128K context, negative IDs and int64 overflow. Draw tests compare both the tensor recipe and independent Python integer oracle for signed 64-bit boundary keys, negative seeds, seeds wider than 64 bits, and all three draw purposes. Output guards remain untouched. [Interpreter evidence](interpreter.json).
- Actual SM121 compilation: **11 variants passed**, all with zero shared memory. Six input variants cover token widths 1/8/32 and host/device positions; five draw variants cover K=0/1/5/7/31. No CUDA context initialized. [Native compile evidence](compile.json).
- Added CUDA graph replay checks keep seed fixed while changing strided anchors, positions, nonces and generation counts, and compare every output bit to the CPU reference. These GPU tests have not run.

No GPU queue submission, model boot, image build or GPU execution was used. GPU bit identity/replay, acceptance, quality, actual step/s and tok/s remain pending. CPU identity and native compilation are separate evidence from those consumer metrics. [Manifest/source identity](manifest.json).

## Reproduce without GPUs

In the existing ST image, mount the checkout at `/work` and an output directory at `/out`. Use `--runtime=runc --network=none --cpus=2 --memory=4g --pids-limit=256`, `CUDA_VISIBLE_DEVICES=`, `NVIDIA_VISIBLE_DEVICES=void`, and one thread each for OMP/OpenBLAS/MKL.

```sh
python3 -m unittest tests.test_engine_decode_inputs tests.test_engine_draws tests.test_engine_drafter tests.test_engine_draft_tuning_integration tests.test_engine_sampling -v
TRITON_INTERPRET=1 python3 probes/engine_decode_inputs_check.py --mode interpreter --output /out/interpreter.json
python3 probes/engine_decode_inputs_check.py --mode compile --output /out/compile.json
python3 probes/engine_decode_inputs_ops.py --output /out/work-count.json
```
