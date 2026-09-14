# Rank-local token embedding fusion

GLM target and drafter share `Glm53Net.embed`. Its CUDA path now combines rank-local ID subtraction, ownership checks, safe indexing, embedding lookup and nonlocal zeroing in one kernel. The existing TP all-reduce and `embed` graph label are retained. The ordinary target prefill also reaches this same entry point.

The kernel loads/stores BF16 payloads as uint16 without floating-point conversion. Owned values retain their bits, including signed zeros and NaN payloads; nonlocal IDs produce positive zero. Foreign IDs do not read weight row zero. Input ID strides, padded weight-row pitches and ragged hidden widths are explicit kernel parameters. Invalid dtype/layout metadata fails before launch, and empty ID vectors return an empty output without a kernel launch. The normal CUDA path is enabled by default.

## Work inventory, not measured speed

For eight decode rows, 32 decode rows and a 2,304-token prefill at hidden width 4,096, the retained CPU reference executes seven tensor operations producing storage: subtraction, two comparisons, OR, ID masking, embedding, and output masking. The candidate executes one CUDA kernel. The inventory excludes view/alias operations and the unchanged collective. This is not a GPU launch count or throughput measurement. See [work-count.json](work-count.json).

Both target and draft embedding use this path during an ordinary speculative cycle, including greedy generation. It removes the intermediate embedding result and masks rather than adding persistent weights, KV state or workspace. Actual graph-pool peak memory and serving speed remain unmeasured.

## Validation

- Existing ST image `sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`, CPU-only runc, no network, 2 CPUs/4 GiB: **67 tests, 32 passed and 35 GPU-only skips**. Tests cover all 65,536 BF16 bit patterns in the CPU reference, rank boundaries, strided storage, one collective per network call, four-rank ownership assembly, bad metadata rejection and empty inputs. [CPU log](cpu-tests.log).
- Actual Triton kernel in the CPU interpreter: **28 exact byte comparisons**, spanning four ranks, 1/8/16/24/32/65/257 rows, ragged and full hidden widths, strided IDs/weights, invalid and int64-boundary IDs, untouched weights and guarded output storage. Raw int16 tensors feed its uint16 reads/writes to avoid the interpreter's BF16-to-FP32 argument adaptation. [Interpreter evidence](interpreter.json).
- **Eight SM121 native variants compiled**, including the production 38,720-row vocabulary shard and hidden width 4,096. All use zero shared memory; no CUDA context initialized. [Compile evidence](compile.json).
- Added deferred CUDA checks compare BF16 output bits during graph replay as IDs and weight payloads change, including ragged shapes. They have not run.

No GPU queue, model boot, image build or GPU execution was used. GPU byte identity/replay, output quality, acceptance, actual step/s and tok/s remain pending. [Manifest and source identity](manifest.json).

## Reproduce without GPUs

In the existing ST image, mount the checkout at `/work` and an output directory at `/out`. Use `--runtime=runc --network=none --cpus=2 --memory=4g --pids-limit=256`, `CUDA_VISIBLE_DEVICES=`, `NVIDIA_VISIBLE_DEVICES=void`, and one thread each for OMP/OpenBLAS/MKL.

```sh
python3 -m unittest tests.test_engine_token_embedding tests.test_engine_glm53 tests.test_engine_drafter tests.test_engine_decode_inputs tests.test_engine_decode_buffers -v
TRITON_INTERPRET=1 python3 probes/engine_token_embedding_check.py --mode interpreter --output /out/interpreter.json
python3 probes/engine_token_embedding_check.py --mode compile --output /out/compile.json
python3 probes/engine_token_embedding_check.py --mode inventory --output /out/work-count.json
```
