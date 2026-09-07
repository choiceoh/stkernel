# First overlap GPU attempt: harness initialization failure

Normal fleet session `moeoverlap10907` acquired at 22:46:17 KST. All four
source/memory checks passed and probe processes started at 22:47:07. All
ranks failed before weight creation, MoE numerical checks or timing:
`initialize_model_parallel` accepts `pipeline_model_parallel_size`, but the
probe passed `pipeline_parallel_size`. This is a probe API error, not a
numerical or performance rejection of the candidate.

The probe exited at 22:47:33. Exact original container IDs, image, config,
host config, mounts, overlay hashes and manifest were restored on all four
nodes, with public port 8000 and health 200 verified by 22:50:21. The
completion receipt records exit 1 and restored_original true. The next
holder `inputctaserve0907` acquired at 22:50:28; our recovery proof is for
22:50:21, not a claim about that later holder's changing service state.

The correction uses the actual mounted source's parameter name. A CPU
preflight binds every distributed lifecycle call against signatures read
from that source and catches the original typo without importing CUDA.
It now runs on all ranks before any stop/start. Cleanup also encloses
partial world/model initialization so an initialization exception still
calls both model-parallel and world cleanup. Three CPU tests cover the
real signatures, original typo rejection, and init/body/teardown failures.

No numerical, GPU-overlap, kernel speed or serving TTFT result exists from
this attempt. The original frozen source44d76c0 and per-rank logs are kept.
