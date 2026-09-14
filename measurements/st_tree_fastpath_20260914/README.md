# Tree fast path: CPU comparison and offline SM121 proof

The revised experiment removes repeated KDA reconstruction, per-node DSA
calls and W4A8 queue synchronization. **There is no measured GB10 serving
speedup or default-adoption verdict.** No GPU, fleet queue or server was used.

`identity.json` records the implementation commit and integration base.
`cpu-ab.json`, `mock.json` and `compile.json` contain checked source hashes.
The ordinary dense path still uses existing GPTQ W4Pack/E4M3 activations;
KDA state/factors stay FP32. Production defaults are unchanged by this PR.

## Matched CPU results

The baseline experiment is merged #947 (`b8e42ab9`). Both variants use the
same checkout's common target operators, fixed tiny weights and inputs,
torch 2.14.0 on one CPU thread. Each arm has three warmups and 80 timed
samples in two B/A/A/B brackets. No compilation/tests were run concurrently
by this task during the recorded timing run. All six target output hashes
match exactly; target head tokens match as well.

| Nodes | Prefix | #947 verify ms | Revised verify ms | Time reduction |
|---:|---:|---:|---:|---:|
| 8, chain | 0 | 2.823 | 1.665 | 41.0% |
| 8, chain | 129 | 3.015 | 1.688 | 44.0% |
| 8, chain | 1024 | 3.169 | 1.726 | 45.5% |
| 15, two branches | 0 | 4.405 | 2.116 | 52.0% |
| 15, two branches | 129 | 4.807 | 2.138 | 55.5% |
| 15, two branches | 1024 | 5.103 | 2.193 | 57.0% |

These timings include transaction preparation and target verification,
excluding proposal and commit. The 7x64 candidate selector separately goes
from 0.529 to 0.253 ms (flat scores) and 0.523 to 0.240 ms (peaked scores):
52.2–54.2% less CPU time. The peaked case retains a depth-seven chain within
eight selected nodes, versus depth four before; proposal mass rises from
2.7953 to 3.1581. **Proposal mass is not measured acceptance.**

The recorder also includes an ordinary linear target reference: 2.064–2.109
ms versus 1.686–1.715 ms for eight-node private tree verification (18.1–19.5%
less time). This is only a diagnostic CPU comparison: the linear call writes
its cache, while tree verification excludes accepted-path commit. Neither
arm is production CUDA-graph serving, and the tiny dense weights are CPU
reference operands, not a timing measurement of the new W4A8 GPU pipeline.

## Structural savings and correctness

- An eight-node chain needs 8 full KDA state updates instead of 36. DFS
  carries the existing FP32 state, with ancestor reconstruction on backtracking.
- One convolution launch reads only the final four taps. Topology uploads
  are shared across all layers.
- One DSA indexer batch and one sparse-MLA batch replace per-node calls.
  Completed pools are compressed once and reused at commit. Logical top-k
  columns and physical counts preserve tie sets; sibling pools remain invisible.
- At M=16, H=4096, I=3072, W4A8 workspace falls from 6,473,456 to 181,760
  bytes. Eliminating partial writes and reads removes 12,582,912 bytes of
  workspace traffic per MLP invocation. This is allocation/traffic arithmetic,
  not measured process memory or DRAM traffic.
- Static FC1/FC2 stream ordering removes worker polling, counters and the
  per-layer host completion vote. The prepared buffer supports low-level
  MLP capture; full tree-step capture is still absent.
- Candidate transfer payload falls from 118,272 to 21,512 bytes for a 7x64,
  width-two lattice. Exact FP32 bits and integer tie keys preserve ordering.
- Commit transfers observed MoE routes once for all layers instead of one
  device-to-host synchronization per layer.

`mock.json` / `tests.txt`: **87 tests, 79 passed, 8 explicitly skipped**.
This includes branch-vs-linear continuation, pool boundaries, tied top-k,
FP32 state, packed W4A8 numerical twins, owner/budget guards and four-rank
CPU agreement. Opt-in device tests cover carry/reconstruction equality and
W4A8 graph replay with changed inputs; those device tests were not executed.

`compile.json`: **31 offline SM121 variants**. Both carry modes and FP32/BF16
convolution weights compile. W4A8 keeps native FP8 MMA; the two staged
kernels contain no atomics, polling or local-memory spill loads/stores.
`interpreter.json`: **12 CPU interpreter checks**, including the actual tree
convolution ancestry/FP32 sum and native packed readers. libdevice activation,
device conversion, warp scheduling and graph replay remain untested on GPU.

`engine-check.txt`: local full engine suite, **202 modules / 1,871 tests,
370 skipped, no failures or missing modules**, before integration rebase.
The final PR head additionally requires the normal engine CI.

## Reproduce

From a clean checkout containing #947 in Git history, with torch CPU,
Triton, numpy and safetensors:

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_dataflow_mock.py --output /tmp/tree-fastpath
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_dataflow_compile.py --output /tmp/tree-fastpath/compile
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 PYTHONPATH=. python probes/engine_tree_dataflow_interpreter.py --output /tmp/tree-fastpath/interpreter.json
# Run timings after other checks finish.
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python probes/engine_tree_fastpath_bench.py --iterations 20 --output /tmp/tree-fastpath/cpu-ab.json
```

Adoption still needs same-build, real-weight GB10 numerical and graph proof,
ordinary-serving versus candidate C=1/C=4 tok/s and TTFT, acceptance/quality,
and 32K/128K memory coverage, with warmup/compile excluded from timing.
