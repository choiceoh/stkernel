# ST completion work — numerical validation and remaining fleet gate

Base: merged PR #541, commit `a5b68c846a9f200ecafd56c8a16e2a414b393eab`.
The changes connect actual GLM decode to CUDA Graphs, capture DFlash2 and
sampling, fix BF16 scatter instability in the fixed GLM TP4 MoE lane, pass
alternate model paths throughout boot and verify standalone runtime identity.
Full-model service qualification is **incomplete**; the first boot failed
before inference. No full-model quality or ITL pass is claimed.

## Numerical results

The previously reported 1.55% was maximum absolute hidden-output difference
divided by the reference tensor's maximum absolute value. It was measured on
two real layers, not token error rate, perplexity or end-to-end quality.

After FP32 scatter accumulation, the real four-node KDA/DSA/MoE test passes
17 cases on each rank: target widths 1 and 6, reversed request/state-slot
ownership, contexts through 4,096, a graph capacity transition, and rejection
of five speculative positions followed by overwrite. Every measured hidden
output is identical to eager, and state/paged-cache bytes are identical.
All four containers exit 0 and are removed after logs are collected.

`probes/engine_moe_real_check.py` measures layer 3's real rank-0 weights,
E=288, hidden 4096, intermediate 512, top-8. It covers 3 seeds, six token
counts (1, 4, 6, 12, 18, 24), shared routing and mixed routing over 16 experts,
including a zero route weight. Each condition has 64 eager and 64 graph
repeats. All 36 conditions have zero native repeat and graph/eager difference.
Maximum error against an independent PTX-reciprocal/FP8-rounding/dequant/GEMM
oracle is 0.6623%. The original FlashInfer implementation remains a diagnostic
control; its worst repeat spread is 17.3653% in this run. Its averaged results
are reported without using the average to conceal individual outliers.

The kernel retains rounding of each weighted route contribution to BF16,
accumulates these contributions in FP32, and casts the result to BF16 once.
The fixed GLM TP4 micro and row-major static lanes use preallocated FP32
scatter planes. Other model shapes and the byte-pinned dynamic prefill kernel
are unchanged. Consequently, these results do not qualify large dynamic
prefill shapes or every real layer/expert distribution.

The division-based PyTorch oracle and the hardware reciprocal/FP8 sequence
produce different FP4 activation bytes at rounding thresholds. That explains
why the former shows larger errors despite no scale-byte changes. The
hardware oracle is independent Triton/PTX plus dequantized GEMM; it is a
numerical diagnostic, not a new serving fallback. The native per-run oracle
gate remains 2%, and native/graph repeat spread is separately bounded by 0.1%.

Timings are retained in the raw MoE report. They are isolated operator stream
timings, with Python launch gaps for eager. Part of this run overlapped fleet
validation, so they do not establish an engine speedup or regression band.
FP32 scatter has an additional accumulation/cast cost.

Real DFlash2 weights pass 7 eager/graph proposal comparisons and 42 accepted
prefix cache comparisons across two physical slots and the 2,048-token ring
boundary. Padding the live attention context to a fixed, masked window makes
reduction geometry consistent for capture. An independent FP64 SDPA oracle
using only live context rows bounds attention error to 0.5587% in these cases.
This probe borrows an isolated target rank's embed/head and does not measure
full-model draft acceptance. A long prefill now writes each drafter ring cell
once, preserving only the latest window.

The GPU unit suite before the later memory-admission guard passes 100 tests
with zero skips. Greedy sampling preserves RNG state, while mixed/stochastic
sampling matches eager outputs and ending generator state. The later guard
adds two CPU admission tests; the updated GPU suite passes 102 tests with zero
skips (`gpu-tests-admission.log`). After releasing intermediate state views
between graph captures, a single-GPU isolated rank rerun passes all 17 graph
cases with identical outputs/cache bytes (`graph-pool-final.log`). That later
memory-lifetime change still needs a fresh full TP4 boot.
`numerical-source-sha256.json` identifies the numerical files used by the
successful probes. Boot/admission and documentation changed afterward.

## Weight preparation and disk cleanup

Presharding all 45 target layers completed: 176.09 GiB read, four 44.50 GiB
rank files written in 941 seconds. The new files are at
`/home/choiceoh/models/st-glm53-9391-up-gate-full/rank{r}of4.safetensors`.
srv4 holds all four; srv2, srv1 and srv3 hold ranks 0, 1 and 2 respectively.
Each file is 47,786,460,680 bytes. The fanout completed using rsync's transfer
checksums; a separate persisted full-file SHA256 inventory remains pending.
These files have the required up/gate layout metadata. Original checkpoints
and older rank files are preserved.

On srv1, 23,247 regenerable startup-cache `.pt` files under
`/home/choiceoh/glm53-cache/glm53-fp8` were deleted after checking that the loader
recomputes them and that no running service consumes the directory. This
freed 231,814,893,568 bytes (about 216 GiB). Original model weights, containers
and volumes were retained. After the new rank and slice copies, the last
successful inspection showed approximately 193 GiB disk space available.

## Full-model boot failure and recovery status

The private `st-full-completion-9391` containers attempted a full TP4 boot at
10:42 UTC. Rank 0 on srv2 failed `torch.empty` for the approximately 55.4 GiB
arena before loading weights. It exited 1, with `OOMKilled=false`; the CUDA
driver logged allocation failures. Rank 2 lost its NCCL store and was stopped.
Both logs are retained. srv1 and srv4 remained pingable but SSH did not return
a usable session via either their Tailscale addresses or the private network.
Stop commands targeted only this task's named containers; their completion
on srv1/srv4 could not be confirmed. No server reboot, production-service stop
or global cache flush was performed.

The follow-up admission guard releases clean file-cache pages for the selected
rank/draft paths, checks immediately free memory plus 16 GiB spare, and uses a
TP vote to prevent peers allocating after one rank fails admission. The loader
also advises consumed file ranges away after each blocking upload. Unit tests
cover a misleadingly high MemAvailable value and ensure weight bytes survive
cache reclamation. This guard has not yet passed a fresh full fleet boot.

The current standalone image was also built and verified on srv2, with all
879 DeepGEMM files intact, no vLLM, the pinned package ABI and the expected GB10
device (`runtime-final-srv2.json`). Image ID:
`sha256:67a9a329cd5da43f44aa9761ef70e8f05ab0329306d6dabe3d5b1b3db59b262b`.
This verifies packaging and device identity, not successful full-model serving.

Remaining release gates:

- Recover srv1/srv4 access and confirm private test-container cleanup.
- Boot the full model with the guard, validate Korean retrieval at 2K/32K/128K,
  corruption, real DFlash2 acceptance and quality relative to the legacy stack.
- Measure repeated full-model decode and concurrent/tier traffic with a
  reproducible baseline; verify NVMe continuation bytes and output behavior.
- Persist full rank hashes, promote the new rank paths, and build/verify the
  final standalone image on every node. The intermediate image and the tested
  two-layer graph path do not satisfy these release gates.

## Reproduction

The probes require the pinned standalone image and mounted current source:
`-w /repo -e PYTHONPATH=/repo`, rank files at `/ranks`, metadata at `/repo/meta`
and DFlash2 at `/draft`. The included fleet executors preserve the tested
rank order, private ports and RoCE settings. They are evidence copies with
explicit task paths/container names, not production launchers.

```bash
python3 probes/engine_moe_real_check.py --ranks /ranks
python3 probes/engine_decode_graph_check.py --distributed --ranks /ranks --ckpt-meta /repo/meta
python3 probes/engine_drafter_graph_check.py --ranks /ranks --ckpt-meta /repo/meta --drafter-dir /draft
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
```

Do not repeat the full-model boot on the unresponsive hosts. Recovery and
memory preflight are prerequisites. `probes/engine_full_check.py` supplies
an ST transport adapter for canonical Korean documents, a target-only eager
comparison, graph/DFlash execution, HTTP concurrency and tier timing; it has
not reached its inference tests in the failed boot.

## Integration with the later main branch

The candidate also incorporates main `d4528459` (PRs #542/#543 and their cache
metadata/indexer work). Graph caches expose the fused indexer's block mapping
contract. The integrated source passes the isolated-rank 17-case real-weight
graph probe with exact eager outputs/state/cache bytes. All 117 GPU unit tests pass
with zero skips (`gpu-tests-merged.log`). The rebuilt image also passes ABI/
provenance/device verification (`runtime-merged-srv2.json`), and its complete
engine source manifest matches the integrated working tree. This integration
has not been rerun on full TP4
because srv1/srv4 remain unreachable. Historical numerical manifests above
identify the pre-integration files; the integrated source manifest is separate.
