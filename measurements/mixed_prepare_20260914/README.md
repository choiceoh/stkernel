# M2 preparation: one route index, value check and metadata upload

The previous packed implementation still sorted cold routes twice, synchronized
separate value reductions and uploaded each metadata table separately. Joining
that work reduces warm preparation/admission medians by **52.7–73.5%**, decode
ready latency by **50.5–62.2%**, and complete FFN latency by **15.7–27.4%** at hot
quota 128 in this same-build, one-GB10 comparison. This is an eager FFN component
result; the mixed path has no serving selector and PR #895 remains draft.

## Implementation and comparison boundary

Source `153b6f8bdbb9ebf18af78c79bb9173f96a6769c2` includes main `2ac7de6f`,
including #920 FC1 input reuse, #921 direct decode tail/pool-cache writes,
#922 templates and #923 compact input/separate FC2 staging. Both timing arms
use the same current compute kernels, router, input tensors and shared packs.

- `prepare_routes` uses one stable uint16 grouping of the fixed 288-expert
  domain to construct hot and cold descriptors. Cold sources are built directly
  in int32. Original token/slot order, whole-tail selection, quotas, M128 padding
  and task windows match both retained reference planners. Every invocation
  plans fresh routes; no histogram, pointer or generation cache is introduced.
- One Triton reduction checks all BF16 source values, FP32 route weights and
  four positive finite expert-scale planes, followed by one scalar readback.
  Shape/device/eager guards remain before dispatch. Runtime D/P extents avoid
  arrival-length specializations. NaN, infinities and scale error precedence
  retain the previous contract.
- `MixedMetadata` packs tables into one owned pinned int32 buffer and performs
  one asynchronous upload. Every table starts on a 16-byte boundary required
  by CuTe. GPU views share the packet's version counter; the hot source view
  remains in the owner's mutation guard. Pinned and device storage live with
  the invocation until its existing reader and consumer fences retire.
  Workspace allocation and padding still occur on every invocation.

The probe's **packed_v1** arm selects the previous two-stage packed planners,
Torch finite/positive reductions, and individual blocking metadata uploads.
**packed_v2** selects the joined planner, fused check and single upload. Both
use the complete binary descriptor digest and the common current owner
lifecycle. This compares previous preparation components in one build, not a
byte-identical historical constructor: planning precedes allocation in both
arms, and both share the current metadata extraction and hot-row device cast.
The native FFN supplies numerical references, not the timing baseline.

The implementation uses existing NumPy/Triton. This mixed route path adds no
Mojo runtime dependency. The separate host-commit Mojo experiment imported with
main is documented in [its own evidence](../../bench/mojo_host/README.md);
it is a different function and is not evidence about this array-planning path.

## Real-weight GPU result

Canonical reservation `st-mixed-prepare0914v2`, ticket `17893566183296004`,
succeeded in **95.9 seconds**. The frozen checkout is
`/home/choiceoh/st-worktrees/codex-gb10-mixed-prepare3`. The immutable GPU image is
`sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed`:
Torch 2.13.0+cu130 / CUDA 13.0 on srv4, one GB10, actual L3 rank 3-of-4 weights.
`Comm(world_size=1)` is the identity collective, not TP4 NCCL. Synthetic
normalized FFN activations use the actual profile router; both arms read the
same RTN shared W4/FP8 packs from the checkpoint, without a GPTQ store.

Four D=8/32 × P=9240/32768 cells × quotas 0/128 × two arms × four samples give
**64 complete measurements**. Arm and quota order alternate. Every sample
creates fresh plans, GPU storage and admission. The table uses quota 128 and
three warm samples per arm/cell, conservatively excluding the first sample.
No cProfile samples enter these timings.

| D rows | P rows | Prepare/admit v1 → v2 | Reduction | Decode ready v1 → v2 | Full FFN completion v1 → v2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 9240 | 40.57 → 10.76 ms | 73.5% | 41.99 → 15.88 ms | 72.52 → 61.16 ms |
| 32 | 9240 | 23.37 → 9.28 ms | 60.3% | 26.36 → 13.04 ms | 52.78 → 38.33 ms |
| 8 | 32768 | 33.23 → 15.71 ms | 52.7% | 34.42 → 16.91 ms | 97.85 → 78.72 ms |
| 32 | 32768 | 50.10 → 19.29 ms | 61.5% | 55.98 → 27.01 ms | 178.49 → 146.37 ms |

All times start before actual routing and fresh preparation. Host prepare time
can leave padding and copies queued; decode-ready and complete times synchronize
their output events and include those writes. For D8/P9240, the median interval
after preparation increases from 1.43 to 4.84 ms, while total decode-ready time
still falls from 41.99 to 15.88 ms. Preparation savings must not be mistaken for
an equal compute-kernel speedup. Quota 0 also improves all three per-cell medians;
its complete raw samples and summary are retained.

The run shares the GPU with production and has visible timing variation,
especially in individual readbacks and the D32/P32768 completion samples. These
are descriptive medians from a small alternating comparison, not confidence
intervals. Do not splice them into the earlier scalar/packed run's timings.

Every measured decode relative maximum/RMS error is **zero**. Worst prefill
relative maximum/RMS error across both arms is **0.976% / 0.241%**, within the
unchanged 2% / 0.4% component gate. Native repeat error is also recorded.
All **260 GPU finite/scale comparisons** against Torch pass, including D>P,
P>D, masked tails, eight planes, first/last entries and IEEE boundary values.
Queued/decode/cold cancellation, foreign-stream consumers, source and shared
pack immutability, and retirement pass. Peak Torch allocation is **3.874 GiB**
within the 8 GiB budget; this includes weights and outputs, not just scratch.

This does not establish full-model TTFT, output tok/s, acceptance or generation
quality. M3 still requires actual arrivals, layer continuation, cache/graph
ownership, TP4 NCCL and matched 32K/128K C=1/C=4 serving comparisons. The first
cold window still packs all remaining cold routes. Earlier S/P/M1 admissions
retain their separate sources and proof scope.

## Attribution and retained failure

`profile_before.json` is a separate warm cProfile capture on `3275c783`, after
the #920 integration and before these preparation changes. It identifies
repeated host grouping, `torch.tensor` calls and waits. Its perturbed 114 ms
total is attribution only. The final report has another separate profile:
17.34 ms total, with one route grouping (6.57 ms), one value check/readback
(5.16 ms) and full descriptor hashing (3.56 ms). Profiling totals are not A/B
samples, and remaining time is not all Python interpreter overhead.

The first candidate `77472b7d`, reservation `st-mixed-prepare0914v1`, passed
the 260 value checks but failed when CuTe refused a misaligned cold source
view. The report and admission are retained as `gpu_alignment_failure*.json`.
The fix aligns every packet span without changing route rows; a nine-expert
CPU regression exercises the formerly unaligned boundary. Only the successful
new frozen reservation contributes to the performance table.

## CPU and compiler evidence

The implementation passed [CI](https://github.com/choiceoh/stkernel/actions/runs/34802703129/job/103848411591).
The pinned Linux gate passed **357 tests / 28 CUDA skips across 385 discovered
tests in 49 isolated modules**. This includes nine packed-plan tests, complete
scalar/v1/v2 descriptor equality, packet alignment and ownership, four-process
Gloo descriptor agreement/refusal, cancellation and slowest-rank retirement,
S/P integration and the current compact-staging decode body.

The CPU route-planning benchmark uses six samples per arm, one unmeasured
warmup and alternating order. Source-table bytes match. Per-cell medians are
**5.22–5.50 → 2.29–2.56 ms** at P=9240 and
**19.74–19.84 → 7.36–7.56 ms** at P=32768. These figures cover planning only;
they exclude GPU copies, value checks, hashing and serving work.

All **eight actual SM121 CuTe/PTXAS/TVM-FFI builds** pass: hot/cold producers and
ordinary/prepared M16, M32 and M128 bodies. The two prefill lengths reuse one
dynamic compiled handle. A ninth build compiles the fused Triton value check
for SM121 with runtime D/P extents. The ordinary kernel AST pin was independently
derived from reviewed main `2ac7de6f`, preserving #923's compact A/SFA staging,
gate-slot release and separate FC2 packed scales in both ordinary/prepared C1.

After the measurement, main `6522564a` (#926 direct decode pool reader and
top-k IDs) was merged in `cc6ccf8b`. The lane-table conflict preserves packet
and mixed-reader binding before main's extracted reference-lane selection.
The focused integration gate passes **124 tests / 14 skips across 138 tests
in 13 isolated modules**, recorded in `cpu_postmerge.json`.
`postmerge_continuity.py` verifies all 25 compiler-source files and 39 of 41
GPU-source files remain byte-identical. The two changed profile files retain
all FFN network methods; the served FFN binding AST is identical after only
the explicitly reviewed pool-reader addition and reference-helper extraction
are normalized. This is source continuity plus CPU integration, not a new GPU
measurement or a live decode-pool performance claim. The timing source remains
the frozen `153b6f8b` build.

## Reproduction

CPU/compiler runs use image
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`,
with one CPU, 3 GiB, no network and CUDA hidden/uninitialized. The CPU image
differs from the available GPU image; both identities are retained.

```sh
# In the CPU image, with PYTHONPATH=/repo and CUDA_VISIBLE_DEVICES=:
python3 /out/cpu_runner.py
python3 probes/engine_mixed_prepare_bench.py --samples 6 --output /out/cpu_benchmark.json
python3 probes/engine_mixed_completion_compile.py --output /out/compile.json

# Canonical GPU queue only, from the frozen clean checkout:
ST_IMAGE=sha256:8190d08e822e1f9d18dda5a127a5d9a9e8c53ef4b5136d1154f9ea4b7727f1ed \
ST_PROBE_TREE=st-mixed-prepare-153b6f8b \
bash bench/fleet.sh run --gpu --detach st-mixed-prepare0914v2 20 \
  "PR895 aligned metadata and joined preparation versus packed v1" -- \
  bash probes/run_engine_probe.sh probes/engine_mixed_tickets_check.py \
  --ranks /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --ckpt-meta /home/choiceoh/models/st-glm53-nvidia-tp4-9391 \
  --samples 4 --compare-preparation --output /cache/st-mixed-prepare0914v2.json

# Rebuild and verify retained reports without a GPU:
python3 measurements/mixed_prepare_20260914/summarize.py
python3 measurements/mixed_prepare_20260914/verify_sources.py
python3 measurements/mixed_prepare_20260914/postmerge_continuity.py
```

Use a new reservation name and probe tree for a new run. Recorded checkouts and
admissions remain frozen. `gpu.json` retains every sample, `gpu_summary.json`
rebuilds the table, and `source_identity.json` checks each report's manifest
against its own source commit, including the earlier profile and rejected run.
