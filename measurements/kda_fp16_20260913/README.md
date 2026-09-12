# GLM53 KDA FP16 state storage

The candidate stores recurrent states in FP16 and keeps all recurrent arithmetic,
gates, prefill working states and functional output states in FP32. Direct ring
writes cast in the existing Triton kernel. There is no extra decode conversion
launch, scaling metadata, duplicate weight set or stochastic rounding.

The production default remains FP32 pending same-runtime quality and performance
evidence. A non-production boot accepts `STK_kda_state_dtype=fp16`; invalid values
are rejected before allocation. Production rejects all experiment overrides.
The canonical ST bracket selects committed source arms, so its FP16 arm changes
only `facts.KDA_STATE_DTYPE` after the common implementation commit.

## Memory contract

For 34 KDA layers, 16 heads per rank, 128x128 states, TP4, four resident request
slots plus null, seven rollback states, and 48 raw prefix snapshots:

| KDA region per rank | FP32 MiB | FP16 MiB | Saved MiB |
|---|---:|---:|---:|
| Active and rollback rings | 1190 | 595 | 595 |
| Prefix snapshots | 1632 | 816 | 816 |
| Boundary staging | 170 | 85 | 85 |
| Total | 2992 | 1496 | 1496 |

These are declared bytes, not measured RSS or a speed claim. `cache_capacity`
uses the FP32 baseline to keep KV block and snapshot counts unchanged; the arena
and budget use the actual dtype. The BF16 convolution/drafter rings and paged
FP8 KV do not change. The compressed RAM cache retains its existing byte cap.

NVMe manifest entries for FP16 carry `state_format=glm53-kda-fp16-v1`. Missing
tags retain their historical FP32 meaning. Foreign state formats cannot be
discovered, replaced or restored as this boot's state, even if payload sizes
happen to match. They remain subject to the existing stale-entry retention/LRU
policy. Snapshot compression remains lossless over the selected storage bits.

## Validation

CPU checks run in `st-engine:main-f838f71a` without GPUs, network access or serving
mutation, with two CPUs and a 4 GiB memory cap. Task source and logs are under
`/home/choiceoh/st-kda-fp16-cpu-0913` on srv2.

- Focused capacity, snapshot/restore, tier and boot tests: 111 tests, 30 skips,
  zero failures. Skips require CUDA or model metadata unavailable to that container.
- Integration and onepass recording checks: 39 tests, zero skips or failures.
- Fleet audit/lease checks: 76 tests passed after refreshing the changed
  single-host test's audited source hash (`cpu-audit.log`).
- Bounded GPU contract: `probes/engine_kernel_check.py --lanes kda-storage`,
  through the canonical fleet queue. Checks FP32 baseline, FP16 writes, seven-token
  graph replay, rollback, and FP16 initial-state continuation into prefill/decode.
  Passed all nine checks on srv4 in 16.485 seconds at
  `a1a97420d141a27a0fd25aa84e4ec0f77fc3ee33` (`gpu-contract.log`).
  State error is bounded by one FP16 rounding plus the existing FP32 arithmetic
  tolerance; untouched slots and padding remain bit-exact. The compiled state
  recurrence uses FP32 arithmetic and converts FP16 only at storage boundaries
  (`gpu-compiler.json`). FP32 sum differences at rounding ties can change a
  stored FP16 value by one representable step; bit-identical cross-dtype results
  are not a valid numerical oracle.
  The expanded contract passed all 12 checks in 13.678 seconds at
  `6403fd57413e70190edd74a17b18c67fff86160e` (`gpu-boundary-contract.log`).
  It also compares CUDA graph boundary staging, snapshots and restored rings
  against the CPU byte oracle for both precisions, four mutable slots, actual
  768-token boundaries, non-crossing rows, NaN bit patterns and padding.
  CUDA staging now derives its views, offsets and strides from the recurrent
  field's actual dtype; the old four-byte assumption would corrupt FP16 snapshots.
- End-to-end gate: a matched FP32/FP16 ST bracket, cold and warm onepass per arm,
  C=1/C=4 quality, Korean corruption, acceptance, actual output tok/s, TTFT and
  memory records. No performance or quality verdict is claimed without these results.

The matched bracket's baseline is `6403fd57413e70190edd74a17b18c67fff86160e`;
its FP16 arm is `939d8c1900dc0d713c6d87b1f2cb664e34a954f1`. Their trees differ
only in `facts.KDA_STATE_DTYPE`. The canonical controller at this revision adds
lease environment variables after preparing its environment digest, which
otherwise pauses a boot ticket. Declare the same identity at submission;
the queue still obtains and verifies the real lease before any boot:

```sh
ST_LEASE_OWNER=queue/kda-fp16-pair0913c \
ST_LEASE_PATH=/home/choiceoh/glm53-logs/st-fleet.lock \
bash bench/fleet.sh st-pair kda-fp16-pair0913c \
  939d8c1900dc0d713c6d87b1f2cb664e34a954f1 \
  --base 6403fd57413e70190edd74a17b18c67fff86160e 60 \
  'Matched FP16/FP32 KDA state, fixed capacity, C1 C4 quality and output throughput'
```

The bracket retains cold/warm JSONL and per-request artifacts under
`/home/choiceoh/glm53-logs/`; session logs are available through
`bash bench/fleet.sh logs kda-fp16-pair0913c`.

The research motivation is [DAMP, Table 1](https://arxiv.org/html/2608.27513v1#S6.T1):
FP16 state storage retained Kimi-Linear KDA reasoning quality better than BF16 at
the same storage cost. That result does not establish GLM53 quality.
