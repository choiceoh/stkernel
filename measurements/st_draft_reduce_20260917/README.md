# Drafter packet reduction and post-convolution fusion

Base: `f5ebf857` (main, including #1086 and #1111).
Branch: `codex/st-draft-reduce-conv-0917`.

Status: implementation defaults on for prepared OneShot K=7 blocks at C=1..4.
Component arithmetic and capture passed on the owned RTX 5050. **TP4 transport,
GB10 latency, engine step/s and acceptance are not measured.** The operator's
instruction not to enqueue fleet work remains in force; no fleet job was queued,
and no serving instance was stopped, restarted or deployed.

## Implementation

Each five-layer `block_rows` call has ten reduction boundaries. Nine now consume
the rank packets inside the convolution/residual/RMS kernel. The last MLP
consumes them inside convolution, then keeps the existing head residual/norm/MX
producer. Production eligibility is OneShot, block length 8 and 8/16/24/32 rows
of width 4096. Other shapes and transports retain their ordinary reduction.

This reuses `OneShot.exchange` and immediate same-stream `RankPackets.consume`.
The reader resolves rank pointers on device on every replay, bypasses L1 for
ring reads, sums ranks 0/1/2/3 with an explicit FP32 left fold, and rounds that
sum to BF16 before applying the convolution coefficients. The coefficient,
convolution output, residual and norm BF16 boundaries match the existing path.
A failed packet consumer poisons the transport rather than retrying a different
collective. Boot checks all declared packet rows at both boundaries of every
layer, independently of the dense projection precision.

Network exchanges and kernel launches at these boundaries are unchanged. The
materialized reduced tensor is removed: C=1 previously wrote 64 KiB at each of
ten boundaries, or 640 KiB per `block_rows` call. This is **neither a peak-memory
saving nor a net traffic claim**: the fused consumer reads and sums peer values
again for preceding convolution taps. The additional reads and lower parallelism
of a row-wise norm consumer can offset the removed intermediate. No percentage
speedup is claimed.

## Evidence

| Check | Result | Scope |
| --- | --- | --- |
| Related CPU suite | 26 passed, 2 skipped (28 total) | Boundary counts, immediate consumption, failure poison, other block lengths, replicated reference, complete boot coverage, existing drafter/post-norm/rank-order tests |
| RTX 5050 arithmetic | 6 cells passed, bit exact | Four local rank buffers; mix, residual and normalized output; strided coefficients/residuals; zero, small, ordinary and large values; rank cancellation |
| RTX 5050 CUDA graphs | 24 replays passed, bit exact | Four replays per cell, changed values and alternating descriptor addresses |
| SM121 offline compile | 6 variants passed | Mix and normalized kernels at three group/tap configurations, GPUs hidden; shared memory 0/32 bytes |
| Real TP4 / GB10 / onepass | Not run | No transport, latency or acceptance verdict |

Cells are `(rows, group, taps)` = `(8,256,2)`, `(16,256,2)`, `(24,256,2)`,
`(32,256,2)`, `(8,16,2)`, `(16,64,4)`, all with block length 8 and width 4096.
Cancellation uses rank values `(16777216, -16777216, 1, 1)`; every compared
output must also have identical raw BF16 bits.

Runtime: owned `choiceoh@ost-97x`, RTX 5050 (SM120), Torch 2.13.0+cu132,
CUDA 13.2. Image `st-engine:glm53-sm120-x86` resolved to
`sha256:9f496f0dabe3a7b495d9b97181913cc20be1e4b3d3fcf2694407e34f24b3981b`.
CPU suite used `/home/choiceoh/stk-venv/bin/python`. GPU execution used the
owned GPU's `flock` lock and a bounded, network-disabled disposable container.

Files: [CPU log](cpu.log), [GPU report](gpu.json), [GPU log](gpu.log),
[SM121 compile report](sm121.json). JSON reports contain source hashes; the
arithmetic report explicitly records `transport_tested: false`.

## Reproduction

From this checkout in the stated runtime:

```sh
OMP_NUM_THREADS=1 python -m unittest tests.test_engine_draft_reduce tests.test_engine_draft_post_norm tests.test_engine_drafter tests.test_engine_oneshot_sum -v
python probes/engine_draft_reduce_check.py --gpu --output /out/gpu.json
```

For offline SM121 compilation use a separate container with no GPU devices:

```sh
CUDA_VISIBLE_DEVICES= python probes/engine_draft_reduce_compile.py --output /out/sm121.json
```

The same arithmetic/capture probe has a `--tp4` mode for a future explicitly
owned TP4 session. With the launcher's existing four-rank environment, execute
on each rank:

```sh
python probes/engine_draft_reduce_check.py --tp4 --output /out/tp4.json
```

It compares ordinary OneShot reduction plus existing consumers with real packet
exchange plus the fused consumers, writing one report per rank. This command
does not acquire a fleet lease and must not be run against another session.
It has not been executed for this record. It is an arithmetic/capture check,
not a throughput benchmark.

For a later same-build engine comparison, set `drafter.reduce_packets = False`
after preparation and before graph capture in the control boot; leave the
default in the candidate. This is a Python probe control, not an environment
knob or a runtime fallback. Final adoption on speed still needs matched K=7
consumer step/s and acceptance on GB10 TP4.
