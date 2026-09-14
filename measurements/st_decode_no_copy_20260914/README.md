# Decode output and page-table copy removal

Initial base: `6522564ad56010c8ae1c6b7140bc6341de84c884` (#926).
Integrated main `e05096dd0a36c02c9e590946d5d3f1bc4281faf3` (#927) before
queue admission; the six changed implementation/probe/test files are identical.

TP4 MLA already returns one fresh contiguous output. The served wrapper still
concatenated that single tensor during decode and graph capture. It now returns
the owned result directly. Multiple head groups still concatenate; explicit
output destinations retain their ownership and validation.

`GraphCaches.gather` now clamps its `index_select` result in place. The gather
already owns a private copy, so the arena's original page map is untouched and
the second table allocation is unnecessary. This does not remove a clamp launch.

Both changes are unconditional defaults. No native kernel, floating-point
operation, precision setting, selection order or recurrent state is changed.
There is no new resident buffer or native compilation requirement.

## Source-level savings at K=7, TP4

| Context output | C=1 | C=4 |
|---|---:|---:|
| Removed copies per target forward (11 DSA layers) | 11 | 11 |
| Removed copied payload | 1.375 MiB | 5.5 MiB |
| Removed logical read + write traffic | 2.75 MiB | 11 MiB |

Each layer's output is `[8*C,16,512]` BF16. These are source counts and byte
counts, not measured GPU latency, memory-pool savings or consumer speed.
The page-table change removes one `C * table_columns * 4` byte allocation.

## Completed CPU evidence

`cpu-tests.log`: 65 tests, 60 passed, 5 GPU-only skips, 4.779 seconds.
`cpu-integration-tests.log`: the same suite passed after #927 integration,
65 tests, 60 passed and 5 GPU-only skips, 4.431 seconds.
Pinned ARM64 runtime:
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`.
The container used runc, one CPU, 2 GiB, no network, and no visible GPU.

The checks cover all 65,536 BF16 bit patterns (including signed zero and NaN
payloads), output identity, independent head groups, destination guard rows,
strided page tables, changed sequence ids, and preservation of the arena map.
The actual served wrapper runs under a stubbed native kernel at decode and
prefill widths, in both capture states, and retains a fresh result each call.

Reproduce the suite in that runtime with:

```sh
python3 -m unittest tests.test_engine_decode_no_copy tests.test_engine_decode_rows tests.test_engine_graph_contracts tests.test_engine_decode_absorb tests.test_prefill_dense_prefix.PrefixKernelTests.test_actual_served_mla_reuses_single_output_for_decode_capture_and_prefill
```

## Existing reservation

`engine_kernel_check.py --lanes dsa_inputs` includes the added copy/ownership
check. It replays the same BF16 payloads in both orders and exercises changed
page maps. A clone represents the unchanged MLA kernel's fresh result: this
checks only output delivery and graph ownership, not attention arithmetic.
It performs no timing, native build or weight load.

Replace the existing 5-minute reservation, retaining its enqueue age; do not
add a new model boot or reservation. GPU results and actual step/s, tok/s and
acceptance remain pending.
