# Decode lengths shared across DSA layers

Base: `793c3fc74c3d760d1cdcb365f14ca7f509eb30a9`, including #929's output
buffers. This change unconditionally shares the sequence lengths and complete
pool counts produced by the existing integer `row_lengths` kernel between
DSA layers of one gathered target batch.

The 45-layer GLM target has 11 DSA layers. An unsplit C=1 or C=4 target forward
therefore calls the producer once instead of 11 times. Experimental split C=4
keeps one independent pair per group: two calls instead of 22. These are source
call counts, not GPU timing or a consumer speed verdict.

No native kernel, floating-point operation, precision, pool selection order,
recurrent state or persistent metadata buffer is added or changed. At K=7,
C=4, the two reused int32 vectors occupy 256 bytes and stay live through the
forward instead of just one layer. GPU allocator peak memory is not measured.

## Lifetime and capture contract

- `gather()` clears the previous pair before every warmup and capture. Thus
  each graph records its own producer, which recomputes on every replay.
- The actual target forward drops the metadata and page-table references in
  `finally`, including a failed forward.
- Context object, token width, pool size and producer identity must stay fixed
  within a gathered batch. A mismatch fails during warmup/capture.
- Split groups own independent metadata; eager prefill is unchanged. A model
  slice with no DSA layer does not invoke the producer.

## Completed CPU check

`cpu-tests.log`: 101 tests, 90 passed, 11 GPU-only skips, 9.598 seconds.
The actual `_select_rows` path matches the independent per-segment reference
through 11 layer calls and rollback. Tests also execute the target forward's
actual function body for success, failure and repeated contexts, check split
group ownership, and cover the integrated #929 head-buffer path.

The pinned ARM64 image is
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`.
The container used runc, one CPU, 2 GiB, no network and no visible GPU.

```sh
python3 -m unittest -v tests.test_engine_decode_lengths tests.test_engine_decode_rows tests.test_engine_decode_no_copy tests.test_engine_graph_contracts tests.test_engine_decode_pool_cache tests.test_engine_decode_pool_reader tests.test_engine_execution tests.test_engine_direct_mhc tests.test_engine_decode_buffers
```

## Existing GPU reservation

`engine_kernel_check.py --lanes dsa_inputs` now includes
`decode_length_reuse`. It captures the unchanged native producer at T=1 and
K=7/T=8, C=1..4 and split C=4. It poisons outputs before replay, changes context
lengths and row order, crosses the 32K/128K ranges, rolls back, and compares exact
integer values in both replay orders. It checks the producer was recorded in
both warmup and capture, once per group. It does not time or load model weights.

Replace the existing 5-minute `st-dsa-no-copy0914` reservation while preserving
its original enqueue time. No additional reservation or model boot is needed.
GPU replay, step/s, tok/s, tokens/step and acceptance are still pending. Final
consumer coverage remains matched 32K/128K, C=1 twice and C=4 once.
