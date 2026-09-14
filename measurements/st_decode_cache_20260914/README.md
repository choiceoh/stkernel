# K=7 direct pool-cache glue

Base: `bec3b6dd7fc258a65f1038e5bc1f5c38b1341052` (includes #914, #918 and #919).
The existing `decode_dsa_inputs=1` default now includes these two changes in
experimental and production serving, for captured M=8/16/24/32:

- The pool window reads the full tail ring through device physical-slot ids;
  it no longer copies the ring with `index_select`.
- One kernel calculates pool addresses, scatters completed key/scale records,
  and writes the new raw keys/gates into the tail ring.

Pool compression, ranking, quantization, recurrent state, precision and K=7
are unchanged. The window finishes before cache writes. The bound route must
execute at every DSA layer and declared width, or boot proof fails.
`STK_decode_dsa_inputs=0` remains the experimental rollback for this bundle.

## Scope and source counts

The affected glue changes from five launches to two per DSA layer, excluding
the unchanged compression kernel: **55 -> 22**, or **33 fewer launches per
target forward** at both C=1 and C=4. Each call also removes a
`C*10*2*128` BF16 gathered-tail tensor and `C*3` int64 address/count elements
(5,144 bytes per active sequence). These are source counts and transient
allocations, **not a measured speedup or graph-pool memory saving**.

## Completed checks

Pinned CPU runtime:
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`,
Linux/arm64, Torch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0.
All containers used runc, no network, NVIDIA_VISIBLE_DEVICES=void and
CUDA_VISIBLE_DEVICES=; none accessed a GPU.

- `cpu-tests.log`: 97 tests, 90 pass / 7 GPU skips. Includes model routing,
  boot proof, graph contracts, default/rollback policy and kernel imports.
- `interpreter-tests.log`: 14 tests, 13 pass / 1 GPU-only skip. Actual Triton
  kernel bodies match independent Torch window and cache-write references
  byte for byte at C=1/2/3/4, mixed contexts near 32K/128K, changed physical
  slots/pages, rollback, padded/strided records and tail rings. Full backing
  storage comparison checks writes outside live records and ring cells.
- `compile.json`: full native extension build reused cache
  `527d3941349c914d1ea68fbb`; SM121 PTXAS compilation passed for the new mapped
  window and fused update (zero shared memory), plus the previous DSA bundle.
- PR875 Oracle head `95c27e6340159413c44a38eef9e54a84b13f9fc5`:
  `oracle.json` covers 32K/128K and C=1/C=4 using the real default flags.
  Changed costs are unpriced, so all four decode deltas remain null.
  The profile template contains no invented measurements.

## GPU and consumer evidence still required

The canonical combined `engine_kernel_check.py --lanes dsa_inputs` probe now
includes `engine_decode_pool_cache.py`. It compares all 11 layer offsets in
the same build, poisons output/padding, changes contexts/slots/pages between
graph replays, and times B/A/A/B in warm and evicted-cache conditions.
The pool-glue timing excludes unchanged compression; it is a component gate.

This work extends the existing short DSA reservation via official
`--replaces`; it does not request another full-model baseline boot.
No GPU, step/s, tok/s or acceptance result exists for this change yet.
The final consumer measurement remains 32K/128K, C=1 twice and C=4 once.
