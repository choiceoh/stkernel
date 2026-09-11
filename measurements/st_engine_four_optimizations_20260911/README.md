# SM121a / TP4 runtime follow-ups

The branch integrates main `b18c97f4` (including the later LocalTP ownership,
pool-slot fusion and HTTP/chat changes). It addresses four runtime costs:

- **Greedy TP communication:** keep vocabulary shards through the target head.
  Encode each rank's maximum FP32 logit and smallest winning global token id
  into one int64 MAX candidate per token. Signed zeros tie and NaNs follow
  PyTorch argmax behavior. Orphan vocabulary rows cannot win. Stochastic/mixed
  sampling gathers the complete logits inside its own graph and preserves
  the seeded sampler and ending RNG state. Layer reductions still use NCCL;
  no legacy one-shot all-reduce or DFlash top-k optimization is claimed.
- **Direct target state writes:** Triton reads only the preceding recurrent
  state and conv history using a device slot id, then writes only changed
  positions. The graph no longer gathers/commits every KDA ring. Small indexer
  tail reads and block-table gathering remain. DFlash's separate context-ring
  implementation is unchanged except for memory measurement hooks.
- **Prefill/decode fairness:** after a prefill chunk delays live decoders, the
  next step decodes before continuing that prompt. A paused prefill retains
  its reserved decode place; waking an idle conversation cannot take it.
  A single prefill chunk can still delay a decoder; this is not a measured
  wall-clock ITL guarantee.
- **Runtime memory budget:** the arena stays explicit, with a separate 12 GiB
  workspace ceiling and 4 GiB physical OS reserve. The native PyTorch allocator
  enforces the byte ceiling (its fraction API is only an internal conversion).
  The largest legal prefill at the beginning/end of KV capacity and every
  target, drafter and sampling graph record allocated/reserved high-water marks.
  Every rank checks physical headroom before HTTP admission. The report goes to
  `--dump-dir/memory-rankN.json`. Direct CUDA allocations are outside the
  allocator; node-wide free-memory checks account for their physical effect.
  Unseen prefill tails remain bounded by the allocator, but are not all warmed.

## Validation

| Evidence | Result | Scope |
| --- | --- | --- |
| `gpu-tests.log` | 156 tests passed, no skips | Integrated source on srv2/GB10, allocator limited to 2 GiB |
| `memory-preparation-test.log` | 1 additional test passed | Largest-prefill contexts, state/slot cleanup after a deliberate warmup failure |
| `real-graph-rank0.log` | 17 cases passed; hidden output, state and KV bytes identical | Actual layers 0 and 3, isolated rank 0 arithmetic, target widths 1/6, reordered slots, contexts through 4096, rejected future writes |
| `tp-vocab-rank0.log` through `tp-vocab-rank3.log` | Each rank passes 18 greedy cases and 8 sampling cases | Real four-node NCCL and CUDA graphs; full gathered-vocabulary reference, ties, infinities, NaNs, signed zero, partially/fully masked shards, mixed/stochastic top-p and RNG |
| `local-tests.log` | 157 tests, 61 dependency/GPU skips | 96 CPU tests pass in the local environment without torch |

The small allocator test sets a 32 MiB byte ceiling, allows an arena allocation
(which reserves a 20 MiB slab in the pinned allocator), and confirms a 64 MiB
request fails before consuming that memory. It restores the preceding allocator
configuration. CPU tests separately cover transient peaks, insufficient physical
reserve and rejection without changing a prior limit.

The final fleet vocabulary probe uses a 512 MiB allocator ceiling, no model
weights, an isolated rendezvous port and task-owned containers. All ranks exit
0 and their containers are removed. Peak allocator reservation is 138,412,032
bytes per rank. The largest greedy candidate array is 192 bytes (24 int64
values); this is a candidate tensor size, not a measured network traffic total.

`engine-code-sha256.json` covers every final engine `.py`/`.cu` file and was
compared with the mounted source used on srv2. `tp-vocab-code-sha256.json`
identifies the fleet communication/selection implementation; the later main
integration changes LocalTP's test ownership machinery, not the NCCL methods.
`runtime-image.txt` identifies the standalone image used for the GPU suite and
real-weight probe. The four-node probe used `st-engine:9391` with the task source
mounted at `/repo`, the same pinned runtime libraries, and no serving overlays.

## Reproduction and limits

Use the standalone image with `-w /repo -e PYTHONPATH=/repo` and mounted current
source. Run the regression suite with:

```bash
python3 -m unittest discover -s tests -p 'test_engine_*.py' -v
```

For the real-weight graph comparison, mount the layer-0/3 rank slice at `/ranks`
and the matching checkpoint metadata at `/repo/meta`:

```bash
python3 probes/engine_decode_graph_check.py --ranks /ranks --ckpt-meta /repo/meta
```

Launch `probes/engine_tp_vocab_check.py` once per rank with the fleet's RoCE
environment, `WORLD_SIZE=4`, rank order srv2/srv1/srv3/srv4, and an unused
`MASTER_PORT`. No model or public HTTP port is used. `fleet_probe.py` preserves
the concrete launcher used in this run.

srv1/srv4 access has recovered, and existing GLM services were running on all
four nodes during this work. Those services were not restarted or replaced.
No full 45-layer boot with the new memory preparation, full-model quality,
DFlash acceptance, load/ITL comparison or production rollout is claimed here.
The 12 GiB workspace value is an enforced ceiling awaiting full-model
qualification, not a measured maximum or guarantee of sufficient capacity.
