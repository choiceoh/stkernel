# GB10 bounded graph experiment

An experimental executor wraps a deterministic retained CUDA graph in a
conditional WHILE node, executing at most 1, 2 or 4 iterations per launch. It
is not connected to serving and introduces no boot flag or precision change.
KDA state remains FP32.

The native executor resets its device iteration index on every replay, keeps
the captured graph/pool and caller allocations alive through completion, and
requires replay on the construction stream. The body must preserve each
iteration's output using the device index and write a stop vote after committing
that iteration. The first iteration must already be authorized by the caller.
Host callbacks, explicit event/semaphore nodes, allocations and nested
conditions are rejected. CUDA instantiation enforces the remaining conditional
body restrictions. Cross-stream capture can encode dependency edges rather
than event nodes; those edges are not categorically rejected.

The GLM policy stops after any row finishes, crosses a prefix block, cannot
fit another full speculative step in its reserved KV or fixed context bucket,
or sees an ordered interrupt. TP4 MAX consensus is required before a rank
branches; the helper refuses a fallback transport. Concurrent raw CPU writes
to CUDA memory are not a cancellation interface. Captured stochastic RNG is
not supported by this experiment.

Validation on 2026-09-13:

- SM121a native compilation passed in 46.88 seconds without a CUDA context;
  exact source hashes are in compile.json.
- 69 ST-image tests ran: 66 passed and 3 unrelated CUDA tests were skipped.
  Coverage includes package boundaries, stop conditions, real LocalTP MAX
  agreement, existing pipeline and graph contracts.
- 27 local tests passed for the CPU stop policy and fleet probe admission.
- The GPU gate is `probes/engine_bounded_loop_check.py`: actual served token
  commit inside a conditional graph, C=1/C=4, limits 1/2/4, changing inputs,
  EOS, generation limit, prefix, KV reservation, bucket and interruption exits,
  per-iteration log equality, owner lifetime and stream rejection. This is
  single-GPU correctness, not real TP4 collective or serving proof. Pending.

## PR760 simulation

`simulate.py` uses the current scheduler constants with C=1/C=4,
2K/32K/128K, 512 generated positions, fixed seed 7 and 3 repeats. Device cost
is zero: the model is deliberately synchronous and generates no language.
Measured Runner median host cost on this macOS host was 3 us at C=1 and
4-6 us for C=4 workloads. The 128K C=4 workload never reached width 4; its
observed widths are preserved in simulation.json.

Hypothetically amortizing *all* this work over four steps would save at most
2.25 us/step at C=1 and 3-4.5 us/step in these C=4 workloads. This is an
optimistic ceiling for the modeled work only. Torch/kernel submission is
absent, existing async hiding is unmeasured, the extra TP4 stop collective
costs time, and per-iteration result consumption still requires host work.
These numbers neither prove a GPU win nor bound the full serving opportunity.

## Serving integration still required

The current AsyncDecode has two pending slots and one prefix staging area per
row. Before connecting this executor, reserve the entire burst, allocate and
retain every iteration's token/acceptance/boundary result, resolve them in order,
and preserve cancellation and streaming semantics. Capture and validate the
actual target/drafter/transport body, including rank-agreed loop termination.
A matched C=1/C=4 onepass with 2K/32K/128K, acceptance, length, quality, tok/s,
TTFT and detailed timing is required before any default decision.

CUDA constraints and handle reset semantics:
[Conditional graph nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes).
