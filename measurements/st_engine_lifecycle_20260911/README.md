# Request and cold-KV lifecycle validation — 2026-09-11

Baseline diagnostic: PR #535 merge `3c9622b7d5eb8b326322c8933a20668381015c2f`.
Final implementation integrates PR #537, `5957441ec0fae475b79e28994afe36839ee04baf`.

The prior HTTP server used its monotonically increasing public request number
as a fixed-size KV row. With two rows, the third request failed despite both
previous requests completing and returning their blocks. The diagnostic in
`baseline-request-failure.log` executes that baseline Server against the same
synthetic engine and real runner/pools used in the new regression suite.

The server now reuses internal rows independently of public IDs, queues at
most 64 pending/uncollected results by default, and admits only requests whose
full generation horizon fits the KV budget. It copies results before releasing
model token buffers. Invalid input returns 400; saturation and engine shutdown
return 503. The stop packet reaches all ranks; cancellation clears both partial
prefills and waiting/running requests. Kernel failures still propagate.

Cold KV is written into a fresh immutable generation and published by an
fsynced manifest rename. Old generations remain valid until that publication.
Retired files and deletion tombstones survive partial cleanup. `cleanup()` can
retry after restart and reclaim unreferenced generated files off the decode
path; one tier owns its directory. Legacy active file names remain readable.

## Results

| Check | Result | Evidence |
|---|---|---|
| CUDA regression suite on srv1 | 80 passed, zero skips | `engine-tests.log` |
| Same suite without PyTorch | 61 passed, 19 skipped | `cpu-tests.log` |
| Repeated/concurrent requests | 40 public IDs complete using two reusable rows, with preserved uncollected results | regression suite |
| Actual GLM adapter, synthetic next-token net | 25 requests complete; tokens, limits, context, rows and slots all released | regression suite |
| Four logical ranks through LocalTP | identical admission/reuse/eviction, continuation and stop; 13 requests per rank | regression suite |
| Real HTTP server | malformed JSON and invalid IDs return errors; subsequent valid request succeeds | regression suite |
| Error/shutdown handling | failed open, failed decode and partial-prefill cancellation release resources and wake clients | regression suite |
| Cold-KV fault injection | data-write/fsync/manifest failures preserve previous bytes; final-manifest/unlink failures remain recoverable | regression suite and `nvme-io.log` |
| O_DIRECT on actual host storage | 16 MiB KV, 2 MiB staging; producer streams and concurrent Futures preserve bytes | `nvme-io.log` |

The I/O probe also performs a successful replacement after the three injected
failures, reopens a tier with pending deletion, and cleans it without disturbing
the other live sequence. Counters report 26,214,400 bytes in completed demotions
and 45,875,200 bytes in completed promotions; failed attempts are not included
in those counters. These are correctness checks, not throughput measurements.

Ten retries after a throwing or malformed decode result also reuse the same
absolute horizon rather than consuming more blocks. A subsequent cancellation
returns all reserved blocks.

## Conversation integration

PR #537's `keep_idle`, park/resume, wake/extend, metadata boot and lane diagnostics
are retained. Public conversation IDs stay stable across turns while each HTTP
request gets a separate ID/event/result. Active or unknown conversations return
409 without disturbing the original request. Continuation budgets include all
previous context and any larger parked draft reservation before promotion.

When a new request needs a row or resident block capacity, the server evicts the
oldest idle conversation. Its state slot, model buffers and parked disk copy are
released. A later continuation of an evicted conversation returns 409. This keeps
new requests flowing after the retained-conversation capacity is reached.
Shutdown also releases idle and parked ownership, including after failed resume.

The suite covers retained turns, eviction, oversized continuations which preserve
the old context, duplicate requests on a busy conversation, failed disk promotion,
and actual GLM-adapter continuation over the cache arena with a synthetic net.
Draft acceptance clipped by EOS/token limits advances only the committed context
and drafter observations, so the next turn begins at the last emitted token.
Foreign block layouts remain unpromotable, cannot be overwritten by demotion,
and keep their active/retired files through ordinary cleanup.

## Environment and reproduction

srv1 GB10, isolated directory `/home/choiceoh/st-engine-f4d7-lifecycle`.
Image `glm53:v13-b12x-it` resolved to:

`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`

`source-sha256.json` records the tested engine, tests and I/O probe; every hash
was checked against the submitted local source. Logs have trailing whitespace
removed without changing diagnostics or numbers.

```bash
cd /home/choiceoh/st-engine-f4d7-lifecycle
mkdir -p evidence scratch
timeout --kill-after=5s 90 docker run --rm \
  --name st-engine-f4d7-lifecycle-tests --gpus all \
  --memory=4g --memory-swap=4g --cpus=2 -e OMP_NUM_THREADS=2 \
  --mount type=bind,src="$PWD",dst=/repo \
  --mount type=bind,src="$PWD/scratch",dst=/root \
  --entrypoint /bin/bash glm53:v13-b12x-it -lc \
  'cd /repo && python3 -m unittest discover -s tests -p "test_engine_*.py" -v && python3 probes/engine_cuda_io_check.py'
```

The final 80-test suite includes PR #537 conversation retention and continuation.
The I/O probe and runner self-check were also rerun after integration. The final
adapter-only context-clipping correction does not change the tested I/O sources. No serving containers, production checkpoint files or other fleet
nodes were changed. The four-rank check here uses LocalTP on one host; this
follow-up does not claim a new four-server model-quality or performance result.
