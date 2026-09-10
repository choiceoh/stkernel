# EP tiled + SF6 canonical onepass 1 — FAIL

Frozen source `f6b0934eb3d14b46cc58c29f6c9983f776eed250`; session `eptiledsf60909v1`, ticket `17889403431195582`, owner `1195582`/start `44273126`. The canonical B0 → B1 → A run ended with payload/outer status **4** at 2026-09-09 17:19:58 KST. Every arm passed factual checks 18/18 and failed the existing Korean gate 1/8 (two CJK characters). ABASE did not run. No gate or result was changed.

| Arm | Fixed1024 tok/s, three requests | Pooled tok/s | Fixed-window step/s | Facts | Korean dirty |
|---|---|---:|---:|---:|---:|
| EPTILEDSF6B0 | 76.09, 53.47, 50.55 | 58.11 | 14.421 | 18/18 | 1/8 |
| EPTILEDSF6B1 | 70.99, 70.31, 67.88 | 69.70 | 20.455 | 18/18 | 1/8 |
| EPTILEDSF6A | 60.88, 77.92, 57.88 | 64.47 | 18.219 | 18/18 | 1/8 |

Decode non-regression is not established: A is -7.51% versus B1 in direct pooled tok/s and -10.93% in fixed-window step/s. B0 is cold and its step rate dropped during the fixed requests; it is not a valid acceptance baseline. All rows above are descriptive because all arms failed quality. Pooled fixed tok/s uses the canonical numerator `sum(completion_tokens - 1) = 3 × 1023` divided by summed `decode_s`; the first token belongs to TTFT. Speculative acceptance counters cover the whole onepass window and must not be treated as exact fixed-request acceptance.

A prefill observations: actual 2,128 tokens warm TTFT 0.758415s / 2805.85 tok/s; 32,545 tokens single combined-request TTFT 9.929369s / 3277.65 tok/s; 128,559 tokens single combined-request TTFT 39.339028s / 3267.98 tok/s. These observations do not establish an accepted performance improvement.

Four A ranks emitted complete 9-case startup canary PASS receipts (54 candidate and 54 control comparisons per rank), exact packed-owner preservation, graph capture and startup-trim completion. Each rank finalized 42 SF6 owners: 4,756,340,736 raw bytes released, 3,604,414,464 packed bytes retained, 1,151,926,272 bytes saved. The EP tiled serving lane proof passed 1/1. This is startup numerical/ownership/storage evidence, not sanitizer or throughput acceptance.

B1 and A four-rank ready snapshots bind exact container/start/image/runtime command identity and all 71 mounted-source hashes to the frozen source. B0's head boot log and canonical record were retained, but no four-rank B0 snapshot was captured. A's first prepared snapshot rejected an empty startup log; the failure is retained separately. Raw private Docker inspect/Env/Cmd are excluded.

Both passive observers exited on terminal/release; their original events and raw stream chunks are preserved with source scripts. Streams include multiple boots and are explicitly unassigned until matched to strict boot snapshots. The fleet holder was absent at collection; recovery was deferred to the idle controller. At 17:26:49 KST the captured idle-controller state was `waiting`, reason `not idle: GPU ownership is not quiet on 10.10.10.3`. This is not a public health check, and public recovery completion is not claimed. `runner-idle-recovery-20260909.json` is an older bootstrap/migration receipt, not this run's recovery-completion proof.

`job/` keeps the original onepass, verdict, submission and CPU2 receipts; `source/` keeps exact selected mounted/canonical source bytes and the full manifest; `snapshots/`, `canary/`, `head/`, and `streams/` retain runtime evidence. `manifest.json` records original and stored byte counts/SHA-256. Logs, source and JSONL use deterministic gzip with unchanged original bytes. No new benchmark, HTTP request, compilation or service mutation was performed by this collector.

`compare.py` is the unchanged post-measurement comparison utility (not frozen serving code). It validates record/request identities and reproduces the descriptive tables while leaving failed arms ineligible for acceptance. Its input is plaintext JSONL; decompress the unchanged archived original into a temporary file first:

```python
import gzip, hashlib, pathlib, subprocess, sys, tempfile
archive = pathlib.Path("measurements/glm53_ep_tiled_20260909/onepass1")
raw = gzip.decompress((archive / "job/onepass.jsonl.gz").read_bytes())
with tempfile.TemporaryDirectory() as directory:
    records = pathlib.Path(directory) / "onepass.jsonl"
    records.write_bytes(raw)
    subprocess.run([sys.executable, str(archive / "compare.py"), str(records),
                    "--sha256", hashlib.sha256(raw).hexdigest()], check=True)
```

`collect.py.gz` preserves the read-only collection utility. Both utilities are identified separately in the manifest. The original CPU2 result is byte-identical to the sibling `cpu2/result.json` archive: 96 tests, no failures/errors/skips, six actual compile artifacts, no CUDA initialization, and runtime postcheck complete. No CPU work was repeated for this archive.
