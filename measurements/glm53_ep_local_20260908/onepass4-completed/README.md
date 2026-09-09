# Onepass 4 completed — warmup covered, overall verdict unresolved

The canonical B1/A/B2 run completed on frozen revision `96cb599d8816ee2585988fc7b750a21e2eb66a0b`, session `eplocalonepass0909v4`, ticket `17888996582431852`. Fleet recorded GO at **2026-09-09 05:50:11 KST**, B1 completion at 06:01:09, A completion at 06:10:00 and B2 completion at 06:16:09. The terminal receipt reports `succeeded`, phase `finished`, payload/overall return code 0 and 1558.2 payload seconds. Before/after terminal collection, the frozen source still had the expected HEAD and an empty worktree status.

**This is a completed measurement, not acceptance of a 40% improvement or a change to defaults.** The final judge remains `incomplete` / `unresolved`: the compatible B2 comparison has only one baseline sample (`floor_n=1`), so no noise floor is available. The earlier B1 comparison is incompatible for this prefill metric because B1 records `cold_compile=true` while A does not. The exact two verdicts are preserved; neither is rewritten as a pass or an invalid-run verdict.

## Direct recorded measurements

| Metric | B1 | A, with compact warmup | B2 |
|---|---:|---:|---:|
| First 2K TTFT | 2.410117 s | 2.394791 s | 1.952244 s |
| Second 2K TTFT | 0.848052 s | 0.837636 s | 0.850642 s |
| Third 2K TTFT | 0.892729 s | 0.959363 s | 0.888468 s |
| 32K TTFT | 10.807547 s | 9.828600 s | 10.802094 s |
| 32K input throughput | 3011.32 tok/s | 3311.25 tok/s | 3012.84 tok/s |
| 128K TTFT | 41.759952 s | 36.504840 s | 41.476623 s |
| 128K input throughput | 3078.52 tok/s | 3521.70 tok/s | 3099.55 tok/s |
| Per-request decode throughput range | 60.03–77.91 tok/s | 21.77–26.86 tok/s | 58.03–73.58 tok/s |
| Quality / dirty responses | 9/9; 0/5 | 9/9; 0/5 | 9/9; 0/5 |
| Serving proof markers | 1/1 | 3/3 | 1/1 |

Relative to B2, the recorded A throughput is **+9.9047% at 32K** and **+13.6195% at 128K**. First-2K TTFT is **22.67% longer**, and decode remains substantially slower. These are descriptive measurements pending the judge's missing baseline floor. The prefill summary's 2K token count is 2128; first-request token-rate calculations in `derived-comparison.json` use its actual 2121 prompt tokens. The five request hashes and prompt-token counts match across all three arms, as do the recorded source, harness, overlay, endpoint, session and workload fields.

A's startup logs show all four ranks completing the **14-key compact preparation** (`10 static + 4 dynamic`, `required=ready=14`) before real graph completion and first traffic. The intended mid-request CuTe compilation gap was covered. Other MHC/Triton/sampler/remap JIT warnings still occur; this archive does not subtract compiler wall time from TTFT or claim all inference JIT disappeared. The independent `final-validation.json` audit binds exact A/B2 snapshot-log prefixes to continuous streams and checks A worker prefixes from graph-ready at 06:06:04 through A completion at 06:10:00 KST. It records zero CuTe compilations and 28 other JIT warnings across four ranks. Its `SCOPED_PASS` covers runtime identity and logged warmup coverage only. The archived `verify-private-captures.py` is the read-only verifier used against retained private snapshots; raw inspect inputs are intentionally not included in this public archive.

## Evidence and boundaries

- `records/final-three-rows.jsonl` and `final-verdicts.jsonl` are exact remote originals, independently matching the parent's saved originals. `through-B1.jsonl`, `through-A.jsonl` and `verdict-through-A.jsonl` retain the earlier completion prefixes. They do not contain B2 results.
- `boot/` contains the complete saved head logs for B1/A/B2. `fleet/terminal-run.log.gz` contains the complete terminal run log. Gzip storage preserves original uncompressed bytes; hashes are recorded separately.
- `A-identity/` and `B2-identity/` contain only allowlisted summaries, source/mount hashes, launch-parser source and safe logs/manifests. **No raw Docker inspect, Env or Cmd is archived.** Each summary was checked against the existing private before/after captures, including stable container identity/start/config, parsed topology and exact frozen mounted-source hashes.
- **B1 has no recovered strict four-node configuration snapshot.** Both original observer attempts failed. Their 96/97-byte outer failure messages, coordinator traces and eight node traces are preserved under `snapshot-failures/`. The private helper fix canonicalized only the unordered `Mounts` list; before/fixed helper sources and factual repair receipts are included. It did not change the frozen serving source. A and B2 later obtained valid strict snapshots; that does not retroactively repair B1 evidence.
- `streams/*.terminal.gz` preserves all eight passive stdout/stderr byte streams through stream-subprocess closure. Every original chunk offset/length/hash was checked against the stored bytes. These raw streams span B1/A/B2 and remain **UNASSIGNED** unless a separate identity/log-boundary check attributes a segment.
- The earlier `*.through-A.gz` files preserve an exact contiguous arrival-time prefix through 06:10:00 KST. Arrival time alone is not proof of per-arm identity, and late-arriving A bytes may lie outside that prefix. `events-through-stream-end.jsonl.gz` and `events-through-observer-finished.jsonl.gz` distinguish data-stream closure from final outer-observer closure.
- The data reader ended after reservation metadata disappeared. The outer observer kept issuing read-only status polls until its owner verified PID 8542 and sent SIGTERM after release. `observer-closure.json` records `observer_finished` at epoch 1788902501.7452621 and a subsequent PID-absent check. The collector did not signal any process. The earlier terminal-boundary snapshot's `observer_end` check is historical; the actual emitted terminal event is `observer_finished`.

**Fleet success and observer cleanup do not prove public serving restoration.** The parent separately reported fleet FREE/queue 0 at about 06:22 KST, but public health was unavailable and default-serving restoration was not verified. No restored-service claim is made here.

`source/frozen-hashes.json` carries the previously verified 142 source/generated/harness hashes. `originals.json` preserves the through-A provenance; `terminal-originals.json` adds terminal artifacts. The terminal collector's final metadata append initially collided with the existing through-A manifest; copied originals were retained and the terminal provenance was written separately, without rerunning measurements. `SHA256SUMS` covers every file in this new archive except itself. No old archive, source, queue or GPU workload was modified during collection.
