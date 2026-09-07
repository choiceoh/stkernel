# GLM53 W4 cache key: SHA256 default enabled

Matched four-node warm boots fell from **234.0 to 226.5 seconds** on average
(**7.5 seconds / 3.2% faster**). Control samples were 236/232 s; SHA256 samples
were 224/229 s, in B/A/A/B order. W4 key time fell from **10.3045 to 2.4990 s**
(75.7%); head model loading fell from 84.85 to 77.40 s. Profile time stayed
36.65 versus 36.85 s. This supports enabling `VLLM_GLM53_MK_PACK_SHA256=1`
in `profiles/glm53.env`.

SHA256 keys hash the same live weight bytes and preserve shape, dtype, pack
version, row/tensor scaling, RTN/GPTQ and LORC rank in the filename. The first
use computes the historical MD5 key too and atomically hardlinks an existing
pack. Later boots omit MD5. Link failures retain the old path; `=0` restores
the historical namespace. No packing or inference arithmetic changed.

## Matched conditions and gates

- Runtime and benchmark source: `7132fd15306166f15ce783fe2580dad661f22801`,
  based on main `944f65c` (#451). The final commit only enables the tested
  profile knob and records these results; runtime module bytes are unchanged.
- Four GB10 nodes, TP4, immutable image
  `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
  Three imported cache/pack modules match across all 20 node/boot snapshots.
- Both arms keep FAST_IO=1 and rank/FP8 caches enabled. PREFILL_WARMUP=0;
  canonical Korean onepass at 2K/32K after every boot. Health time uses a fresh
  container ID and 1-second polling from immediately before the launcher.
- PRIME was 541 s, including rank publication (srv1 alone spent 184.238 s
  saving) and compilation. It is excluded from the warm comparison. All four
  timed boots report compile-cache `action=reuse` with identical content IDs;
  onepass's legacy `cold_compile` annotation is not the boot timing oracle.
- All timed boots: 4/4 rank hits, 976/976 FP8 hits, zero cache errors, and
  254 W4 pack hits per rank with zero repacks. Candidate boots: 254 SHA hits
  per rank, zero MD5 fallback, alias publication or alias errors.
- Every boot: quality 6/6 and Korean corruption 0/4. GPU probes passed 36
  exact checks before deployment and another 36 in the deployed container,
  including both serialization formats, unchanged pack fields/strides,
  non-default streams, a 64 MiB boundary, and synchronous transfer fallback.
- CPU: 21 focused pack tests plus 3 receipt regressions; local pre-rebase
  70,981 logic / 30 megakernel / 87 fleet checks. Post-rebase remote static
  gate passed 6,687 logic / 30 megakernel / 92 fleet checks (host lacks torch;
  this does not replace the local torch tests or GPU proof).
- Sampled available RAM stayed above 7.50 GiB on the head and 9.59–14.18 GiB
  on workers; no net swap growth. All final arm containers were running and
  had no OOM flag. These are 10-second host samples, not CUDA peak measurements.

This is a two-sample-per-arm warm-start result, not a cold-checkpoint or general
throughput claim. Raw acceptance varied: control 51.77/49.19%, candidate
49.87/45.65%; decode medians were 21.9/21.9 versus 21.9/21.8 step/s. Generated
responses were not identical between boots. Exact pack-byte proof and the
retrieval/corruption gate are separate from those response and throughput
observations; no acceptance or throughput improvement is claimed.

The trial exited 0 and released fleet session `packkey0907`. At
2026-09-07 22:46:46 KST the final control had HTTP 200, FAST_IO=1 and SHA256=0.
The next holder had acquired the fleet; this change does not restart it.
The enabled profile applies on the next deployment/start using this PR.

## Evidence and reproduction

[report.md](report.md) / [report.json](report.json) contain phase, all-rank,
response and resource summaries. [boot-receipts.txt](boot-receipts.txt),
[onepass.jsonl](onepass.jsonl), both GPU JSON receipts and the source/image
identities are retained here. [raw-file-sha256.json](raw-file-sha256.json)
binds the full raw files preserved at:

- remote `/home/choiceoh/glm53-logs/pack-key-20260907`
- local worktree `runs/pack-key-20260907`

Run the versioned `bench/startup_cache_boots.sh` through normal
`fleet.sh run --gpu` with `STARTUP_CACHE_MODE=pack-key`. The recorded
[fleet-driver.sh](fleet-driver.sh) shows this run's deployment and proof steps.
Use a new evidence directory and session for a rerun. The profile's new default
does not change the bracket because both arm values are explicit.

To regenerate/validate the summaries against the retained raw directory:

```sh
python3 measurements/glm53_pack_key_20260907/pack-key-report.py runs/pack-key-20260907
python3 measurements/glm53_pack_key_20260907/collect_receipts.py runs/pack-key-20260907
```
