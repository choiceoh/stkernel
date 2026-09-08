# Identical-source deployment and startup

Identical-source redeployment now reaches health in **213 seconds instead of 334 seconds on average: 121 seconds / 36.2% faster**. The four-node fleet bracket `deploycache0908v6` completed at runtime source `f1814b2e676d12d0ceac8cd6843934e8da8b7fdb`, with an unchanged container image and runtime environment. `DEPLOY_PRESERVE_IDENTICAL` is promoted to **1 for GLM deployment**; `0` restores legacy publication. The measured synchronization helper and runtime overlay bytes are unchanged by this promotion.

| Warm arm | BASE1 | FAST1 | FAST2 | BASE2 |
|---|---:|---:|---:|---:|
| Health wall (seconds) | 341 | 211 | 215 | 327 |

The 387-second cache-creation PRIME is excluded. In each candidate boot, all 60 files on all four nodes retained SHA256, inode and mtime, and no Ninja build log changed. Both controls rewrote all 60 identical files and rebuilt both OSAR and megakernel extensions on every node. Model load fell 134.75 → 79.30 seconds and memory profiling 103.85 → 36.15 seconds; phase timers are nested.

All five boots passed first text/image/video output checks and manual review, quality 6/6 and Korean corruption 0/4. Every timed boot had four rank hits, 976 FP8 hits with zero misses/errors, and 258 W4 hits per rank under the current CTA4 profile. Each head log contains seven loopback POST completions, zero external POST completions and zero before the first health response. Available host RAM stayed above 7.44 GiB; swap usage did not increase. Fleet accepted the supervised handoff at **10:23:42 KST**, with payload exit 0.

The result applies to identical-source redeployment with populated caches. It does not establish an empty-cache first-install speedup, general throughput improvement or bit-exact output equivalence. [Detailed timings](report.md), [machine verification](validation.json), and [full report](report.json) preserve these boundaries.

## Implementation and measurement

The candidate avoids rewriting identical overlay files on all four nodes. The production GLM deployer uses the shared `glm53_sync_overlays` helper with rsync checksum comparison and normalized 0644 permissions. Unchanged files retain their inode and modification timestamp. Changed bytes are published with a fresh destination timestamp, including same-size edits from an older checkout, so Ninja still rebuilds changed CUDA sources. Other model deployment paths keep their existing behavior.

The deployment guard still requires a clean checkout based on current main, and full SHA256 parity still runs after publication. The final head is also checked against the canonical source files before serving as the worker comparison reference. The startup experiment first passes this official deployment admission. Timed arms then replay only identical source publication: the existing manifest, source revision and every file hash must already match on all four nodes before any write. This replay cannot admit different code or a different revision. It uses the same production rsync helper or the existing install/scp commands.

Health wall starts after source publication and includes the container restart through first successful HTTP health response, with one-second polling. Publication duration is separate; timed replay excludes the official deployer's CPU admission checks, which are common to both paths. PRIME prepares caches and is excluded. The balanced order is BASE1, FAST1, FAST2, BASE2. All arms use `PREFILL_WARMUP=0` to avoid the separate background prefill benchmark; required model/MM profiling and real graph/kernel warmup still run. Graph-profile skip remains disabled in every arm. Every boot retains model/MM profiling, real graph capture and warmup, then exercises first text/image/video requests and the canonical 2K/32K Korean onepass.

## Validation

Local core validation on `4c04a5c` passed 71,046 checks and 38 megakernel regressions; the separately executed fleet suite passed 115 tests. The measured `f1814b2` Linux production admission passed 6,751 checks, 38 megakernel regressions and 115 fleet regressions, including the current CTA4 profile default. Ten Linux restore/supervisor tests passed. Real rsync and canonical SHA256 file checks passed on both macOS and Linux (six tests). Same-source admission, boot receipts and multilingual first-request checks passed (four, ten and three tests). The three audited CPU helper contracts and all four injected faults were rechecked. GPU startup/readiness/output proof is recorded separately from these CPU checks.

## Earlier attempts

`deploycache0908v2` stopped before publication when main advanced beyond its checkout. `deploycache0908v3` completed a 231-second PRIME with first text/image/video and quality checks, then stopped before BASE1 when #456 and #458 merged. No timed pair came from either attempt. The latter supervisor restored approved main successfully at 08:47:43 KST. A pending v4 was canceled before admission as #439 merged. These are not included in any performance comparison. The identical-source replay was added to keep an admitted runtime fixed throughout the comparison without weakening production admission.

`deploycache0908v5` reached a 458-second cold PRIME and a 343-second BASE1. Its video response correctly described red followed by blue in English, but the original sanity predicate recognized only Korean color names and stopped the bracket. The unchanged raw response and failure are retained in `/home/choiceoh/glm53-logs/overlay-deploy-20260908-v5` on srv2; v5 is excluded from this archive. The predicate now accepts both languages; wrong colors, missing timing, replacement characters and reasoning-channel leakage still fail. No candidate pair came from v5, and its timings are excluded from the final comparison. The supervisor handed restoration responsibility to the next verified boot holder.

## Reproduction

Run from a clean, current-main-based checkout on srv2:

```sh
export REPO=$PWD
export STARTUP_CACHE_EVIDENCE=/home/choiceoh/glm53-logs/overlay-deploy-TRIAL
bash bench/fleet.sh run --gpu SESSION 35 "identical source publication B/A/A/B" -- bash bench/run_startup_overlay_deploy.sh
```

The fleet supervisor owns final production restoration or a verified handoff. Regenerate and verify the report without GPU requests:

```sh
python3 measurements/glm53_overlay_deploy_20260908/overlay-report.py RAW_DIRECTORY REPO --verify
```

## Evidence bundle

`raw-evidence.tar.gz` contains the exact captured v6 boot logs, before/after publication and post-boot file/Ninja receipts, source/runtime identities, memory samples, responses, driver log and filtered fleet lifecycle. `raw-file-sha256.json` identifies each archived file. The extra Ninja snapshot taken after ownership handoff is excluded. Human-readable summaries, all first requests and onepass results are also retained alongside the archive.

To regenerate the report, extract the archive into an empty directory and run the verifier against this PR checkout (or measured source `f1814b2`). No GPU or service request is made:

```sh
mkdir -p /tmp/glm53-deploy-evidence
tar -xzf measurements/glm53_overlay_deploy_20260908/raw-evidence.tar.gz -C /tmp/glm53-deploy-evidence
python3 measurements/glm53_overlay_deploy_20260908/overlay-report.py /tmp/glm53-deploy-evidence . --verify
```

The final canonical-head SHA256 check and default promotion were added after the fleet run. The SHA check passed six real-file tests on macOS and Linux, including stale and missing destinations. Before integrating later main changes, the promotion commit runtime overlay/build/profile bytes and synchronization function body were verified identical to the measured revision. Shell syntax and whitespace checks pass. No additional GPU trial was needed for this read-only integrity check or for selecting the already measured path by default.

## Integration after measurement

Main `b1afa41` was integrated after the completed trial, including #461 MoE defaults and #466 observation tooling. Those upstream changes are not part of the measured 334 → 213 second comparison. The recorded runtime remains `f1814b2`; the verifier reads canonical files from that Git revision rather than silently substituting the latest checkout. Our measured synchronization helper is unchanged. Merge conflicts were limited to appending both measurement records and re-auditing the CPU contract digest: all current-main `test_logic.py` AST nodes are identical after excluding this PR's added full-runner graph regression wrapper and its invocation. No service restart or GPU trial followed the completed fleet handoff.

Integrated CPU gate: **all OK (71059 checks; 38 megakernel regressions; 115 fleet regressions)**. All three audited CPU contracts passed and all four injected faults were detected again.
