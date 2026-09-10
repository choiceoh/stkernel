# Public EP4 recovery after onepass6

Normal idle recovery restored the existing approved EP4 source `5e0216cfdc83c3cce9fc6b7f70a28a5511824d64`. The four newly created containers started at **2026-09-10 15:23:18 KST**. Read-only full verification passed on all four ranks and public `GET /health` returned **200**. The experiment's hybrid source was not adopted.

`receipt.json` preserves the approved receipt/source, new IDs and start times, fixed image, 71 read-only mount/source checks per rank, and source-bound startup evidence: 12 cases, 72 candidate plus 72 stock comparisons per rank, graph completion and SF6 FINALIZED42. Baseline native global keys are 23 fields for M4/6/8 and 20 for M12/16/24/32. PREP is configured on/cuda with shadow1/selfcheck64; this read-only check does not create fresh PREP execution or performance evidence. OPT/HYBRID environment entries were absent, recorded as absent; exact approved EP4 source and baseline keys establish the inactive configuration.

At **15:34:23 KST**, the recovery holder and `idle-recovery-owner` were both absent and the approved recovery pointer was unchanged. The subsequent idle observation was **waiting**, with the original reason `not idle: live serving lacks idle request counters: num_requests_running`. This is preserved rather than relabeled healthy. Public readiness is established separately by the four-rank full proof and health200; no request was sent to populate those counters.

The long startup's printed compiler stack was a `[boot-stamp] load-model still running ... its thread is at` watchdog sample, not a final exception. Compilation continued and model loading completed in 466.836498 seconds. No manual boot, restart, inference, deployment, queue or source change was performed during verification.

Only allowlisted compact proof is archived. Raw Docker Env/Cmd/inspect and the diagnostic log remain private. Original full-proof, closure and verifier hashes link this derived receipt to the read-only collection. `SHA256SUMS` covers the archived files.
