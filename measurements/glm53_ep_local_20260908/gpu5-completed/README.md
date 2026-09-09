# Full MoE GPU v5

FAIL: 4 passed, 1 failed, 12 not run. Inner/outer exit: 1/1.

| Cell | Status | Phase | Compact / local wall ms |
|---|---|---|---|
| remap | PASS | complete | not measured |
| balanced4096 | PASS | complete | 14.4372 / 6.6425 |
| balanced6912 | PASS | complete | 19.6874 / 7.0251 |
| balanced8192 | PASS | complete | 24.3346 / 7.4742 |
| concentrated6912 | FAIL | changed-candidate | not measured |
| remote4096 | NOT_RUN | — | not measured |
| duplicate4096 | NOT_RUN | — | not measured |
| zeros4097 | NOT_RUN | — | not measured |
| balanced16384 | NOT_RUN | — | not measured |
| memcheck-remap | NOT_RUN | — | not measured |
| memcheck-balanced4096 | NOT_RUN | — | not measured |
| memcheck-remote4096 | NOT_RUN | — | not measured |
| memcheck-zeros4097 | NOT_RUN | — | not measured |
| racecheck-remap | NOT_RUN | — | not measured |
| racecheck-balanced4096 | NOT_RUN | — | not measured |
| racecheck-remote4096 | NOT_RUN | — | not measured |
| racecheck-zeros4097 | NOT_RUN | — | not measured |

Failure: RuntimeError('GPU cell concentrated6912 failed with exit 1')

Original immutable identity and running state match archived restoration. Incoming running=False. Normal supervisor restore/handoff and release are separately preserved in fleet-state/. No live equality is required after normal handoff.

Source d53fd44f2b8a69869cddc9df10fa4cbb21bc56d9; CPU16 proof 70c06279f97f2cb951606ae64be65038c957a9274bb3885d2371dc998a16578c; capsule manifest b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab.

All original job files, frozen composed/contract sources, CPU16 archive, CPU2 runtime receipts, and scheduler snapshot are preserved. Capsule binaries are hashed at collection but not copied. archive_manifest.json records original/stored sizes and SHA256; logs/PTX/cubins use deterministic gzip. Summary extraction is not a test rerun.

Synthetic single-GB10 EP remap plus MoE wrapper; compact baseline, E72/H4096/N2048.
No shared expert, TP4/EP4 transport, full-model TTFT, serving quality or decode acceptance.
Nondefault-stream execution is checked; this suite does not establish CUDA graph replay.
Timing does not override numerical or sanitizer failure. Unrun cells remain unverified.
