# Fleet evidence reuse and editable paused reservations

Measured on srv2 on 2026-09-08 with source `3401404c48ae4d06a6d0a6f1a8931cba83dd42e6` and the existing `/home/choiceoh/vllm-env/bin/python`. The final documentation-only revision `2934bed9a2f3bb4bc6494f2269b216a544a51acb` consumed the identical receipt with `--verify-only`.

| CPU deployment validation | Wall time | Tests executed again |
| --- | ---: | --- |
| First complete validation | 155.637 s | Yes |
| Same evidence, next invocation | 1.107 s | No |

This single producer/reuse pair saved 154.530 s (99.3%) of repeated CPU validation wall time. Receipt verification still checks source/runtime/image/tokenizer identity. This is not a measurement of GPU queue wait, baseline frequency, production startup, or total experiment turnaround.

The release gate passed 71,121 checks with complete coverage and no skips: 71,115 logic checks (including 274 fleet regressions and 38 kernel regressions) plus 6 overlay publication tests. The CPU Docker chat contract separately passed 26 cases and 3,765 parser replays. Linux supervisor integration passed 28 real-process tests on srv4 at `ef74fb77fb32105367b7aa4d46f56064308fb2a7`; subsequent code only changed interpreter selection and release-test fixtures, with those final fixtures covered by the srv2 release gate.

The producer ran outside any fleet reservation with GPU visibility disabled. No GPU experiment, production deployment or serving restart was requested to obtain this evidence. Existing installed dependencies were used; no package installation was needed.

Reproduce on the configured fleet host from the tested source:

```bash
FLEET_DIR=/home/choiceoh/glm53-logs/fleet python3 bench/fleet_validation.py validate --repo "$PWD" --profile glm53
FLEET_DIR=/home/choiceoh/glm53-logs/fleet python3 bench/fleet_validation.py validate --repo "$PWD" --profile glm53 --verify-only
```

`receipt.json` preserves the producing revision and artifact hashes; `timings.json` records both invocations. `tests_run` in a reused receipt describes its original coverage, not newly executed tests. Full artifacts remain under the receipt path on srv2.
