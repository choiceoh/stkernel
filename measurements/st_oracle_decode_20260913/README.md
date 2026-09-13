# ST Oracle decode budget and CLI repair

This is a CPU sensitivity calculation, not a new engine measurement. Run
`python3 measurements/st_oracle_decode_20260913/analyze.py` to reproduce
`forecast.json`; every input artifact is recorded with its SHA-256.

The timing seed is the **incomplete** 32K C=1 preparation on source
`4ff6c3729fa58f9551e8928f41f92fe57106efd3`: 453 iterations, 48.546 ms/step,
57.947% accepted drafts. Its later consensus failure means it is not a
qualified baseline. Assuming the previously measured KDA batching and terminal
MHC savings transfer in full gives 47.915 ms/step (20.870 step/s), leaving
2.460 ms to remove for 22 step/s. Even removing all remaining KDA work would
not meet that target. These are bounds for choosing work, not speed claims.

The structural C=1/C=4 and 32K/128K budgets reuse historical #838 component
coefficients and assume equal acceptance. They are not current consumer results.

Using the Oracle exposed three CLI failures: composition ignored explicit
acceptance/k/step-time overrides; its chunk arguments were applied after the
scheduler contract was constructed; and `--no-calib --json` crashed on a null
calibration. The fix configures the contract first, budgets the reserved draft
slots, applies k before deriving byte costs, honors explicit flat step time,
and records the actual executed contract in JSON. The composed chunk shapes
9216/2304 are those of the source component profile, not a claim about the
current engine's scheduler configuration.

Validation: `tests.test_step_tools` passed all 55 tests (24.545 seconds),
including the actual `storacle.py sim --compose` CLI with k=0/k=6 and no
calibration. `cpu.log` preserves the result. No GPU or full model was used.
