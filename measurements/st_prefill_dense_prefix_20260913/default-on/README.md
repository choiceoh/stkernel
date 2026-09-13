# Operator-enabled dense-prefix serving default

Source `a351f0b8`, based on merged PR #887 (`051570e7`). The operator requested the
new dense-prefix kernel be enabled. Both production and experimental boot declarations
now default to `prefill_dense_prefix=1`. An experimental boot can override it with
`STK_prefill_dense_prefix=0`; production continues to reject all environment overrides.
Bare model/ExecutionPlan constructors remain neutral, as for the other serving choices;
boot constructs the actual plan from the declared settings.

22 focused CPU tests passed without skips in 0.971 seconds (`cpu-tests.log`):
`tests.test_prefill_dense_prefix.PrefixGeometryTests`, `tests.test_engine_knobs`,
`tests.test_engine_native_execution`. The existing ST image was used with no GPUs,
runc, network=none, CPU=2, memory/swap=4 GiB, pids=256 and a read-only source mount.
Image: `sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`.

PR #875 Oracle `e2bfbb9a` compared `051570e7` with `a351f0b8`, using the retained actual
checkpoint config, C=1, contexts 2000/128000 and **no --set override**. Its actual boot
plan changes from false to true (`oracle-pr875.json`). This proves default selection;
it does not provide new GPU timing or quality evidence. The numerical kernel body is
unchanged. No GPU queue submission, engine build/boot, deployment or restart occurred.
