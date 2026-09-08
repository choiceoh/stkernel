# Public-restore CPU fixture failure

Binding diagnostic v4's supervisor restoration failed before completing its
CPU gate. The retained v4 fleet log reports missing executables `new`,
`too-late` and `stale` in `test_fleet_pending`: these are synthetic fixture
commands. The normal supervisor exports FLEET_PREPARE_MANIFEST, which leaked
into the test reservation and activated real preparation during a fixture edit.

A separate main-based change, [PR #486](https://github.com/choiceoh/stkernel/pull/486),
clears that variable only within the synthetic fixture. A regression runs an
existing edit case with a populated parent manifest and verifies that cleanup
restores the parent's environment and leaves the manifest bytes unchanged.
Runtime preparation and restoration rules are unchanged.

The same injected parent environment reproduced two failures and two errors
among the original seven tests on main `0ac504b7`. After the fixture change,
all eight tests passed with zero skips. `before.log`, `after.log`, `summary.json`
and the exact patch retain the command, source hashes and parent-manifest
checks. The fix is commit `5075316` on the separate branch
`codex/glm53-fleet-restore-test-env`; PR #486 was opened as a draft.

This is local CPU evidence. It is not a successful public restoration or GPU
admission, and it does not show that approved main contains the fix. The next
GPU attempt still needs a working normal restoration path.
