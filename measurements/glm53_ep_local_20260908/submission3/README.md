Normal fleet session eplocal0908v3 was cancelled before GPU execution.

Frozen source ebbe255100be7a00c293e8c4799ec403043c63c3 passed normal
preflight and queued at 15:35:09 KST on September 8. After attempt2 revealed
the missing compute-sanitizer path, this waiter was cancelled through
fleet.sh cancel at 15:37:21 KST. The driver exited 143 and no capture
directory exists. This is a cancelled admission, not a numerical failure.

The corrected runner pins and mounts the actual sanitizer installation,
checks it without GPU access before pausing service, and requires a clean
sanitizer summary. Those changes are compiled/tested under cpu7 and will
run from a new frozen source/session. This frozen source must stay unchanged.

Raw submission, freeze receipt, driver, exit and cancellation output are
preserved here. sha256.json covers original bytes, including decompressed
fleet.log.gz. state.json records the independent no-capture check.
