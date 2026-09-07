# Fleet boundary recovery follow-up

Code: `0fdf45a`. Portable CPU gates pass 127 tests; Linux CPU gates pass 136 tests, including nine real-process supervisor cases. The full Linux run preceded a macOS-only test fixture isolation correction; all 15 runtime tests passed again on Linux after it. Every report has complete coverage and no skips. Docker, SSH, serving and restore are fixtures; these results do not establish live GPU turnaround savings.

The new cancellation case interrupts admission after holder creation and before restore debt transfer. The supervisor recovers and restores once without starting its payload. Other cases cover the actual base64 launcher command, candidate image rejection, newline-less production checkout configuration, preserving a candidate branch ahead of main, and enabled/disabled SHA cache receipts. The initial macOS full gate exposed a cold `total_memory` cache interacting with a mocked subprocess; the fixture now isolates that discovery explicitly.

No CPU or GPU numerical tolerance was relaxed. The sampled CPU log `decstepcpu20907.log` on srv2 passed 70,998 logic checks; the recent MOEREFORM0908 M2 failure was an actual GPU difference of 3.5 against 0.1875, not a CPU rounding rejection. That observation alone cannot calibrate a new tolerance.
