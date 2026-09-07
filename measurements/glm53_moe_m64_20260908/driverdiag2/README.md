# Driver-first detector controls, 2026-09-08

The six API processes completed, with error counts 0 / 0 / 34 / 0 / 35 / 1 for torch-only, driver-only, torch-driver, driver-torch, torch-driver-invalid and driver-torch-invalid respectively. Driver-first initialization avoids the 34 bootstrap reports while still detecting the deliberately invalid `cuDeviceGet(999999)` call as CUDA_ERROR_INVALID_DEVICE. The process records preserve that intentional failure separately from successful ordinary driver calls.

The wrapper then failed because its parser accepted only the plural `errors`; Compute Sanitizer correctly printed `ERROR SUMMARY: 1 error`. The raw sixth log and its program completion are intact, but its wrapper JSON was never written. Device-memory and race canaries were not reached. This failed collection is not numerical, sanitizer or serving acceptance.

Source `dad37e5251c8a8d4aacf0d67d34ea619cfa14474`, normal session `moem64driver20908`, GO 05:45:27 KST, probe 05:46:30–05:46:45, exact incoming four-node recovery and outer exit 1 at 05:49:23. `summary.json` reparses the intact logs with singular/plural support. The original logs remain losslessly gzip-compressed.

The follow-up corrects the parser and makes successful API, device-memory and race detector controls a mandatory first step of the normal gate. Only after that check can driver-first sanitizer initialization and the full TP4/numerical/memcheck/racecheck sequence run. All reporting options and numerical thresholds remain unchanged. The intentional memory and race defects exist only in isolated diagnostic containers, never in the model runtime.
