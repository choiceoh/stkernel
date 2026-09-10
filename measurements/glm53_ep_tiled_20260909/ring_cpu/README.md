# EP tiled SF6 A-ring CPU evidence

Normal CPU fleet session: `eptiledcpu0909ring2`, worker `10.10.10.4` (srv4).
Original output: `/home/choiceoh/glm53-ep-tiled-ring-cpu-evidence`. This archive validates completed originals; collection ran no tests or GPU work.

The original CPU receipt reports **101 tests, zero failures/errors/skips**, four static and two dynamic CuTe lowerings, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process.

| Variant | A-ring | Registers | Stack bytes | Local bytes |
|---|---:|---:|---:|---:|
| Static M6 / SF6 | true | 123 | 0 | 0 |
| Static M12, M24, M32 / SF6 | false | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | 168 | 112 | 0 |

M6 carries `glm53_ep_static_sf6_a_ring_v1`; the other static keys retain their previous 16 fields. Resource values describe compiled artifacts, not measured runtime occupancy or performance.

Final source `5663b1beb13e97116e6244af02a22ecc5074a263` committed a generated build snapshot with the bytes already compiled. `rebind.json` preserves the worker's explicit rebind, and `source-verification.json` independently checks every mounted and contract hash against that immutable commit. This was **not another compile**. The original result SHA is `30a05469b124b567573d873ec03e27954398d732ba63879d04fced72a80b565e` and is unchanged.

All 20 original worker files, their before/after inventory, original rebind, and local captured CPU stdout are retained. The original first-attempt log is separate: that attempt returned 2 because the worker source directory was absent, before compilation. `submission-reconstructed.json` explicitly reconstructs the successful normal fleet command and reported exit 0 from the parent's execution-tool record; it is not original submission stdout. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON receipt contents are unchanged. No runtime files or historical archives were edited.

This is CPU compilation and contract evidence only. GPU numerical correctness, graph behavior, throughput, quality, and default adoption acceptance remain separate. No such acceptance is claimed here.
