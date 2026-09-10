# EP tiled SF6 A-ring CPU evidence

Normal CPU fleet session: `eptiledcpu0909ring3`, worker `10.10.10.4` (srv4).
Original output: `/home/choiceoh/glm53-ep-tiled-ring3-cpu-evidence`. This archive validates completed originals; collection ran no tests or GPU work.

The original CPU receipt reports **101 tests, zero failures/errors/skips**, four static and two dynamic CuTe lowerings, and CUDA uninitialized. Runtime identity was rechecked and CPU contracts ran in a separate process.

| Variant | A-ring | Registers | Stack bytes | Local bytes |
|---|---:|---:|---:|---:|
| Static M6 / SF6 | true | 123 | 0 | 0 |
| Static M12, M24, M32 / SF6 | false | 118 | 0 | 0 |
| Dynamic M33, M8192 / SF6 | unchanged | 168 | 112 | 0 |

M6 carries `glm53_ep_static_sf6_a_ring_v1`; the other static keys retain their previous 16 fields. Resource values describe compiled artifacts, not measured runtime occupancy or performance.

Exact compiled source is `57914a3f8bb01a76a20099b9c2605be3ea15b7f4`. `source-verification.json` independently checks every mounted and contract hash against that immutable commit. The corrected startup canary recognizes the exact 17-field M1..8 SF6 A-ring key, and its CPU fixture derives keys from the actual compiler source. No source rebind was needed. The original result SHA is `247e41c1a72974886b0098e14892f1af53860575184c4bee235e271f423fd4b6` and is unchanged.

All 20 original worker files, their before/after inventory, exact execution metadata, bound plan, and captured CPU stdout are retained. `execution.json` records the actual normal fleet command, return code 0, start/end times and stdout hash; total execution was 44.415 seconds. Unlike reconstructed submission metadata, this record was written by the process that executed the command. Logs/PTX/cubins use deterministic gzip; original and stored hashes/sizes are in `manifest.json`. Raw JSON receipt contents are unchanged. No runtime files or historical archives were edited.

This is CPU compilation and contract evidence only. GPU numerical correctness, graph behavior, throughput, quality, and default adoption acceptance remain separate. No such acceptance is claimed here.
