# Onepass9: B1 completed; candidate startup failed

Frozen source: `e4425b984e3455614744f0f3072916b1b296bd0f` at
`/home/choiceoh/stkernel-ep-onepass-0909-9`. Normal fleet session
`eplocalonepass0909v9`, ticket `17889108153442432`.

The payload ran on 2026-09-09 from 08:40:16 to 08:55:27 KST; the supervisor
finished at 08:55:28 KST with payload/outer return code **1**. Terminal checks
confirmed the frozen source remained clean and unchanged, the supervisor was
not alive, and this session no longer held the reservation. Recovery was
deferred to the normal idle controller; this archive does not attest restored
public service.

`records/onepass.jsonl` contains exactly one original measurement row,
**EPONEPASS9B1**. Its quality result was 18/18 and Korean contamination 0/8;
its fixed 2K pooled decode rate was **73.5950561113 output tok/s**
(`3069 / sum(decode_s)` across the three fixed requests, excluding each first
output token). The individual fixed rates were 76.9346, 71.6949, and 72.3717
output tok/s; the recorded decode window median was 19.8495998991 steps/s.
The candidate A
failed its required startup self-test before serving readiness. There are no
A, B2, or B3 serving measurement rows and no paired speedup or regression
verdict. Missing A measurements are not zero throughput.

## Preserved failures

| Node / rank | Failed phase | Evidence |
| --- | --- | --- |
| local / 0 | short6 initial preparation | weights byte 16: actual 42, reference 0 |
| 10.10.10.1 / 1 | concentrated6912 changed-C2 | one row exceeded the unchanged candidate peak limit |
| 10.10.10.3 / 2 | short6 initial preparation | weights byte 16: actual 42, reference 0 |
| 10.10.10.4 / 3 | short6 initial preparation | weights byte 16: actual 42, reference 0 |

The three T6 failures expose a **self-test dtype mismatch**. The actual
receipts record BF16 router weights and scales under the serving process's
default dtype. The frozen self-test allocated its legacy remap scratch as
FP32, then compared its raw bytes with the candidate's correctly same-dtype
BF16 staging. An eight-weight row is consequently 32 versus 16 bytes; after
the first all-remote row, byte 16 compares different rows. The serving wrapper
normally sizes remap scratch from `topk_weights.dtype`. This harness error
does not establish a Triton store-width or dtype-branch defect. The separate
CPU9 artifact archive retains the FP32 and BF16 PTX specializations.

The rank-1 **numerical failure remains a failure** independently of that
harness issue. Failed row 6388 had relative L2 0.0114599466 against its
0.0231547039 limit, and relative peak 0.0405405387 against its
0.0399999991 limit. The worst output column was 2918, absolute difference
1.5. `failures/10.10.10.1.json` preserves the full bounded raw BF16 words,
routes, scales, actual per-row metrics, and unchanged limits. Its generic raw
candidate field is named `C1`; the enclosing recorded phase identifies this
call as **changed-C2**. Neither the dtype diagnosis nor a future run waives
this original failure. These results also do not isolate atomic accumulation
as its cause.

## Contents and checks

- `boot/` and `fleet/`: original B1/A boot logs and terminal fleet log,
  losslessly gzip-compressed.
- `failures/`: all four original parsed self-test receipts, each matched to
  exactly one FAIL JSON in that node's original stdout stream.
- `streams/`: closed passive stdout/stderr streams and events. Every recorded
  chunk hash, byte count, and contiguous offset was verified; closure records
  the observer's normal terminal/release stop and absent process.
- `B1-identity/`: all four source/image/topology/flag summaries plus original
  log/manifest snapshots. Before/after private container inspections were
  validated in memory; raw Docker `Config.Env` and `Config.Cmd` were not
  archived. There is no comparable candidate serving snapshot.
- `source/`, `originals.json`, and `terminal-capture.json`: frozen manifest and
  source hashes, original/stored byte provenance, and terminal checks.
- `collect-evidence.py`: read-only collector used for this archive. It never
  submitted workloads or changed services, reservations, or frozen sources.

`SHA256SUMS` covers every archive file except itself. Decompressed original
hashes are recorded in `originals.json`; original failure and measurement
records were not rewritten. No new tests or GPU runs were performed during
collection.
