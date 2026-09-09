# CPU20: M16 shared-FC1-A direct scatter lowering passed

Normal fleet CPU session `epdecodecpu0909v20` ran on srv1 through the
srv2 scheduler. Its inner compile/test evidence spans 2026-09-09 11:51:08–11:52:09 KST (60.97 seconds; not the full outer job duration).
Frozen source: `82d81d65b6a74e09abd9be0a8f05508e4855f6f1`,
`/home/choiceoh/stkernel-ep-onepass-0909-20`. Payload, copy and outer exit
codes were all zero.

**120 tests passed, zero failures/errors/skips.** The original result records
`PASS`, `phase=complete`, `cuda_initialized=false` and
`binding_runtime_rechecked=true`. Actual compilation produced three micro
CuTe kernels, one dynamic-prefill CuTe kernel, and all 24 Triton preparation
variants. This archive is CPU evidence; GPU numerics and throughput remain
separate.

The candidate is M16/top8, with `scatter_fp32`, `ep_direct_scatter`,
`shared_fc1_a`, and `ep_m16` all true and their four ordered version tags.
The actual CuTe coordinate validator executed during lowering. Its PTX
contains `.reqntid 96, 1, 1`, 64-thread epilogue barriers and 96-thread full
CTA barriers. It contains **zero `setmaxnreg` instructions**: M16 uses static
register allocation, avoiding different register-reallocation instructions
inside its single warpgroup. The prior M32 candidate emitted two such
instructions for its separate warpgroup roles.

The candidate PTX is 217,619 bytes and cubin 86,928 bytes. Actual resource
output is `REG157 STACK0 SHARED1024 LOCAL0`; PTX has no local loads/stores.
The SHARED field is the reported static shared allocation and does not
include the dynamically requested shared region. It is not an occupancy
measurement. M64/top1/FP32 and M64/top8/BF16 controls retain REG222/220 and
zero stack/local; both PTX and cubin are byte-identical to CPU19.

Receipt-bound PTX observations reproduced by `verify.py`:

- Four FC1 unroll stages (4462–4510, 4529–4578, 4596–4644, 4662–4710)
  each emit six TMA instructions on their matching phase barrier, with
  `expect_tx=27648`. Destination offsets remain
  2048/51200/18432/53248/34816/55296 for phase 0 and
  10240/52224/26624/54272/43008/56320 for phase 1. Register-based addresses
  are resolved to their exact defining PTX instructions.
- Consumer end-FC1 barrier1660 and producer barrier4715 are both
  `bar.sync 2, 96`; producer FC2 TMA begins at4743 afterward. Final task
  barriers4304/4860 retain the same full-CTA count. There are32 static TMA
  sites, equal to CPU19:24 FC1 and8 FC2.
- Each direct-scatter block (2934–3287, 3288–3621, 3622–3953,
  3954–4300) contains two ID loads, two weight loads and two row guards,
  plus16 FP32 vector RED sites and one common64-thread post barrier.
  No shared output stores, BF16 shared loads or proxy fences occur in
  those blocks. The two BF16 rounding steps remain in the emitted data path.

These are compiled instruction sites and declared transaction sizes, not
runtime traffic, numerical acceptance, or a speed estimate.

`evidence.tar.gz` preserves all61 original files:28 PTX,28 cubin,4 resource
logs and the result. `result.json` duplicates the exact original bytes.
`head/` preserves submission, driver, terminal exit and full fleet log.
Worker/head/local original hashes matched before and after collection;
both frozen checkouts were clean/full with no alternates. Mounted and
contract-source hashes matched the result and submission. The same image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`
and strict116-file capsule manifest
`b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`
were verified. The unchanged12GiB admission guard and4GiB/2CPU no-device
runner are recorded in the original submission/driver evidence.

`collect.py` only read/copied completed evidence after GPU20 submission.
`verify.py` is a pure local archive verifier, not a rerun of tests or
compilation. Its output is `verification.json`; `capture.json` records both
collection snapshots. `SHA256SUMS` covers every file here except itself.
