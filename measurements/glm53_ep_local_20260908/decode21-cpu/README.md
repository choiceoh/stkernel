# CPU21b: TP SF6 Q0 compilation and contract gate

The normal CPU-only run **passed 151 tests with 0 failures, errors, or skips**. All 30 requested kernels compiled: 3 micro CuTe variants, 1 EP prefill CuTe variant, the stock/candidate TP SF6 CuTe pair, and 24 Triton remap variants. CUDA remained uninitialized and the isolated bindings runtime passed its final identity check. This archive records CPU compilation and integrity evidence; it does not establish GPU numerics, graph execution, throughput, or default acceptance.

Frozen revision: `028f98167376f0a0857c20c7ec89a3505c1a000f`. Source on head and worker: `/home/choiceoh/stkernel-ep-onepass-0909-21`. The normal fleet CPU job was `/tmp/glm53-ep-decode-cpu-0909-21b`, with worker `choiceoh@10.10.10.4` and evidence `/home/choiceoh/glm53-ep-cpu-0909-21/evidence`. Outer duration was 164.739 seconds; the inner receipt reports 162.260 seconds. Exit, payload, and evidence-copy return codes were all 0. The earlier srv1 admission refusal remains separately preserved in `../decode21-admission-failed`.

The immutable image was `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`. The CPU-only capsule used cuda-bindings 13.0.3 with manifest SHA256 `b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`. This is an isolated compiler dependency and does not assert that normal serving uses that package version.

`capture.json` preserves before/after source, image, capsule, and original-file identities. All **67 originals** (30 PTX, 30 cubins, 6 resource logs, and the result; 6,636,641 bytes) matched worker, copied head evidence, and the local archive. `evidence.tar.gz` preserves their original bytes. The standalone `result.json` is byte-identical to the original receipt. No tests or compilation were repeated during collection.

| TP SF6 prefill artifact | Stock | Q0 candidate |
|---|---:|---:|
| PTX bytes | 1,370,394 | 1,074,676 |
| Cubin bytes | 502,000 | 316,344 |
| Registers | 168 | 168 |
| Stack bytes | 1,648 | 112 |
| Static shared bytes | 1,024 | 1,024 |
| Reported LOCAL | 0 | 0 |
| Static `ld.local` / `st.local` sites | 197 / 36 | 4 / 5 |
| Sites before first MMA-role register increase | 193 / 31 | 0 / 0 |

Both TP keys retain the same E288/K4096/I512/top8/M128 SF6 geometry and runtime options. The candidate adds only `glm53_tp_sf6_q0_v1` after `sf6_direct_prefill_v1`. Both PTXs declare a 288-thread CTA. All remaining local instruction sites after the first MMA-role register increase are 4 loads and 5 stores in each version. These are static instruction counts, not dynamically executed traffic or performance measurements. `LOCAL=0` does not mean no local-memory accesses or prove absence of spills; explicit local depots and stack allocations remain. Static shared bytes exclude dynamic shared-memory allocation.

All three micro PTX/cubin pairs are byte-identical to CPU20. The M16 variant therefore retains its verified 96-thread CTA, 64/96-thread barriers, no `setmaxnreg`, REG157/STACK0/LOCAL0, grouped metadata reads, and shared FC1 A/SFA path. Exact artifact hashes, cited instructions, local-memory sites, and comparison scope are in `verification.json`.

Reproduce the pure local archive verification from this directory with `python3 verify.py`. It reads this archive and the preserved CPU20 archive, imports no GPU modules, and launches no work. `collect.py` documents the read-only collection process; running it again is unnecessary. `SHA256SUMS` binds every file in this directory except itself.
