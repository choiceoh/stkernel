# CUDA 13.2 packed-byte arithmetic and cuBLAS tuning review

SF6 word restoration and the native W4A8 LUT expansion use PTX 9.2 `add.u8x4`
by default. Each byte adds modulo 256 independently. This removes software
carry isolation; it does not change the FP8 values, scales, floating-point
accumulation or BF16 rounding boundaries. There is no new runtime knob.
The MLA FP8-to-BF16 half bridge restored by #956 is retained.

The cuBLAS part is a [tuning feasibility review](CUBLAS_REVIEW.md) with a
reproducible CPU scale-layout proof, not an installed GEMM backend. The
review identifies large FP8 prefill, the already-FP8 draft FC and vocabulary
head as candidates, and accounts for W4 expansion and lost fusion costs.

## Changed execution paths

- Static SF6 shared and direct-register word restoration broadcasts one
  base word and uses `add_u8x4`. Byte readers, scale lane mapping, ring
  ownership, publications and barrier order are unchanged.
- The short Q0 and long SF6 prefill word producers use the same native
  arithmetic. Their pinned parent files remain unchanged.
- Dense and MLA CUDA sources replace 32 `__vadd4` call sites in total with
  explicit native `add.u8x4`. CTA geometry and ordered-K products are retained.
- The shared CuTe device helper is included in the compile cache identity.
  Vendored-source provenance hashes are refreshed.
- The SF6 SASS probe accepts the lowercase `x` in `VIADD.U8x4`. The old
  opcode parser silently omitted that instruction. The retained static
  binaries were recounted without recompiling them.

## Completed validation

All GPU devices were hidden. The existing CUDA 13.2.1 image was run with
`--runtime runc`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`,
no network, bounded CPU/memory, and a read-only source mount. CPU tests also
used `--init` so detached test brokers were reaped. No service was restarted,
no GPU code was launched, and no fleet ticket or one-pass job was created.

| Gate | Result | Evidence |
|---|---|---|
| Engine CPU suite | 204 files, 1887 tests reported; 0 failed, 0 cannot run, 318 explicit skips | [cpu-final.txt](cpu-final.txt) |
| SF6 byte semantics | All 256 bases × 64 codes × four lanes; whole words, ring reuse, publication order and direct-register copy layout | Existing SF6 staging/register/prefill tests updated for the instruction contract |
| Dense native compile and dlopen | PASS, 58.02 s; 440 emitted packed-byte add sites | [compile-dense.json](compile-dense.json), [resources](dense.resources.txt), [codegen](codegen.json) |
| MLA native compile and dlopen | PASS, 56.47 s; 144 emitted packed-byte add sites | [compile-mla.json](compile-mla.json), [resources](mla.resources.txt), [codegen](codegen.json) |
| Static CuTe | All seven variants PASS; five word-path variants each emit 24 packed-byte add sites | [compile-static.json](compile-static.json) |
| Dynamic CuTe | Short Q0 M65 and long M8193 PASS; each emits 160 packed-byte add sites | [compile-dynamic.json](compile-dynamic.json) |
| cuBLAS input representation | PASS: 255 exponent values and 13,824 scale bytes, including row padding | [cublas-feasibility.json](cublas-feasibility.json) |

Static CuTe variants use 111–121 registers, zero stack and local memory.
The two dynamic families report 168 registers, 112 bytes stack, zero local
memory. These are candidate resource reports, not a baseline comparison or
a claim of spill reduction. The unchanged M16/M32 static scalar paths emit
no packed-byte addition, as expected.

Runtime: srv2 CPU container, Linux aarch64, image
`sha256:a9b53fd066bb4fa0c4d12982f7c5dcdd5a2591900a0088c3ffeb41ba0868425c`,
Torch `2.13.0+cu132`, CUDA toolkit `13.2.78`, CuTe DSL `4.6.2`, Triton `3.7.1`.
Native target is `sm_121a`. Binary and changed-kernel source hashes are in
the receipts. The full suite ran on the modified `de8bfff6` tree; subsequently
fast-forwarded #957/#958 did not change these six compiled kernel sources,
and all six hashes were checked again. CI checks the resulting PR tree.

The first full CPU attempt exposed a missing test-only byte-add binding in
the direct-register oracle; this was repaired. The other failures were a
container without an init reaper and two measurement directories omitted
from the CPU source copy. The final run includes those real directories
and passes. No production behavior was altered to bypass those tests.

## Reproduction

In an equivalent CPU-only image, mount this repository at `/repo` and an
owned writable output directory at `/out`. Execute from `/repo`:

```sh
python3 tools/check.py --list --jobs 2
python3 probes/engine_cuda132_native_compile.py --worker dense --output /out/dense.json
python3 probes/engine_cuda132_native_compile.py --worker mla --output /out/mla.json
python3 probes/engine_moe_sf6_compile.py --register-scales --sass --output /out/sf6.json
python3 measurements/st_cuda132_packed_20260914/compile_prefill.py --output /out/dynamic.json
python3 measurements/st_cuda132_packed_20260914/collect_codegen.py --output-dir /out
python3 measurements/st_cuda132_packed_20260914/cublas_review.py --output /out/cublas-feasibility.json
```

The native compile probe's output paths must match the later codegen probe.
The raw static report and disassembly remain under the owned remote
`/tmp/st-cuda132-adoption-0914.kQXMKs/out`; its hash is in the compact receipt.
Only repetitive scale-layout address lists were removed from that receipt.

**Remaining evidence:** actual GPU outputs/replay, resource behavior under
execution, component times, serving step/s and acceptance. Instruction counts
and CPU layout agreement do not establish a speedup. The operator's no-queue
instruction remains in effect.
