# EP tile-major CPU1: compilation PASS only

Normal fleet CPU session `eptiledcpu0909v1` compiled frozen source
`a62d492b90caf712d0522528c0baa02926c256d6` on srv4. The outer command returned 0
in 24.69 s. The source checkout was `/home/choiceoh/stkernel-ep-tiled-0909-cpu1`
on head and worker. Both checkouts were still clean at collection.

Four static shapes (M6/12/24/32) and two independently lowered dynamic shapes
(M33/8192) produced **6 PTX, 6 cubins and 6 resource logs**, plus the original
`result.json`. This early run did not execute a CPU unit-test suite, GPU
numerics, CUDA graph replay, sanitizer, serving requests or performance tests.
`cuda_initialized=false`, the capsule runtime was rechecked, and the receipt
explicitly denies GPU-numerics and performance acceptance.

| Compiled shapes | Registers | Stack bytes | PTX local load/store sites |
|---|---:|---:|---:|
| Static M6/12/24/32 | 115 | 0 | 0 / 0 |
| Dynamic M33/8192 | 168 | 112 | 4 / 5 |

Both fresh dynamic passes have the same cache key and byte-identical PTX and
cubin. Resource logs report `SHARED:1024 LOCAL:0`; SHARED is the static resource
field, not total dynamic shared memory, and LOCAL:0 does not erase the dynamic
kernel's recorded stack/local instructions. These are compiler facts, not
measured speed or occupancy claims.

The payload used immutable image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`,
a read-only CUDA bindings 13.0.3 capsule (manifest `b29ac01b…`), runc without GPU
devices, 4 GiB / 2 CPU limits, and the unchanged 12 GiB available-memory guard.
`submission.json` and `fleet.log` are exact local submission originals. The
frozen probe/runner sources preserve the actual launch/guard contract.

Collection preserved all **19 worker files / 3,705,136 bytes** in their original
relative paths and in `original-evidence.tar.gz`, under a 128 MiB collection
bound. Worker-before/after inventories match every original file hash. Image,
source and capsule identity remained unchanged; the 116 listed capsule files
matched their manifest hashes. The measured 22 mounted-source hashes and five
contract-source hashes were independently compared with immutable git source
bytes; see `source-verification.json`.

`verify.py` reruns the **actual frozen artifact validator** and pinned runtime
receipt validator against these originals. It also compares every tar member
with the extracted original. Run locally, without CUDA or remote access:

```sh
python3 -B measurements/glm53_ep_tiled_20260909/cpu1/verify.py
```

`verification.json` contains the saved successful recheck. `SHA256SUMS` covers
all archive files except itself. No default promotion or GPU result is implied.
