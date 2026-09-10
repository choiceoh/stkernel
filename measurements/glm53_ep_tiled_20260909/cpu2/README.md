CPU2 completed successfully for EP tile-major SF6 on source `f6b0934eb3d14b46cc58c29f6c9983f776eed250`. This archive contains CPU compilation and contract evidence only; it does not establish GPU numerics, serving readiness, throughput, or a speedup from SF6.

The normal fleet session `eptiledcpu0909v2` ran on srv4 through srv2, using a no-device runc container with 4 GiB memory, two CPUs and the unchanged 12 GiB host admission guard. Submission ran from 1788940144.336396 to 1788940184.135463 (39.799067 s), returning 0. The immutable image is `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`; the bindings capsule manifest is `b29ac01b3a45a5c140ba205187c86411412c5b74e7ac31417747e028588debab`.

All six real CuTe lowerings passed: static M6/12/24/32 and dynamic M33/8192. Static keys explicitly select `sf6_v1`; dynamic keys end in `glm53_ep_prefill_local_fp32_v2`, `glm53_ep_tiled_sf6_v1`. The two fresh dynamic passes have identical keys, PTX bytes and cubin bytes. The inherited SF6 layout verification ran for both static geometries, preserving M6 t,r and M12/24/32 t.

The separate child process ran 96 CPU contracts with zero failures, errors or skips. Its original `contracts.json` matches the parent `result.json` source/runtime/test fields. Both receipts report CUDA uninitialized and runtime identity rechecked. No tests or compilation were rerun during collection.

| Artifact | Registers | Stack bytes | PTX local loads / stores |
|---|---:|---:|---:|
| Static M6 | 121 | 0 | 0 / 0 |
| Static M12/24/32 | 118 | 0 | 0 / 0 |
| Dynamic M33/8192 | 168 | 112 | 4 / 5 |

All resource reports show static SHARED=1024 and LOCAL=0. SHARED here excludes dynamically allocated shared memory; LOCAL=0 alone does not prove absence of spill or stack traffic. The PTX counts are static instruction sites, not runtime transaction counts.

`original-evidence.tar.gz` and extracted paths preserve all 20 worker files byte for byte (4,409,683 bytes): six PTX, six cubins, six resource logs and the two original JSON receipts. Worker inventories before and after copy bind every original hash/size and verify clean source, image and all 116 capsule files. `head-source.json` separately records the clean head checkout. `submission.json` and `fleet.log` are exact local submission originals. `source-verification.json` records the immutable git-byte comparison of all 22 mounted source hashes and 21 contract hashes; exact contract sources are retained under `contract-source/`.

`verification.json` records the successful original artifact/runtime validator checks. Reproduce the pure local verification with:

```sh
python3 -B measurements/glm53_ep_tiled_20260909/cpu2/verify.py
```

This command checks archived originals and imports the exact frozen artifact/runtime validators. It performs no GPU work, remote actions, compilation, or unit-test run. `collect.py` records the bounded read-only collection procedure; it intentionally refuses an already populated archive. `SHA256SUMS` covers every archive file except itself.
