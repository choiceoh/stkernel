# V5 CUDA binding static lookup mapping

Read-only extraction and binary inspection after the V5 diagnostic. No container
exec, image/package change, GPU call, queue action or serving mutation occurred.
`copy.json` identifies the inspected pinned-image container and binary SHA.
`package-identity.json` verifies cuda-bindings 13.3.1 metadata, the wheel RECORD
hash/size, no changed target files in its writable layer, and no mounts at or
above the copied targets. V5 used the same immutable image; its removed
one-shot diagnostic container cannot itself be re-inspected.

All 34 V5 sanitizer frames map to 34 distinct BLR x23 instructions. The loader
entry resolves x23 from the ELF string `cuGetProcAddress_v2` at 0x6c2f0.
The sanitizer frame is the byte preceding the return PC (call address + 3).
The copied ELF's .rodata provides the symbol name; AArch64 argument instructions
provide requested version, flags, and query-result pointer. Every reported
lookup requests a version above V5's observed driver API 13000: 9 request 13010,
12 request 13020, and 13 request 13030. All flags are 0 and result pointers NULL.

`lookup-map.json` retains register/address derivations and return checks for
all 34. `lookup-map.md` is its compact table. `map.py` reproduces the mapping from
the binary, full disassembly and original V5 memcheck log at its recorded /tmp
path. The first two symbol-address derivations follow the proven forward
branch 0x14ef4 -> 0x165ac; linear text traversal would incorrectly use unrelated
error-path register assignments. Branch evidence is recorded explicitly.

This establishes the requested-version mismatch at every recorded error site.
It does not by itself prove that mismatch is the sole cause of INVALID_VALUE,
identify every runtime-loaded driver/tool library, or establish a successful
replacement binding/tool version. V5 remains sanitizer exit 86 / 34 errors;
MoE numerics, races and performance remain separate pending gates.

This archive retains the34 call-site argument excerpts and identity evidence. The complete binary and full disassembly stay outside the repository; their original sizes/hashes and source paths are in transfer-manifest.json. To reproduce, obtain the exact pinned cydriver binary and matching GNU objdump disassembly under a separate artifact directory, then run `python3 map.py --artifacts /path/to/artifacts --log ../capture/memcheck.log.gz`. Root independently reran the mapping and matched all34 ordered symbol/version pairs against the official13.3.1 template.
