CPU17 actual normal --cpu PASS: 146 tests, 29 contracts, 13 mounted files, 24 remap variants; CUDA uninitialized.

The original exact-byte comparison REJECT and VERIFICATION_ERROR.txt are preserved unchanged, both here and in cpu17.collecting. All CuTe artifacts and remap PTX are byte-identical to CPU16. The 24 remap full cubins are NOT byte-identical: the strict pinned checker accepts only the recorded source-mtime debug fields; executable and all other bytes match.

Original cubin bytes were not stripped or normalized. Original/stored SHA and sizes remain in archive-manifest.json. Remote originals, current source contracts, and the pinned capsule were checked before and after copying. This is compile identity adjudication, not GPU numerical or speed acceptance; GPU v5 FAIL remains unresolved.
