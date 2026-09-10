# CPU11 row allocator PTX inspection

The physical-row expression compiles to exactly **14 → 2 PTX instructions**
at each of seven unroll/tail sites. The first two sites remove 24 static
instructions together; all seven remove 84. The signed quotient/remainder
correction is absent from these replacement expressions.

CPU10 revision: `a9d0d1e3e8161ee44eac02191021ea9512a3e2cc`; CPU11 revision: `695274d9d4c62041dacaa4fcd0861ce9e357bd4b`.
Both decompressed PTX hashes match their original compilation receipts.

| Site | CPU10 PTX lines | CPU11 PTX lines | Count | Removed |
| --- | --- | --- | --- | --- |
| 1 | 632–645 | 632–633 | 14 → 2 | 12 |
| 2 | 686–699 | 674–675 | 14 → 2 | 12 |
| 3 | 740–753 | 716–717 | 14 → 2 | 12 |
| 4 | 794–807 | 758–759 | 14 → 2 | 12 |
| 5 | 861–874 | 813–814 | 14 → 2 | 12 |
| 6 | 915–928 | 855–856 | 14 → 2 | 12 |
| 7 | 977–990 | 905–906 | 14 → 2 | 12 |

The count covers only the arithmetic after the expert-base load and before
the `physical_row * 4` byte-offset conversion. Atomics, loads and downstream
address/store instructions are excluded.

First site (CPU11 lines 632–633):

```ptx
shl.b32 %r911, %r910, 7;
add.s32 %r906, %r911, %r902;
```

`%r902` is the unchanged row atomic result (line 628), and `%r910` is the
expert tile base loaded at line 631. Thus `%r906 = base * 128 + row`.
Lines 634–647 retain the nine operations that widen this physical row by
four bytes, address/store token_map and token_weights, and address/store
the same physical row in shared route scratch.

Second site (CPU11 lines 674–675):

```ptx
shl.b32 %r924, %r923, 7;
add.s32 %r919, %r924, %r915;
```

`%r915` comes from the row atomic at line 670; `%r923` comes from the expert
base load at line 673. The following nine operations (676–689) match the
old site under register renaming. This downstream match holds at all seven
sites. The token index remains `batch_base + warp_idx` at line 596 in both
artifacts; global pointer parameters 26/27/33/34 are unchanged.

The JSON companion records each instruction, register and line number.
PTX counts are **not SASS, GPU timing or TTFT results**; 84 counts seven
static clones, not the saving for one routed row. General compiler resource
and artifact-size checks are reported separately by the parent inspection.
