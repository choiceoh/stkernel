# CPU13 selected-scale state PTX inspection

**The separate consumer scan is gone.** CPU12 has seven static selected-scale load sites (four loop copies and three tail copies); CPU13 has one first-scale load after the decoded count's empty-token branch. The seven unordered Float32 comparisons move into lane0's existing producer route allocation. Their combined static count remains seven.

| Static operation | CPU12 | CPU13 |
|---|---:|---:|
| Producer raw expert-scale shared loads | 7 | 7 |
| Consumer selected-scale shared loads | 7 | 1 |
| Consumer count/state word loads | 1 | 1 |
| Producer count/state word stores | 1 | 1 |
| Producer unordered Float32 comparisons | 0 | 7 |
| Consumer unordered Float32 comparisons | 7 | 0 |

The original PTX bytes match the receipts:

- CPU12: `55a8b0c48de4629874c1ee042adb05254657472db2795c3873248c04b4a4c1f4`.
- CPU13: `21a5c860b7f02aa887dd1cd8e90a5953094842b83ee24b118697c2f4d334d2de`.

CPU12 consumer load sites are lines 972, 981, 986, 991, 1009, 1020, 1028, with the scan backedge at line 1000. CPU13 lines 998–1010 have one state load, count mask / empty branch, one first-scale load, and loop setup; no scale-selection scan or backedge remains. The later SF-block loop remains.

## Packed state and empty-token proof

CPU13 lines 549–553 derive the state word and first-scale addresses from the same per-warp route base: offsets 1276 and 1152 differ by 124 bytes, exactly 31 Int32 slots. Lines 977–981 pack and publish `(count | (equal << 4))`; line 984 retains `bar.warp.sync`.

```ptx
ld.shared.s32 %r1011, [%r1008];
and.b32 %r33, %r1011, 15;
setp.eq.b32 %p121, %r33, 0;
@%p121 bra $L__BB0_122;
ld.shared.s32 %r1202, [%r1012];
setp.lt.s32 %p122, %r1011, 16;
```

Thus the low-four-bit mask dominates the empty-token branch. State 16 (count 0 / equal 1) skips the first-scale load and jumps directly to the next-batch phase at line 1834. The compiler folds the equality decode into `state < 16`; for count 0..8 / equality 0..1, this exactly selects the varied-scale branch when equality is 0.

## Raw bits and first-selected NaN behavior

Each producer site stores the same integer word it loaded from the expert-scale cache. `setp.neu.f32` then performs unordered Float32 inequality, matching CPU12's opcode. A `count == 0` predicate selects the old equality flag over that comparison's result on the first selected route. The compare instruction may execute for the first NaN, but its result cannot clear equality. The first-scale consumer reload retains the published raw word, including signed zero and NaN payload bits.

| Producer site | Raw load | Raw store | Count0 predicate | Float compare | First-count result select |
|---|---|---|---|---|---|
| 1 | 656 | 660 | 662 | 664 | 666 |
| 2 | 704 | 708 | 710 | 712 | 714 |
| 3 | 752 | 756 | 758 | 760 | 762 |
| 4 | 800 | 804 | 806 | 808 | 810 |
| 5 | 861 | 865 | 867 | 869 | 871 |
| 6 | 909 | 913 | 915 | 917 | 919 |
| 7 | 965 | 969 | 971 | 972 | 974 |

For C valid local routes, the consumer's selected-scale reads are **C → 1 for C > 0**, saving C−1 shared warp-broadcast instructions; C = 0 reads none in both builds. C = 1 saves no scale load. Producer expert-scale reads and the varied-scale quantizer's per-SF reads remain unchanged. No extra shared equality load, store, or slot was introduced.

These are **PTX instruction counts, not memory transactions, SASS, or speedup**. The same-address shared accesses are broadcasts across the warp. Moving the comparisons into lane0's dependency chain can change runtime scheduling; the performance effect still needs a matched GPU measurement.

Assembler resources remain **REG168 / STACK112 / SHARED1024**. PTX shrank 942077 → 941539 bytes; cubin 300864 → 298456 bytes. This inspection establishes no GPU numerics, racecheck, throughput, or full-model TTFT result.
