# CPU12 Q0 scale-address PTX inspection

The row-dependent Q0 scale address compiles to **41 → 9 PTX arithmetic instructions per static store site**, a reduction of 32. All **10** compiler unroll/tail sites match the same dependency graph within each build: seven equal-scale and three varied-scale sites. The total across these static copies is **410 → 90**, or **320 removed**.

Both decompressed PTX files match their compilation receipts:

- CPU11: `f4670451e10ffd170f243eddf11bb675d5c2f3d8b46877cbafc5014696b9ac90` (955476 bytes).
- CPU12: `55a8b0c48de4629874c1ee042adb05254657472db2795c3873248c04b4a4c1f4` (942077 bytes).

The inspection traces each scale store's Int32 byte offset back to exactly one shared physical-row load and its precomputed SF-column offset. It excludes those input operations, payload stores, address widening, and the final byte store. Register names and independent instruction order are normalized when checking all ten dependency graphs; opcodes and literal constants remain exact.

| Site | Path | CPU11 arithmetic lines | CPU12 arithmetic lines | Count | Scale store line |
|---|---|---|---|---|---|
| 1 | equal | 1286–1297, 1305–1333 | 1294–1302 | 41 → 9 | 1336 → 1305 |
| 2 | equal | 1341–1352, 1360–1388 | 1317–1325 | 41 → 9 | 1391 → 1328 |
| 3 | equal | 1396–1407, 1415–1443 | 1340–1348 | 41 → 9 | 1446 → 1351 |
| 4 | equal | 1451–1462, 1470–1498 | 1363–1371 | 41 → 9 | 1501 → 1374 |
| 5 | equal | 1518–1529, 1537–1565 | 1400–1408 | 41 → 9 | 1568 → 1411 |
| 6 | equal | 1575–1586, 1594–1622 | 1423–1431 | 41 → 9 | 1625 → 1434 |
| 7 | equal | 1631–1642, 1650–1678 | 1451–1459 | 41 → 9 | 1681 → 1462 |
| 8 | varied | 1799–1810, 1818–1846 | 1587–1595 | 41 → 9 | 1849 → 1598 |
| 9 | varied | 1954–1965, 1973–2001 | 1710–1718 | 41 → 9 | 2004 → 1721 |
| 10 | varied | 2120–2131, 2139–2167 | 1844–1852 | 41 → 9 | 2170 → 1855 |

CPU12 equal representative, arithmetic lines [[1294, 1302]]:

```ptx
shl.b32 	%r1251, %r1239, 8;
and.b32 	%r1252, %r1251, -32768;
shl.b32 	%r1253, %r1239, 4;
and.b32 	%r1254, %r1253, 496;
shr.u32 	%r1255, %r1239, 3;
and.b32 	%r1256, %r1255, 12;
or.b32 	%r1257, %r36, %r1252;
add.s32 	%r1258, %r1257, %r1254;
add.s32 	%r1259, %r1258, %r1256;
```

CPU12 varied representative, arithmetic lines [[1587, 1595]]:

```ptx
shl.b32 	%r1115, %r1076, 8;
and.b32 	%r1116, %r1115, -32768;
shl.b32 	%r1117, %r1076, 4;
and.b32 	%r1118, %r1117, 496;
shr.u32 	%r1119, %r1076, 3;
and.b32 	%r1120, %r1119, 12;
or.b32 	%r1121, %r38, %r1116;
add.s32 	%r1122, %r1121, %r1118;
add.s32 	%r1123, %r1122, %r1120;
```

For H4096/M128, CPU12 computes the physical-tile field as `(physical_row << 8) & 0xffff8000`, outer-row field as `(physical_row << 4) & 496`, and inner-row field as `(physical_row >> 3) & 12`. It combines these disjoint fields with the precomputed SF-column offset. The old signed quotient/remainder correction chains disappear from this dependency graph.

**PTX counts are not SASS, executed work, GPU timing, or TTFT results.** Ten sites count compiler copies, not ten mandatory operations for each routed row. The exact runtime saving depends on selected routes and branches. Assembler totals remain **REG168 / STACK112 / SHARED1024**; CuTe PTX shrank 955476 → 942077 bytes and cubin 307264 → 300864 bytes. No runtime speedup or occupancy change is established.
