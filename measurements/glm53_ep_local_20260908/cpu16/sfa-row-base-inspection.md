CPU16 hoists the physical-row SFA permutation into the lane 0 producer. Every emitted Q0 consumer site now loads the cached row base and combines only the SF field. This is static compiler evidence.

| Compiler fact | CPU14 | CPU16 |
|---|---:|---:|
| Equal / varied Q0 store sites | 7 / 3 | 7 / 7 |
| Prefix row-load sites | 8 in loop | 72 straightline |
| Consumer row-address operations | 9 ALU | address add + shared load + combine add |
| PTX bytes | 939211 | 964424 |
| Cubin bytes | 288664 | 300032 |
| REG / STACK / SHARED | 168 / 112 / 1024 | 168 / 112 / 1024 |

CPU16 producer has seven static row-base calculation/store copies. Representative PTX lines 889–896 contain eight bit-field ALUs, followed by cache-address +32 at line 897 and shared store at line 899. The unchanged physical row is stored at line 887. All seven copies were checked.

CPU14 prefix lines 371–383 use signed correction; CPU16 lines 365–378 use add 127, unsigned shift 7 and accumulation. The old eight-expert body loops at lines 480–484; the new prefix spans lines 364–723 with 72 row loads and no backedge.

The varied Q0 loop also grew from 3 to 7 static copies, while equal stays at 7. The static store count change from 10 to 14 reflects compiler unrolling, not more runtime routes or stores. PTX grew by 25,213 bytes and cubin by 11,368 bytes; no exact byte attribution or performance claim is made.

CPU14 first equal consumer, lines 1258–1271:

```ptx
	st.global.u64 [%rd131], %rd652;
	// end inline asm
	shl.b32 	%r1241, %r1229, 8;
	and.b32 	%r1242, %r1241, -32768;
	shl.b32 	%r1243, %r1229, 4;
	and.b32 	%r1244, %r1243, 496;
	shr.u32 	%r1245, %r1229, 3;
	and.b32 	%r1246, %r1245, 12;
	or.b32 	%r1247, %r39, %r1242;
	add.s32 	%r1248, %r1247, %r1244;
	add.s32 	%r1249, %r1248, %r1246;
	cvt.s64.s32 	%rd136, %r1249;
	add.s64 	%rd137, %rd29, %rd136;
	st.global.b8 	[%rd137], %rs113;
```

CPU16 first equal consumer, lines 1579–1588:

```ptx
	st.global.u64 [%rd156], %rd677;
	// end inline asm
	add.s32 	%r1632, %r1630, 32;
	// begin inline asm
	ld.shared.s32 %r1631, [%r1632];
	// end inline asm
	add.s32 	%r1649, %r38, %r1631;
	cvt.s64.s32 	%rd161, %r1649;
	add.s64 	%rd162, %rd30, %rd161;
	st.global.b8 	[%rd162], %rs113;
```

CPU14 first varied consumer, lines 1551–1564:

```ptx
	st.global.u64 [%rd110], %rd653;
	// end inline asm
	shl.b32 	%r1104, %r1065, 8;
	and.b32 	%r1105, %r1104, -32768;
	shl.b32 	%r1106, %r1065, 4;
	and.b32 	%r1107, %r1106, 496;
	shr.u32 	%r1108, %r1065, 3;
	and.b32 	%r1109, %r1108, 12;
	or.b32 	%r1110, %r41, %r1105;
	add.s32 	%r1111, %r1110, %r1107;
	add.s32 	%r1112, %r1111, %r1109;
	cvt.s64.s32 	%rd112, %r1112;
	add.s64 	%rd113, %rd29, %rd112;
	st.global.b8 	[%rd113], %rs114;
```

CPU16 first varied consumer, lines 1844–1853:

```ptx
	st.global.u64 [%rd107], %rd678;
	// end inline asm
	add.s32 	%r1370, %r1336, 32;
	// begin inline asm
	ld.shared.s32 %r1369, [%r1370];
	// end inline asm
	add.s32 	%r1376, %r40, %r1369;
	cvt.s64.s32 	%rd109, %r1376;
	add.s64 	%rd110, %rd30, %rd109;
	st.global.b8 	[%rd110], %rs114;
```

The JSON binds receipt/submission revisions, decompressed PTX/cubin SHA256 and all relevant repeated instruction sites. Each original compressed payload was hashed; its decompressed bytes were checked against the receipt size and SHA256. CPU15 remains FAIL; its saved PTX/cubin are byte-identical to CPU14 under the same 13.0.3 runtime as CPU16, so it is only a compiler-payload compatibility bridge.

CPU14 source `881456a1f43fcf61de1bb5822b883dd2fd32e693`; PTX SHA256 `7c4bdc1d65f84314089dc399c39e91c90b12c699b34257de697c9fda151d1cc8`; cubin SHA256 `a657728bfb29e02e032b0fa43b69cf169fa5ac63b0cb74bf3954100211720dd4`.

CPU16 source `111fff02f66f3e07a8332da1117afb75b161f0d6`; PTX SHA256 `aa04b5bd7ed9209c7d9826361439e4e3c331b0e160152deed3074332b57b65ab`; cubin SHA256 `0dc09f29c66e4625c71519d101a1708f321613db7b5eb686750f37de98d9faa5`.

No compilation, test repetition, GPU call or queue change was performed. Added shared loads, altered unrolling and binary growth require actual GPU evaluation; this report proves neither numerics nor throughput/TTFT improvement.
