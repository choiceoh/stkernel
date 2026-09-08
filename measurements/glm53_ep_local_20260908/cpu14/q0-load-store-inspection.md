CPU14 retains eight static route-weight load sites, and all eight now follow a valid-ID branch. The ten Q0 adaptive store sites became ten plain `st.global.u64` sites.

| Compiler fact | CPU13 | CPU14 |
|---|---:|---:|
| Histogram / producer weight-load sites | 1 / 7 | 1 / 7 |
| Weight loads guarded by valid ID | 0 | 8 |
| Adaptive / plain Q0 store sites | 10 / 0 | 0 / 10 |
| Equal / varied scale store sites | 7 / 3 | 7 / 3 |
| PTX bytes | 941539 | 939211 |
| Cubin bytes | 298456 | 288664 |
| REG / STACK / SHARED | 168 / 112 / 1024 | 168 / 112 / 1024 |

The static weight-load count did not fall: the invalid-ID path now branches around those loads. These are compiler facts, not measured traffic or speed improvements.

Histogram, CPU13, decompressed PTX lines 290–296:

```ptx
	ld.global.b32 	%r11, [%rd51];
	add.s64 	%rd52, %rd28, %rd50;
	ld.global.b32 	%r792, [%rd52];
	setp.gt.u32 	%p30, %r11, 71;
	setp.eq.f32 	%p31, %r792, 0f00000000;
	or.pred 	%p32, %p30, %p31;
	@%p32 bra 	$L__BB0_39;
```

Histogram, CPU14, decompressed PTX lines 290–296:

```ptx
	ld.global.b32 	%r11, [%rd51];
	setp.gt.u32 	%p30, %r11, 71;
	@%p30 bra 	$L__BB0_40;
	add.s64 	%rd52, %rd28, %rd50;
	ld.global.b32 	%r792, [%rd52];
	setp.eq.f32 	%p31, %r792, 0f00000000;
	@%p31 bra 	$L__BB0_40;
```

First producer copy, CPU13, decompressed PTX lines 621–627:

```ptx
	ld.global.b32 	%r20, [%rd17];
	add.s64 	%rd18, %rd28, %rd63;
	ld.global.b32 	%r908, [%rd18];
	setp.gt.u32 	%p80, %r20, 71;
	setp.eq.f32 	%p81, %r908, 0f00000000;
	or.pred 	%p82, %p80, %p81;
	@%p82 bra 	$L__BB0_74;
```

First producer copy, CPU14, decompressed PTX lines 619–625:

```ptx
	ld.global.b32 	%r19, [%rd17];
	setp.gt.u32 	%p79, %r19, 71;
	add.s64 	%rd18, %rd28, %rd63;
	@%p79 bra 	$L__BB0_76;
	ld.global.b32 	%r908, [%rd18];
	setp.eq.f32 	%p80, %r908, 0f00000000;
	@%p80 bra 	$L__BB0_76;
```

CPU13 Q0 store lines: 1266, 1289, 1312, 1335, 1372, 1395, 1423, 1559, 1682, 1816.

CPU14 Q0 store lines: 1258, 1281, 1304, 1327, 1364, 1387, 1415, 1551, 1674, 1808.

The JSON records exact instructions, skip targets and parameter origins for all eight load sites and ten stores in each version. The source parameter reference is the SHA-verified inherited kernel; only the EP-local source differs in the two compiler source maps.

CPU13 source `18148116a7abb242741d5b112c8d735353d9fc71`; PTX SHA256 `21a5c860b7f02aa887dd1cd8e90a5953094842b83ee24b118697c2f4d334d2de`; cubin SHA256 `397eaf35853780c85b37ea3a9c8491f4e4fd9f0c3c9380e1ca298b419fe0f4a3`.

CPU14 source `881456a1f43fcf61de1bb5822b883dd2fd32e693`; PTX SHA256 `7c4bdc1d65f84314089dc399c39e91c90b12c699b34257de697c9fda151d1cc8`; cubin SHA256 `a657728bfb29e02e032b0fa43b69cf169fa5ac63b0cb74bf3954100211720dd4`.

Reproduce without compilation or GPU access:

```sh
python3 measurements/glm53_ep_local_20260908/cpu14/verify_q0_load_store_inspection.py
```

This checks archived receipt and payload hashes before regenerating and comparing the derived JSON/Markdown. It does not establish GPU numerics, race freedom, throughput or TTFT.
