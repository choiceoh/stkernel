# B8 canonical JIT log audit

Canonical prefix (290411 bytes, SHA2801c46f…) and after (361198 bytes, SHA7ab82b…) exactly match the retained B boot log. The immutable readback preserved a70787-byte delta; source407271d9.

The log reports seven inference JIT events: six at16:49:00–02 in the first2K request (TileLang mhc_pre_big_fuse_with_norm_tilelang; Triton _finish, _copy_pad, _compute_local_logits_stats_kernel, _rejection_kernel, _resample_kernel), plus _gumbel_sample_kernel at16:50:44 in fixed2K rep0. These are actual monitor reports, not deductions from `cold_compile`.

The fourth POST corresponds to the canonical32K request by the unchanged sequential harness and exclusive eight-request accounting. Its exact preserved segment is delta lines49–93 (full captured log1607–1651); timestamped lines span16:49:20–46. It contains no compilation/JIT message. Recorded TTFT11.702632s is a single combined request, even though the compatibility record repeats it into cold/warm fields. There is no logged basis to attribute this32K TTFT to compilation.

The JIT monitor was active in warn mode before requests at16:48:31. Its implementation/suppression and non-head compiler hooks were not audited; absence of messages is not a no-JIT guarantee. Warning durations are not recorded. No TTFT correction or performance causality is inferred. `bench/onepass.py:421–422` merely copies MK_COLD_COMPILE=1 as a first-boot marker; lines445–504 establish sequential request order and combined-context behavior.

Original bytes, exact event lines, hashes and read-only before/after stat identity are in audit.json/readback.json and the three .raw files. No GPU requests, service changes or source edits occurred.
