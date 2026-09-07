# Check4: candidate launched, numerical gate failed

Normal fleet GO: 2026-09-08 02:41:58 KST. Probe: 02:43:10–02:44:13.
Exact incoming container recovery and outer exit 1 completed at 02:46:51.
Health 200 was independently observed at 02:50. The incoming profile belonged
to the prior experiment; recovery restored its exact identity.

All four BF16 fallback cases passed. Every active 6144/6912/8192 balanced/skew
case failed on every valid row, including local MoE before TP transport.
Independent stock/changed/local controls and M128 capture passed. FP8,
sanitizers and direct TTFT were blocked. Numerically wrong timings are not
speedup evidence.

The subsequent audit identified a Q0 scale-address error: it divided physical
rows by logical M64 although global scale atoms are M128. Row 64's first scale
went to 32768 instead of 8; near the end this exceeds the scale allocation.
The follow-up separates scale coordinates from M64 task coordinates in all
four producer paths. Only fresh GPU evidence can establish whether this fully
explains the observed errors. Raw logs and complete recovery evidence are
preserved here. Do not rerun unchanged check4 or serving1.
