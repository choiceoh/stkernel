# SF6 direct-prefill onepass: INVALID, no performance verdict

The canonical A/B finished at 2026-09-09 05:50:08 KST with supervisor/payload
return code 1. Both records exist. The canonical analyzer is INVALID and
comparison=null; the checker and tested source were not changed.

The baseline onepass failed exclusivity: 11 requests completed versus its own 8,
one request was still running after completion, and 101/233 traffic samples had
more than one running/queued request. Peak counts were 3 running and 2 waiting.
The canonical three-repetition metric cannot be salvaged by selecting a subset;
per-sample traffic timestamps were not retained to prove individual rep isolation.
B onepass exited 2 and the chain returned 1.

Both arms also failed the common MHC T16 selftest. All eight rank reports lack
MHC PASS/capture, so no valid prepared/runtime receipts were admitted. Their
latest failed snapshots are retained as failed, not promoted to accepted proof.
The same FP32 post_mix/comb_mix mismatch occurs in both arms: maxima
1.788139343e-7/2.384185791e-7, while residual and layer_input are exact and all
outputs finite. T16 dispatches the ordinary FP32 kernel; fp32=False is a test
option and does not imply BF16 kernel execution. Failure disables the MHC
consumer globally, including T<=8; the ordinary MHC and OSAR PDL are separate.
The eager/graph mismatch root cause remains unknown.

| Recorded metric | A: direct SF6 | B: raw scales, contaminated |
|---|---:|---:|
| Pooled decode steps/s | 20.28170 | 17.27348 |
| ms/step | 49.30553 | 57.89222 |
| Output tok/s | 61.95479 | 56.35038 |
| 2K warm TTFT, s | 0.83828 | 0.88910 |
| 32K TTFT, s | 10.88157 | 10.86199 |
| 128K TTFT, s | 42.22854 | 41.95131 |
| Quality | 18/18 | 18/18 |
| Korean corruption | 0 | 0 |
| Lowest host MemAvailable, GiB | 10.24701 | 10.21787 |

These are raw per-boot observations, not an improvement claim. A's individual
record and 147 memory samples pass their audit. B's individual record fails;
its 170 memory samples pass the 10 GiB guard. A cold_compile=true and B=false,
so their cold-prefill values are also incomparable. Net memory savings cannot
be attributed from these host minima or from the tensor-release counter alone.

Direct SF6 lifecycle was observed on every candidate rank: 42 packed layers,
3,604,414,464 packed bytes, one finalization releasing 4,756,340,736 original
scale bytes (4.4296875 GiB), M6 serving and zero fallbacks. Baseline SF6 markers
were all zero. All eight rank reports have actual KV 665, GMU 0.6429 and
compact/inline 0. Independent read-only review verified 16 latest receipt
artifact hashes, eight log/report hashes and marker re-parses, and all 63 source
hashes against frozen 85e25370. This establishes retained source/lifecycle
observations, not full kernel numerical/race correctness or runtime acceptance.

Session sf6-direct-0909v1, ticket 17888991282385029, launch
be907d63be5a4736a1fc232fbae9fd6a. Started 05:25:29 KST after 1.2 s queue wait;
payload 1478.6 s. Frozen source 85e25370779c2b8a6c9aaefa275b1dc8f27d60d0,
remote checkout /home/choiceoh/stkernel-sf6-direct-0909. Exact argv/spec are in
probes/sf6_direct_onepass_v1.json and probes/sf6_direct_prepare.json.

The complete 37 MiB raw final snapshot, including all failed attempts and logs,
is preserved at /tmp/sf6-direct-onepass-0909v1/final; the interim snapshot is
unchanged. This folder retains the original two-record JSONL, memory samples,
final supervisor receipt/exit, canonical INVALID analysis, diagnostic raw
per-boot numbers, and a SHA256 manifest of the complete snapshot. It does not
contain the full raw log/attempt archive. No failed receipt was rewritten.

No extra GPU run, source mutation on the admitted checkout, default promotion,
or merge was performed. The heartbeat is paused after reporting this outcome.
Earlier v5 evidence remains separate. The independently discovered external
traffic and common MHC failure require resolution before an accepted speed test.
