# SPDX-License-Identifier: Apache-2.0
# usage: python3 tools/trace_step_composition.py <profiler .pt.trace.json.gz>  (rank-0 decode trace; steps are cut at _gather_block_tables_kernel)
import gzip, json, sys, collections, statistics
p = sys.argv[1]
with gzip.open(p, "rt") as fh:
    tr = json.load(fh)
ev = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])
# step boundaries: the first _prepare_uniform_decode_kernel... use _gather_block_tables_kernel (once per step in prep)
starts = [i for i, e in enumerate(ev) if e["name"].startswith("_gather_block_tables_kernel")]
print("kernels", len(ev), "steps (gather_block_tables)", len(starts))
def cat(n):
    if n.startswith("k_oneshot"): return "AR k_oneshot"
    if "ncclDevKernel" in n: return "nccl"
    if "moecute" in n or "moe_static" in n: return "MoE expert kernel"
    if "deep_gemm::sm120_fp8_fp4_gemm" in n: return "deep_gemm fp8/fp4 GEMM"
    if "deep_gemm" in n: return "deep_gemm other"
    if "cutlass_80_wmma" in n or "splitKreduce" in n or "gemmSN" in n: return "bf16/cublas GEMM"
    if "fused_recurrent" in n or "conv1d" in n or "layer_norm_gated" in n: return "KDA"
    if "BatchMLAPaged" in n or "concat_and_cache_mla" in n: return "MLA decode"
    if "mqa_logits" in n or "topKPerRow" in n or "expand_pools" in n or "kpool" in n.lower(): return "indexer"
    if "mhc" in n: return "MHC"
    if "per_token_group_quant" in n or "act_and_mul" in n or "single_group_topk" in n or "_deneb_gate" in n: return "MoE glue"
    if "flashinfer::sampling" in n or "_selector_walk" in n or "_cache_draft" in n or "kernel_mha" in n: return "drafter/sampler"
    if n.startswith("triton_poi") or "at::native" in n or "elementwise" in n: return "elementwise glue"
    return "other"
tot = collections.defaultdict(list); cnt = collections.defaultdict(list); steplen = []; idle = []
ar_durs = []
for a, b in zip(starts[5:-1], starts[6:]):
    seg = ev[a:b]
    t0, t1 = seg[0]["ts"], seg[-1]["ts"] + seg[-1]["dur"]
    steplen.append(t1 - t0)
    busy = sum(e["dur"] for e in seg)
    idle.append(t1 - t0 - busy)
    s = collections.Counter(); c = collections.Counter()
    for e in seg:
        k = cat(e["name"]); s[k] += e["dur"]; c[k] += 1
        if k == "AR k_oneshot": ar_durs.append(e["dur"])
    for k in s: tot[k].append(s[k]); cnt[k].append(c[k])
n = len(steplen)
print(f"steps analysed {n}: step len median {statistics.median(steplen)/1000:.2f} ms, idle(gaps) median {statistics.median(idle)/1000:.2f} ms")
rows = sorted(tot.items(), key=lambda kv: -statistics.median(kv[1]))
for k, v in rows:
    print(f"{statistics.median(v)/1000:7.2f} ms  {statistics.median(cnt[k]):6.0f}/step  {k}")
ar_durs.sort()
print("k_oneshot dur percentiles (us): p10 %.1f p50 %.1f p90 %.1f max %.1f, n=%d" % (ar_durs[len(ar_durs)//10], ar_durs[len(ar_durs)//2], ar_durs[len(ar_durs)*9//10], ar_durs[-1], len(ar_durs)))
h = collections.Counter(int(d // 10) * 10 for d in ar_durs)
print("k_oneshot histogram (10us bins):", sorted(h.items())[:15])
