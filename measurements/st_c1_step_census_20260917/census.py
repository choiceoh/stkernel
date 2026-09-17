"""Fold this record's raw CUPTI table into kernel families. `python3 census.py [profile.json]`."""
import json
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                   os.path.join(HERE, "profile-c1-3a84c197.json")))
tot = d["device_us_per_step"]; ks = d["kernels"]; steps = d["steps"]
print("steps", steps, "| device_us_per_step", tot, "| distinct kernels", len(ks))
print("fields:", list(ks[0]))
PAT = [("moe_static","MoE b12x static"),("b12x","MoE b12x 기타"),("mhc","mHC"),("mla","MLA"),
       ("kda","KDA"),("gated_delta","KDA"),("chunk_","KDA"),("conv","causal conv"),
       ("kpool","kpool"),("hadamard","kpool"),("fwht","kpool"),("indexer","indexer"),
       ("qsa","QSA"),("dsa","DSA/indexer"),("oneshot","one-shot 통신"),("one_shot","one-shot 통신"),
       ("nccl","통신 NCCL"),("allreduce","one-shot 통신"),("publish","패킷 publish"),
       ("router","router"),("topk","top-k"),("gemm","dense GEMM"),("cublas","dense GEMM"),
       ("w4a8","dense GEMM"),("sample","샘플링"),("draft","드래프터"),("embed","임베딩"),
       ("norm","노름"),("rope","회전"),("pack","입력 pack"),("memcpy","복사/이동"),
       ("copy","복사/이동"),("elementwise","포인트와이즈"),("vectorized","포인트와이즈"),
       ("cast","포인트와이즈"),("fill","포인트와이즈")]
def fam(n):
    s = n.lower()
    for p, name in PAT:
        if p in s: return name
    return "기타"
key = "us_total"
print("time field:", key)
agg = {}
for k in ks:
    v = float(k.get(key) or 0); f = fam(k["kernel"])
    a = agg.setdefault(f, [0.0, 0.0, 0])
    a[0] += v; a[1] += float(k.get("calls", 0)); a[2] += 1
acc = 0.0
print("%-20s %10s %7s %13s %8s" % ("가족", "us/step", "%", "launch/step", "kernels"))
for f, (us, calls, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    per = us / steps; acc += per
    print("%-20s %10.0f %6.1f%% %13.1f %8d" % (f, per, 100*per/tot, calls/steps, n))
print("%-20s %10.0f %6.1f%%" % ("합계", acc, 100*acc/tot))
print()
print("=== '기타' 로 빠진 상위 커널 ===")
for k in sorted([k for k in ks if fam(k["kernel"]) == "기타"], key=lambda k: -float(k.get(key) or 0))[:12]:
    print("  %8.0f us/step  %6.1f launch  %s" % (float(k[key])/steps, float(k.get("calls",0))/steps, k["kernel"][:110]))
