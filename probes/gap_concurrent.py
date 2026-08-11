# For the biggest gaps on main stream, what runs concurrently on OTHER streams?
# Distinguishes true GPU idle (nothing runs = optimizable) from overlap
# (another stream is the critical path).
import gzip,json,sys
from collections import defaultdict
doc=json.load(gzip.open(sys.argv[1],"rt"))
evs=[e for e in doc.get("traceEvents",[]) if e.get("ph")=="X" and "kernel" in e.get("cat","")]
by_tid=defaultdict(list)
for e in evs: by_tid[e.get("tid")].append((e.get("ts",0.0),e.get("dur",0.0),e.get("name","")))
for t in by_tid: by_tid[t].sort()
main=max(by_tid,key=lambda t:sum(d for _,d,_ in by_tid[t]))
seq=by_tid[main]
# all events sorted for concurrent lookup
allev=sorted((ts,ts+d,nm,tid) for tid in by_tid for ts,d,nm in by_tid[tid])
import re
def sh(x):
    x=re.sub(r"^void ","",x)
    if "deep_gemm" in x:
        m=re.search(r"impl<(\d+)u, (\d+)u, (\d+)u",x); return f"gemm<{m.group(2)},{m.group(3)}>" if m else "gemm"
    return re.match(r"([A-Za-z0-9_:]+)",x).group(1)[:26]
# find gaps prev=gemm<6,4096> next=_save_partial_states, sample a few
samples=[]
for i in range(1,len(seq)):
    g=seq[i][0]-(seq[i-1][0]+seq[i-1][1])
    if g>0.5 and sh(seq[i-1][2])=="gemm<6,4096>" and "save_partial" in seq[i][2]:
        gap_start=seq[i-1][0]+seq[i-1][1]; gap_end=seq[i][0]
        samples.append((g,gap_start,gap_end))
        if len(samples)>=3: break
print(f"main tid={main}. gemm<6,4096>->save_partial gaps sampled: {len(samples)}")
for g,gs,ge in samples:
    # what runs on other streams during [gs,ge]?
    conc=defaultdict(float)
    for ts,te,nm,tid in allev:
        if tid==main: continue
        ov=min(te,ge)-max(ts,gs)
        if ov>0.01: conc[(sh(nm),tid)]+=ov
    busy=sum(conc.values())
    print(f"\n  GAP {g:.1f}us [{gs:.0f}..{ge:.0f}]  concurrent-busy {busy:.1f}us ({100*busy/g:.0f}% filled):")
    for (nm,tid),d in sorted(conc.items(),key=lambda kv:-kv[1])[:5]:
        print(f"    {d:6.1f}us  {nm} (tid {tid})")
