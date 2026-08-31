# tg128 with completion_tokens (all generated tokens incl reasoning), like
# llama-benchy. Concurrency sweep. tok/s = total completion_tokens across
# streams / wall (aggregate) — leaderboard ranks the peak over concurrency.
import os
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL="http://127.0.0.1:8000/v1/chat/completions"; MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4-flash")
def one(args):
    mt, seed = args
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":
      f"Write an extremely long detailed technical essay, at least 3000 words, about renewable energy. Never stop early. (ref {seed})"}],
      "max_tokens":mt,"ignore_eos":True,"min_tokens":mt,"stream":True,
      "stream_options":{"include_usage":True},"temperature":0.8,
      "chat_template_kwargs":{"thinking":False}}).encode()
    req=urllib.request.Request(URL,body,{"Content-Type":"application/json"})
    tf=None; comp=0
    with urllib.request.urlopen(req,timeout=600) as r:
        for raw in r:
            l=raw.decode().strip()
            if not l.startswith("data: "): continue
            p=l[6:]
            if p=="[DONE]": break
            ev=json.loads(p)
            if ev.get("usage"): comp=ev["usage"].get("completion_tokens",comp)
            ch=ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") and tf is None:
                tf=time.time()
    return comp
mt=int(sys.argv[1]) if len(sys.argv)>1 else 128
for C in [int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["1","2","5"])]:
    best=0
    for rep in range(3):
        t0=time.time()
        with ThreadPoolExecutor(max_workers=C) as ex:
            comps=list(ex.map(one,[(mt,rep*100+i) for i in range(C)]))
        wall=time.time()-t0
        agg=sum(comps)/wall
        best=max(best,agg)
        print(f"  tg{mt} c{C} rep{rep+1}: comp/stream {comps[0]} | agg {agg:6.1f} tok/s",flush=True)
    print(f"tg{mt} c{C} BEST(agg) {best:.2f}",flush=True)
