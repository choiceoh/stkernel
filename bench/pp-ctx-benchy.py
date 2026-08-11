# llama-benchy pp2048 (prefill tok/s) + ctx_tg@dN (long-context decode)
import json, sys, time, urllib.request, random
URL="http://127.0.0.1:8000/v1/chat/completions"; MODEL="deepseek-v4-flash"
W="reactor harbor lattice quarry ember meridian syntax granite voltage cirrus tundra beacon ledger prism cobalt willow cascade anvil nocturne vellum".split()
def prompt(ntok,seed):
    rng=random.Random(seed); return " ".join(rng.choice(W) for _ in range(int(ntok/1.3)))
def pp(ntok,seed):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt(ntok,seed)+" End."}],
      "max_tokens":1,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"thinking":False}}).encode()
    req=urllib.request.Request(URL,body,{"Content-Type":"application/json"})
    t0=time.time(); tf=None; pt=0
    with urllib.request.urlopen(req,timeout=300) as r:
        for raw in r:
            l=raw.decode().strip()
            if not l.startswith("data: "): continue
            p=l[6:]
            if p=="[DONE]": break
            ev=json.loads(p)
            if ev.get("usage"): pt=ev["usage"].get("prompt_tokens",pt)
            ch=ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") and tf is None: tf=time.time()
    ttft=(tf or time.time())-t0; return pt, pt/ttft if ttft>0 else 0, ttft
def ctxtg(ctxN,seed):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt(ctxN,seed)+" Now write a long essay about tides."}],
      "max_tokens":128,"min_tokens":128,"ignore_eos":True,"stream":True,"stream_options":{"include_usage":True},"temperature":0.8,"chat_template_kwargs":{"thinking":False}}).encode()
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
            if ch and (ch[0].get("delta") or {}).get("content") and tf is None: tf=time.time()
    return comp
mode=sys.argv[1] if len(sys.argv)>1 else "pp"
if mode=="pp":
    print("=== pp2048 (prefill tok/s) ===")
    for r in range(4):
        pt,rate,ttft=pp(2048,r+1); print(f"  pp rep{r+1}: {pt} tok, {rate:6.1f} tok/s (TTFT {ttft:.2f}s)")
else:
    for d in [int(x) for x in sys.argv[2].split(",")]:
        rates=[]
        for r in range(2):
            t0=time.time(); comp=ctxtg(d,r*10+d); wall=time.time()-t0
            # decode-only rate approximated: comp/(wall - prefill_est). use bench-ctx style: comp / gen after first
            rates.append(comp/wall)  # rough end-to-end incl prefill
        print(f"  ctx_tg@d{d}: ~{sum(rates)/len(rates):.1f} tok/s (e2e incl prefill, comp {comp})")
