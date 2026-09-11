"""Full GLM TP4 validation through the real runner and HTTP admission path.

Run in four private containers. This probe adds measurement hooks locally;
production request handling is unchanged. Timing reports committed bursts,
not a fabricated per-token stream. The canonical Korean document and answer
rules come from bench/onepass.py; this is an ST transport adapter, not a
claim that the legacy vLLM service was run in the same boot.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

import torch

from engine.base.comm import Comm
from engine.base.instruments import Recorder
from engine.base.serve import Server
from engine.profiles.glm53.boot import build, tokenizer
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.adapter import NullDrafter


def load_bench(name):
    root = Path(__file__).resolve().parents[1] / "bench"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), root/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data), ensure_ascii=False), flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks',required=True)
    ap.add_argument('--ckpt-meta',required=True)
    ap.add_argument('--drafter-dir',required=True)
    ap.add_argument('--tier-dir',required=True)
    ap.add_argument('--port',type=int,default=29692)
    ap.add_argument('--contexts',type=int,nargs='+',default=[2000,32000,128000])
    ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args()
    comm=Comm.init(world=4)
    engine=None
    try:
        rec=Recorder(f'full-r{comm.rank}')
        _,net,caches,engine,runner=build(comm,None,served(),args.ranks,8.73,4,True,rec,
            max_new=32,tier_dir=args.tier_dir,ckpt_meta=args.ckpt_meta,drafter_dir=args.drafter_dir)
        emit('loaded',rank=comm.rank,phases=rec.table(),free_gib=torch.cuda.mem_get_info()[0]/2**30)
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(args.ckpt_meta)
        template=(Path(__file__).resolve().parents[1]/'launchers/chat_template_mm_v2.jinja').read_text()
        def encode(content,thinking=False):
            return tok.apply_chat_template([dict(role='user',content=content)],chat_template=template,
                add_generation_prompt=True,tokenize=True,thinking=thinking)
        prompt=encode('대한민국의 수도를 한 단어로 답해줘.')
        def direct(ids,limit):
            engine.add(0,ids,max_new=limit);runner.submit(0,len(ids))
            while 0 not in runner.idle: runner.step()
            result=engine.generated(0)
            runner.cancel(0);engine.forget(0)
            return result
        # Explicit target-only eager baseline, before any graph is captured.
        drafter=engine.drafter; engine.drafter=NullDrafter()
        baseline=direct(prompt,24)
        engine.drafter=drafter
        emit('eager_target',rank=comm.rank,ids=baseline,text=tok.decode(baseline))
        with rec.phase('capture decode'): engine.capture_decode(4)
        emit('captured',rank=comm.rank,shapes=list(engine.decode_graphs.graphs.graphs),
             phases=rec.table(),free_gib=torch.cuda.mem_get_info()[0]/2**30)
        candidate=direct(prompt,24)
        emit('graph_dflash',rank=comm.rank,ids=candidate,text=tok.decode(candidate),
             equals_eager=candidate==baseline,accepted=engine.accepted_total,drafted=engine.drafted_total)
        if args.smoke_only:
            assert '서울' in tok.decode(candidate), 'full-model smoke did not answer Seoul'
            return
        # Full requests enter through Server, including automatic NVMe parking.
        server=Server(engine,runner,comm,port=args.port,host='127.0.0.1',tokenizer=tokenizer(args.ckpt_meta))
        steps=[]; phases=[]; results=[]; errors=[]
        original_decode,original_prefill=engine.decode,engine.prefill
        def decode(seqs,blocks,slots):
            counts=[len(engine.tokens[s]) for s in seqs]
            t0=time.perf_counter(); result=original_decode(seqs,blocks,slots); end=time.perf_counter()
            if comm.rank==0: steps.append(dict(kind='decode',seqs=list(seqs),start=t0,end=end,
                committed=[len(engine.tokens[s])-n for s,n in zip(seqs,counts)]))
            return result
        def prefill(seq,start,tokens,blocks,slot):
            t0=time.perf_counter();result=original_prefill(seq,start,tokens,blocks,slot);end=time.perf_counter()
            if comm.rank==0: steps.append(dict(kind='prefill',seq=seq,tokens=tokens,start=t0,end=end))
            return result
        engine.decode,engine.prefill=decode,prefill
        for name in ('park','resume'):
            original=getattr(runner,name)
            def measured(seq,*args,original=original,name=name,**kwargs):
                t0=time.perf_counter(); n=original(seq,*args,**kwargs); end=time.perf_counter()
                if comm.rank==0: phases.append(dict(kind=name,seq=seq,key=kwargs.get('key',args[0] if args else seq),bytes=n,start=t0,end=end))
                return n
            setattr(runner,name,measured)
        def post(ids,limit=360,conversation=None):
            payload=dict(ids=ids,max_tokens=limit,temperature=0.)
            if conversation is not None: payload['conversation']=conversation
            req=urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/completions',
                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=3600) as response: return json.load(response)
        def client():
            try:
                os.environ['BENCH_MODEL']='st-glm53-validation'
                one=load_bench('onepass.py'); quality=load_bench('check-quality.py'); korean=load_bench('korean-corruption.py')
                for ctx in args.contexts:
                    doc=quality.build(ctx,7+ctx)
                    questions='\n'.join(f'{i+1}. {q}' for i,(_,q,_) in enumerate(quality.FACTS))
                    text=f'문서:\n{doc}\n\n{one.INSTRUCTION_COMBINED}{questions}'
                    result=post(encode(text,thinking=True),1080)
                    low=result['text'].lower()
                    hits=[all(any(alt in low for alt in group) for group in expect) for expect in one.FACT_EXPECT]
                    damage=korean.scan(result['text'],len(result['ids'])==1080)
                    row=dict(context_target=ctx,hits=hits,corruption=damage,**result)
                    results.append(row);emit('quality',**row)
                    assert all(hits), f'retrieval failure at {ctx}'
                    assert not any(v for k,v in damage.items() if k not in korean.INFORMATIONAL), 'Korean corruption'
                first=post(prompt,32)
                follow=post(encode('그 도시의 대표적인 궁궐 이름 하나만 답해줘.'),32,first['conversation'])
                emit('nvme_continuation',first=first,second=follow)
                # Concurrent admission, followed by requests arriving during decode.
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures=[pool.submit(post,encode(f'한국의 계절 {season}의 특징을 한국어로 설명해줘.'),128)
                             for season in ('봄','여름','가을','겨울')]
                    for future in futures: emit('concurrent',**future.result())
            except BaseException as exc:
                errors.append(repr(exc));emit('client_error',error=repr(exc))
            finally: server.alive=False
        httpd=server._serve_http() if comm.rank==0 else None
        if comm.rank==0: threading.Thread(target=client,daemon=True).start()
        try:
            while True:
                stop=comm.broadcast_object(not server.alive if comm.rank==0 else None)
                if stop: break
                if not server.once(): time.sleep(.002)
        finally:
            if httpd is not None: httpd.shutdown();httpd.server_close()
        emit('full_results',rank=comm.rank,passed=not errors,accepted=engine.accepted_total,
             drafted=engine.drafted_total,steps=steps,tier=phases,errors=errors)
        failed=comm.broadcast_object(bool(errors) if comm.rank==0 else None)
        assert not failed,'full-model validation failed'
    finally:
        if engine is not None: engine.close_decode()
        comm.close()


if __name__=='__main__':main()
