"""Full-model finite-state bisection of the exact onepass 32K prompt.

Exclusive fleet only. Each variant starts with fresh caches; SP and NVFP4
are changed explicitly in the probe, never as a serving fallback.
"""
import argparse
import json
from pathlib import Path

import torch

from engine.base.comm import Comm
from engine.base.instruments import Recorder
from engine.profiles.glm53.boot import build, chat_renderer, tokenizer
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.net import Step


class Nonfinite(RuntimeError):
    pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks',required=True)
    ap.add_argument('--ckpt-meta',required=True)
    ap.add_argument('--drafter-dir',required=True)
    ap.add_argument('--prompt',required=True)
    ap.add_argument('--hold-dir',help='Private diagnostic commands; bounded at 1200 seconds')
    ap.add_argument('--requests',help='Run exact sequential prompts after production graph capture')
    args=ap.parse_args()
    comm=Comm.init(timeout_s=300)
    engine=None
    try:
        comm.prepare_oneshot()
        F,net,caches,engine,runner=build(comm,None,served(moe_static='t,r,sf6,q0',consume_scales=True),
            args.ranks,8.73,4,True,Recorder('finite-prefill'),execution='native',
            ckpt_meta=args.ckpt_meta,drafter_dir=args.drafter_dir)
        if args.requests:
            replay_requests(args, comm, net, caches, engine, runner)
            return
        prompt=Path(args.prompt).read_text()
        rendered=chat_renderer(args.ckpt_meta)([dict(role='user',content=prompt)],dict(thinking=True))
        tok=tokenizer(args.ckpt_meta)
        ids=torch.tensor(tok.encode(rendered).ids,device='cuda',dtype=torch.int64)
        print(json.dumps(dict(rank=comm.rank,tokens=len(ids))),flush=True)
        active={'layer':-1}
        def check(value,name):
            if not isinstance(value,torch.Tensor) or not value.is_floating_point():return
            bad=(~torch.isfinite(value)).any().int().reshape(1)
            own=bool(bad.item())
            if comm.all_reduce_max(bad).item():
                print(json.dumps(dict(rank=comm.rank,nonfinite=name,local=own,shape=list(value.shape))),flush=True)
                raise Nonfinite(name)
        original_linear=net.linear
        def linear(x,name):
            check(x,'linear/input/'+name)
            y=original_linear(x,name)
            check(y,'linear/output/'+name)
            return y
        net.linear=linear
        for method in ('_kda','_dsa','_dense','_moe'):
            original=getattr(net,method)
            def wrapped(layer,x,*args,_fn=original,_name=method,**kwargs):
                active['layer']=layer
                check(x,f'{_name}/{layer}/input')
                y=_fn(layer,x,*args,**kwargs)
                check(y,f'{_name}/{layer}/output')
                return y
            setattr(net,method,wrapped)
        sp=net.prefill_transport
        from dataclasses import replace
        lane_wrappers={}
        for name in ('conv_prefill','kda_chunk','kda_output_norm','mhc_pre','mhc_post','moe'):
            original=getattr(net.lanes,name)
            def lane(*args,_fn=original,_name=name,**kwargs):
                check(args[0],f'lane/{_name}/{active["layer"]}/input')
                out=_fn(*args,**kwargs)
                for i,value in enumerate(out if isinstance(out,tuple) else (out,)):
                    check(value,f'lane/{_name}/{active["layer"]}/output{i}')
                return out
            lane_wrappers[name]=lane
        net.lanes=replace(net.lanes,**lane_wrappers)
        for name in ('all_gather','reduce_scatter'):
            original=getattr(sp,name)
            def collective(x,_fn=original,_name=name):
                check(x,f'sp/{_name}/{active["layer"]}/input')
                out=_fn(x)
                check(out,f'sp/{_name}/{active["layer"]}/output')
                return out
            setattr(sp,name,collective)
        pairs={name:layer.nvfp4 for name,layer in net.dense.items() if name!='head'}
        for use_sp,use_nv in ((True,True),(False,True),(True,False),(False,False)):
            net.prefill_transport=sp if use_sp else None
            for name,pair in pairs.items():net.dense[name].nvfp4=pair if use_nv else None
            caches.reset()
            slot=caches.slots.take(0)
            caches.pool.reserve(0,len(ids)+6)
            print(json.dumps(dict(rank=comm.rank,variant=dict(sp=use_sp,nvfp4=use_nv))),flush=True)
            try:
                for ctx in range(0,len(ids),6912):
                    step=Step.prefill(ids[ctx:ctx+6912],ctx,0,slot)
                    caches.prepare(step)
                    h=net.forward(step,caches)
                    check(h,'final_hidden')
                    logits=net.head(h[-1:]);check(logits,'head')
                    selected=int(logits.argmax(-1).item())
                    print(json.dumps(dict(rank=comm.rank,context=ctx,length=len(step.ids),token=selected,
                                          text=tok.decode([selected]),passed=True)),flush=True)
            except Nonfinite as exc:
                print(json.dumps(dict(rank=comm.rank,failed=str(exc))),flush=True)
            finally:
                caches.pool.release(0);caches.slots.give(slot)
        comm.barrier()
    finally:
        if engine is not None:
            if engine.memory is not None:
                engine.memory.write(Path('/home/choiceoh/glm53-logs/st-native-diag-dumps')/f'memory-rank{comm.rank}.json')
            engine.close_decode()
        comm.close()


def replay_requests(args, comm, net, caches, engine, runner):
    """Keep the production graph/prefix order; inspect at request-step boundaries."""
    engine.capture_decode(4)
    tok=tokenizer(args.ckpt_meta)
    render=chat_renderer(args.ckpt_meta)
    original=engine._forward
    def forward(step):
        h,aux=original(step)
        bad=(~torch.isfinite(h)).any()
        if aux is not None:bad=bad|(~torch.isfinite(aux)).any()
        bad=comm.all_reduce_max(bad.int().reshape(1)).item()
        print(json.dumps(dict(rank=comm.rank,prefill=[(s.ctx,s.length) for s in step.segments],nonfinite=bool(bad))),flush=True)
        if bad:raise Nonfinite('prefill output')
        return h,aux
    engine._forward=forward
    target=engine.decode_graphs
    original_run=target.run
    def replay(step,shape=None):
        h,aux,logits=original_run(step,shape)
        bad=(~torch.isfinite(h)).any()|(~torch.isfinite(logits)).any()|(~torch.isfinite(aux)).any()
        if comm.all_reduce_max(bad.int().reshape(1)).item():
            print(json.dumps(dict(rank=comm.rank,nonfinite='decode',context=int(step.segments[0].ctx),shape=shape)),flush=True)
            raise Nonfinite('decode graph')
        return h,aux,logits
    target.run=replay
    for seq,request in enumerate(json.loads(Path(args.requests).read_text())):
        ids=tok.encode(render([dict(role='user',content=request['content'])],dict(thinking=True))).ids
        print(json.dumps(dict(rank=comm.rank,request=seq,tokens=len(ids))),flush=True)
        engine.add(seq,ids,max_new=request['max_tokens'])
        runner.submit(seq,len(ids))
        while runner.step() is not None:pass
        generated=engine.generated(seq)
        print(json.dumps(dict(rank=comm.rank,request=seq,generated=generated,text=tok.decode(generated))),flush=True)
    comm.barrier()
    if args.hold_dir:
        import time
        directory=Path(args.hold_dir)
        directory.mkdir(parents=True,exist_ok=True)
        namespace=dict(torch=torch,comm=comm,net=net,caches=caches,engine=engine,runner=runner,tok=tok,render=render,Step=Step)
        deadline=time.monotonic()+1200
        consumed=set()
        print('DIAGNOSTIC_CONTROL_READY',flush=True)
        while time.monotonic()<deadline:
            command=None
            if comm.rank==0:
                pending=sorted(p for p in directory.glob('*.py') if p.name not in consumed)
                if pending:
                    path=pending[0]
                    command=(path.name,path.read_text())
                    consumed.add(path.name)
            command=comm.broadcast_object(command)
            if command:
                name,source=command
                if source.strip()=='STOP':break
                print('DIAGNOSTIC_COMMAND '+name,flush=True)
                exec(compile(source,name,'exec'),namespace)
                torch.cuda.synchronize();comm.barrier()
                print('DIAGNOSTIC_DONE '+name,flush=True)
            else:time.sleep(1)


if __name__=='__main__':main()
