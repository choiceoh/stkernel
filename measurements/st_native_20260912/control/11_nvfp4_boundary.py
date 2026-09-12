import json,flashinfer
caches.reset();slot=caches.slots.take(0);caches.pool.reserve(0,6912)
step=Step.prefill(ids[:6912],0,0,slot);caches.prepare(step)
x=net.embed(step.ids).chunk(4,0)[comm.rank];res=x[:,None,:].expand(x.shape[0],net.F.hc,net.F.hidden).contiguous()
_,_,x=net._hc_pre(0,res,'attn');fixed_x=net.prefill_transport.all_gather(x.contiguous()).clone()
layer=net.dense['L0.kda.in_proj'];weight,ws,wscale,padded=layer.nvfp4
reference=None
for mode in ('none','quant_sync','none','scale_sync','none','gemm_sync'):
    worst=0.
    for attempt in range(20):
        flat=fixed_x.clone()
        scale=(2688./flat.abs().amax().float().clamp_min(1e-12)).reshape(1)
        if mode=='scale_sync':torch.cuda.current_stream().synchronize()
        data,sf=flashinfer.nvfp4_quantize(flat,scale)
        if mode=='quant_sync':torch.cuda.current_stream().synchronize()
        alpha=(1./(scale*wscale)).float()
        output=torch.empty(flat.shape[0],padded,device=x.device,dtype=x.dtype)
        flashinfer.mm_fp4(data,weight.T,sf,ws.T,alpha,torch.bfloat16,output,16,False,'cutlass')
        if mode=='gemm_sync':torch.cuda.current_stream().synchronize()
        if reference is None:torch.cuda.synchronize();reference=output.clone()
        diff=(output.float()-reference.float()).abs().max().item();worst=max(worst,diff)
    print(json.dumps(dict(rank=comm.rank,nv_boundary=mode,worst=worst)),flush=True)
caches.pool.release(0);caches.slots.give(slot)
del reference,fixed_x,flat,scale,data,sf,alpha,output,x,res
