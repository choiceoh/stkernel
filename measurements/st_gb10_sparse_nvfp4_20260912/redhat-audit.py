"""Read-only CPU sampling of existing rank-file FP4 sparsity; no GPU use."""
import json, os, struct
import numpy as np

path='/ranks/rank0of4.safetensors'
fd=os.open(path, os.O_RDONLY)
header_bytes=struct.unpack('<Q',os.pread(fd,8,0))[0]
header=json.loads(os.pread(fd,header_bytes,8))
base=8+header_bytes
results=[]
for layer in (3,15,30,44):
    for name in ('w13','w2'):
        key=f'L{layer}.moe.{name}'
        spec=header[key]
        e,n,half_k=spec['shape']
        size=n*half_k
        for expert in (0,73,144,287):
            data=os.pread(fd,size,base+spec['data_offsets'][0]+expert*size)
            assert len(data)==size
            raw=np.frombuffer(data,dtype=np.uint8).reshape(n,half_k)
            sf_spec=header[key+'_sf']
            sf_size=(sf_spec['data_offsets'][1]-sf_spec['data_offsets'][0])//e
            sf_bytes=os.pread(fd,sf_size,base+sf_spec['data_offsets'][0]+expert*sf_size)
            assert len(sf_bytes)==sf_size
            sf=np.frombuffer(sf_bytes,dtype=np.uint8)&0x7f
            assert not (sf==0x7f).any(), 'NaN scales invalidate the audit'
            assert not (sf==0).any(), 'Zero scales need expanded effective-zero analysis'
            zero_lo=(raw&7)==0
            zero_hi=((raw>>4)&7)==0
            active_pair=(raw&0x77)!=0
            eligible=(active_pair.reshape(n,-1,4).sum(-1)<=2)
            # Original weight tiles: 16 output rows, K128. Every group must fit.
            tiles=eligible.reshape(n//16,16,(half_k*2)//128,16).all(axis=(1,3))
            results.append(dict(layer=layer,weight=name,expert=expert,shape=[n,half_k*2],
                values=int(raw.size*2),zero_values=int(zero_lo.sum()+zero_hi.sum()),
                pair4of8_groups=int(eligible.size),eligible_groups=int(eligible.sum()),
                tiles16x128=int(tiles.size),eligible_tiles=int(tiles.sum()),read_bytes=size+sf_size,
                scales_checked=sf_size))
os.close(fd)
tot={k:sum(x[k] for x in results) for k in ['values','zero_values','pair4of8_groups','eligible_groups','tiles16x128','eligible_tiles','read_bytes']}
tot.update(zero_percent=100*tot['zero_values']/tot['values'],pair4of8_eligible_percent=100*tot['eligible_groups']/tot['pair4of8_groups'])
print(json.dumps(dict(method='CPU only, packed FP4 zero magnitude including -0; four fixed experts in four layers; every sampled block scale verified finite and nonzero',totals=tot,samples=results),indent=2))
