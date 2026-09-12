"""Compile SM121 or interpret the draft write without acquiring a GPU."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from engine.kernels.draft_observe import _write_context

if os.environ.get('TRITON_INTERPRET') != '1':
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile
    for t, heads in ((1, 1), (7, 2), (7, 4)):
        kernel = compile(ASTSource(_write_context,
            dict(C='*bf16', Weights='*bf16', Inv='*fp32', Positions='*i64', Slots='*i64', Valid='*i64', Field='*bf16'),
            constexprs=dict(T=t, L=5, HK=heads, D=128, W=256, FIELD_HK=heads,
                            SLOT_STRIDE=5*2*256*heads*128+128, LAYER_STRIDE=2*256*heads*128, EPS=1e-6)),
            target=GPUTarget('cuda',121,32), options={'num_warps':4})
        print('SM121 compiled', t, heads, kernel.metadata.shared)
else:
    # The interpreter's NumPy path does not reproduce BF16 arithmetic here.
    # Use FP32 to check indexing/masks; GPU tests require exact BF16 equality.
    from engine.kernels.norm_rope import norm_rope, warm
    torch.manual_seed(71)
    n,t,layers,heads,dim,window = 4,7,5,2,128,32
    field_heads = heads + 1
    row_size = layers*2*window*field_heads*dim
    backing = torch.full((5, row_size+128), -17., dtype=torch.float32)
    plain = torch.empty(5,layers,2,window,field_heads,dim)
    field = backing.as_strided(plain.shape, (row_size+128,*plain.stride()[1:]))
    weights = torch.randn(layers,dim)
    context = torch.randn(n,t,layers,2,heads,dim)
    slots = torch.tensor([4,2,1,0])
    pos = torch.arange(n*t).view(n,t)+29
    valid = torch.tensor([0,1,6,7])
    expected=field.clone()
    for layer in range(layers):
        key=norm_rope(context[:,:,layer,0].reshape(n*t,heads,dim),weights[layer],1e-6,pos.reshape(-1),10000.).view(n,t,heads,dim)
        for row in range(n):
            for token in range(int(valid[row])):
                expected[slots[row],layer,0,pos[row,token]%window,:heads]=key[row,token]
                expected[slots[row],layer,1,pos[row,token]%window,:heads]=context[row,token,layer,1]
    _write_context[(n*t,layers*heads)](context,weights,warm('cpu',dim,10000.),pos,slots,valid,field,
        t,layers,heads,dim,window,field_heads,field.stride(0),field.stride(1),1e-6,num_warps=4)
    torch.testing.assert_close(field.float(),expected.float(),rtol=.016,atol=.016)
    assert torch.equal(field[:,:,1],expected[:,:,1])
    assert torch.equal(field[slots[0]],expected[slots[0]])
    assert torch.equal(backing[:,row_size:],torch.full_like(backing[:,row_size:],-17.))
    print('FP32 interpreter PASS: all-layer keys/values, wrap, mixed accepted lengths, padded slots, ghost preservation')
assert not torch.cuda.is_initialized()
print('No CUDA device initialized; this is not GPU validation')
