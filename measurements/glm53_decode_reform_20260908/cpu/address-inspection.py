import os, sys, runpy
os.environ['CUTE_DSL_ARCH']='sm_121a'
import torch
import cutlass.cute as cute
torch.cuda.is_available=lambda: True
torch.cuda.get_device_capability=lambda *a,**kw:(12,1)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
original=md.MoEStaticKernelV4._setup_attributes
def inspect_setup(self, hidden_size):
    original(self, hidden_size)
    print('A2_LAYOUT',self.decode_reform,self.a2_smem_layout,flush=True)
    rows=[]
    for row in range(8):
        outer=int(cute.crd2idx((row,0,0),self.a2_smem_layout.outer))
        composed=int(cute.crd2idx((row,0,0),self.a2_smem_layout))
        rows.append((row,outer,composed//2, row*64+(((row>>1)&3)<<4)))
    print('ROW outer_nibbles,current_raw_byte,legacy_byte',rows,flush=True)
md.MoEStaticKernelV4._setup_attributes=inspect_setup
sys.argv=['b12x_static_compile_check.py','--specs','t|t,r','--m','2','--max-rows','640']
runpy.run_path('/repo/probes/b12x_static_compile_check.py',run_name='__main__')
