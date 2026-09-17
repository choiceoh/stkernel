"""Post-convolution fusion preserves residual state and per-block boundaries."""
import unittest
from unittest.mock import patch
import torch
from engine.kernels.draft_conv import tap_add_norm,tap_mix
from engine.kernels.common.norm_rope import add_norm


class PostNormTests(unittest.TestCase):
    def test_cpu_matches_the_original_composition_and_preserves_inputs(self):
        torch.manual_seed(41)
        for n,t,g in ((1,8,4),(3,8,4),(2,7,8)):
            x,res=[torch.randn(n*t,32,dtype=torch.bfloat16) for _ in range(2)]
            delta=torch.randn(n*t,2,2,32//g,dtype=torch.bfloat16)[:,1]
            base=torch.randn(2,32,dtype=torch.bfloat16);w=torch.randn(32,dtype=torch.bfloat16)
            untouched=[v.clone() for v in (x,res,delta,base,w)]
            expected=add_norm(res,tap_mix(x,delta,base,g,t),w,1e-6)
            actual=tap_add_norm(x,delta,base,res,w,1e-6,g,t)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip((x,res,delta,base,w),untouched):self.assertTrue(torch.equal(a,b))
            self.assertNotEqual(actual[0].data_ptr(),res.data_ptr())

    def test_full_drafter_single_and_batched_blocks_match_unfused_boundaries(self):
        from tests.test_engine_drafter import DrafterTests
        d,field,dev=DrafterTests().make_full_drafter()
        # Expand the tiny fixture to two identical layers to exercise the
        # MLP -> next input norm seam, in addition to the attention seam.
        from dataclasses import replace
        d.F=replace(d.F,layers=2)
        d.p.update({name.replace('layers.0.','layers.1.'):v.clone()
                    for name,v in list(d.p.items()) if name.startswith('layers.0.')})
        field=field.repeat(1,2,1,1,1,1)
        t=d.k+1;n=2
        ids=torch.arange(n*t,device=dev)%20
        positions=torch.arange(t,device=dev).repeat(n)+5
        slots=torch.tensor([0,1],device=dev);ctx=torch.tensor([5,5],device=dev)
        def reference(x,delta,base,residual,weight,block):
            return add_norm(residual,tap_mix(x,delta,base,d.F.conv_group,block),weight,d.F.rms_eps)
        for batched in (False,True):
            def run():
                return d.block_rows(ids,positions,slots,ctx,field,n,t) if batched else d.block(ids[:t],positions[:t],field[0],5)
            with patch.object(d,'_post_conv_norm',side_effect=reference):expected=run()
            actual=run();torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_invalid_block_and_shape_are_rejected(self):
        x=torch.zeros(8,32,dtype=torch.bfloat16);d=torch.zeros(8,2,8,dtype=x.dtype)
        base=torch.zeros(2,32,dtype=x.dtype);w=torch.ones(32,dtype=x.dtype)
        for block,group,eps in ((3,4,1e-6),(8,0,1e-6),(8,4,float('nan'))):
            with self.assertRaises(ValueError):tap_add_norm(x,d,base,x,w,eps,group,block)


if __name__=='__main__':unittest.main()
