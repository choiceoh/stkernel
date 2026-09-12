"""MK fusion agrees with independent mHC equations under changed-input replay."""
import importlib.util
import unittest

from tests.image_kernels import PRESENT, REASON

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and PRESENT, "requires CUDA; " + REASON)
class MHCTests(unittest.TestCase):
    def test_reference_and_replay(self):
        from engine.kernels.dense.mhc import MHC
        from engine.modules.hyper_connection import mhc_pre, mhc_post
        from engine.profiles.glm53.net import rmsnorm
        torch.manual_seed(813)
        for lossless in (True,False):
            fn = torch.randn(24,16384,device='cuda')*.006
            if lossless:
                fn = fn.bfloat16().float()
            owner = MHC({'fn':fn})
            scale = torch.tensor([.2,.3,.4],device='cuda')
            base = torch.randn(24,device='cuda')*.1
            norm = torch.randn(4096,device='cuda',dtype=torch.bfloat16)
            for n in (1,6,12,24,32,36,48,64):
                x = torch.randn(n,4096,device='cuda',dtype=torch.bfloat16)
                res = torch.randn(n,4,4096,device='cuda',dtype=torch.bfloat16)
                post = torch.rand(n,4,1,device='cuda')
                comb = torch.rand(n,4,4,device='cuda')
                def call():
                    return owner('fn',x,res,post,comb,scale,base,norm,1e-5,1e-6,2.,20)
                def check(got):
                    rc = mhc_post(x,res,post,comb)
                    pm,cm,li = mhc_pre(rc,fn,scale,base,1e-5,1e-6,1e-6,2.,20)
                    li = rmsnorm(li,norm,1e-5)
                    for actual,expected in zip(got,(rc,pm,cm,li)):
                        rel = (actual.float()-expected.float()).norm()/expected.float().norm()
                        self.assertLess(rel.item(),.006,(lossless,n,rel.item()))
                check(call())
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): out=call()
                try:
                    for _ in range(3):
                        x.normal_();res.normal_();g.replay();check(out)
                finally:
                    g.reset()


if __name__=='__main__':unittest.main()
