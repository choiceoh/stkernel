"""Sampled selector preserves conditional distributions and keyed draws."""
import unittest
from unittest.mock import patch

import torch

from engine.kernels.draft_sample import sampled_walk, _by_torch


class SampledWalkTests(unittest.TestCase):
    def test_conditional_rows_ties_and_private_support(self):
        scores = torch.full((2,3,4,4), -1000.)
        scores[:,0,0,2] = 1.
        scores[:,1,2,3] = 1.
        scores[:,2,3,1] = 1.
        cand = torch.arange(24).view(2,3,4)
        for greedy, last in ((True,False),(False,True)):
            actual = sampled_walk(scores,cand,torch.tensor([0.,.8]),torch.ones(2,3),
                                  greedy_rows=greedy,last_mass=last)
            self.assertEqual(actual[0].tolist(),[[2,7,9],[14,19,21]])
            self.assertTrue(torch.equal(actual[2].sum(-1),torch.ones(2,3)))
            self.assertNotEqual(actual[1].data_ptr(),cand.data_ptr())
            self.assertTrue(torch.equal(actual[1],cand))
        zeros = torch.zeros(2,3,4,4)
        out,_,q = sampled_walk(zeros,cand,torch.tensor([0.,1.]),torch.ones(2,3))
        self.assertEqual(out.tolist(),[[0,4,8],[15,19,23]])
        self.assertEqual(q[0,0].tolist(),[1.,0.,0.,0.])
        self.assertEqual(q[1,0].tolist(),[.25]*4)

    def test_full_drafter_sampled_entry_points_use_the_same_distributions(self):
        from tests.test_engine_drafter import DrafterTests
        with patch('torch.cuda.is_available', return_value=False):
            d,field,dev = DrafterTests().make_full_drafter()
        n,k = 2,d.k
        slots = torch.tensor([0,1],device=dev)
        anchors = torch.tensor([4,7],device=dev)
        positions = torch.tensor([5,5],device=dev)
        temps = torch.tensor([0.,.8],device=dev)
        uniforms = torch.linspace(0.,1.,n*k,device=dev).reshape(n,k)
        def rows():
            return d.propose_rows(field,slots,anchors,positions,temps=temps,uniforms=uniforms,vocab=21)
        def single():
            return d.propose_sampled_tensor(anchors[:1],5,field[0],.8,uniforms[0],21)
        for call in (rows,single):
            with patch('engine.kernels.draft_sample.sampled_walk',side_effect=_by_torch):
                expected = call()
            actual = call()
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_shape_and_dtype_errors(self):
        scores = torch.zeros(2,7,16,16)
        cand = torch.zeros(2,7,16,dtype=torch.int64)
        temps,uniforms = torch.ones(2),torch.zeros(2,7)
        for args in ((scores[:,0],cand,temps,uniforms),
                     (scores,cand,temps,uniforms[:,:6]),
                     (scores,cand.float(),temps,uniforms)):
            with self.assertRaises(ValueError):
                sampled_walk(*args)

    @unittest.skipUnless(torch.cuda.is_available(), 'keyed draw boundaries need the CUDA kernel')
    def test_uniforms_adjacent_to_conditional_cdf_boundaries(self):
        from engine.base.sampler import _inverse_cdf
        for n in (1, 4):
            k, c = 7, 16
            torch.manual_seed(403+n)
            scores = torch.randn(n,k,c,c,device='cuda')
            cand = torch.arange(n*k*c,device='cuda').reshape(n,k,c)
            temps = torch.full((n,),.8,device='cuda')
            rows = torch.arange(n,device='cuda')
            for last_mass in (False, True):
                for direction in (-1, 0, 1):
                    u = torch.empty(n,2*k+1,device='cuda')[:,:k]
                    prev = torch.zeros(n,dtype=torch.int64,device='cuda')
                    for step in range(k):
                        p = torch.softmax(scores[rows,step,prev]/temps[:,None],-1)
                        walk = p.cumsum(-1)
                        mass = walk[:,-1] if last_mass else p.sum(-1)
                        cut = walk[:,(step*3)%c]/mass
                        if direction:
                            cut = torch.nextafter(cut,torch.full_like(cut,float('inf') if direction>0 else -float('inf')))
                        u[:,step] = cut.clamp(0.,1.)
                        if last_mass:
                            pick = torch.searchsorted(walk,(u[:,step]*mass)[:,None].contiguous(),right=True).squeeze(1)
                            prev = torch.minimum(pick,walk.argmax(-1))
                        else:
                            prev = _inverse_cdf(p,u[:,step])
                    opts = dict(greedy_rows=not last_mass,last_mass=last_mass)
                    for a,b in zip(sampled_walk(scores,cand,temps,u,**opts),_by_torch(scores,cand,temps,u,**opts)):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
