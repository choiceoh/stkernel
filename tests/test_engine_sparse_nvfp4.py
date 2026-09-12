"""Sparse NVFP4 pattern, scale and stream contracts for the diagnostic probe."""
import os
import unittest
import torch
from probes.engine_sparse_nvfp4 import Library, Projection, synthetic, qualify, validate_sparse


class SparsePatternTests(unittest.TestCase):
    def test_all_six_pair_masks_and_signed_zero(self):
        masks = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        for pair in masks:
            packed = torch.full((1,4), 0x88, dtype=torch.uint8)
            packed[0,pair] = 0xF1
            validate_sparse(packed)

    def test_scalar_two_of_four_is_insufficient(self):
        # Low nibbles in every pair are nonzero: scalar 2:4, not pair 4:8.
        with self.assertRaises(ValueError):
            validate_sparse(torch.tensor([[0x01,0x01,0x01,0x01]], dtype=torch.uint8))
        with self.assertRaises(ValueError):
            validate_sparse(torch.tensor([[0x11,0x11,0x11,0x00]], dtype=torch.uint8))

    def test_pair_pruning_and_quantizer_midpoint_ties(self):
        from probes.engine_sparse_nvfp4_prune import prune_pairs, quantize32
        weight = torch.tensor([[[1.,1., 1.,1., 2.,2., 3.,3.]]])
        self.assertTrue(torch.equal(prune_pairs(weight),
                        torch.tensor([[[0.,0., 0.,0., 2.,2., 3.,3.]]])))
        x = torch.zeros(1,1,32)
        x[0,0,:7] = torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.])
        x[0,0,7:14] = -x[0,0,:7]
        x[0,0,-1] = 6.
        packed, scale = quantize32(x)
        codes = torch.stack((packed&15,packed>>4),-1).flatten(-2)
        self.assertEqual(scale.item(),0x38)  # E4M3 1.0
        self.assertEqual(codes[0,0,:14].tolist(),[0,2,2,4,4,6,6,8,10,10,12,12,14,14])


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('ST_SPARSE_PROBE_LIBRARY'),
                     'requires CUDA and ST_SPARSE_PROBE_LIBRARY built from the probe')
class SparseKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12,1):
            raise unittest.SkipTest('requires SM121a')
        cls.library = Library(os.environ['ST_SPARSE_PROBE_LIBRARY'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')

    def make(self, n=3, zero_scales=False, zero_weights=False):
        w,ws = synthetic(2,128,256,sparse=True,seed=15)
        x,xs = synthetic(2,n,256,sparse=False,seed=19)
        if zero_scales:
            ws[...,::2] = 0
            xs[...,1::2] = 0
        if zero_weights:
            w.fill_(0)
        return Projection(self.library,w,x,ws,xs)

    def test_six_masks_varied_scales_batches_and_tail(self):
        for n in (1,3,129):
            p = self.make(n=n)
            try:
                qualify(p)
            finally:
                p.close()

    def test_signed_zero_pairs(self):
        w,ws = synthetic(2,128,256,sparse=True,seed=31)
        x,xs = synthetic(2,3,256,sparse=False,seed=33)
        w[w == 0] = 0x88
        p = Projection(self.library,w,x,ws,xs)
        try:
            qualify(p)
        finally:
            p.close()

    def test_zero_scales_and_zero_weights(self):
        for zero_scales,zero_weights in ((True,False),(False,True)):
            p = self.make(zero_scales=zero_scales,zero_weights=zero_weights)
            try:
                qualify(p)
                for variant in (0,1):
                    self.assertEqual(torch.count_nonzero(p.run(variant)).item(),0)
            finally:
                p.close()

    def test_nondefault_stream_graph_replay(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            p = self.make()
            graphs = []
            try:
                for variant in (0,1):
                    expected = p.run(variant).clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        p.run(variant)
                    graphs.append(graph)
                    p.outputs[variant].fill_(float('nan'))
                    graph.replay()
                    self.assertTrue(torch.equal(p.outputs[variant],expected))
            finally:
                stream.synchronize()
                for graph in graphs:
                    graph.reset()
                p.close()


if __name__ == '__main__':
    unittest.main()
