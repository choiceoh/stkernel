"""Exact ordinary-reduction boundary and changing-address graph replay."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires admitted GB10 GPU")
class DirectMhcCudaTests(unittest.TestCase):
    def test_rank_fold_rounding_and_dynamic_descriptor_replay(self):
        from engine.kernels.dense.mhc import MHC
        if torch.cuda.get_device_capability() != (12, 1):
            self.skipTest("requires GB10 SM121")
        torch.manual_seed(931301)
        for packed in (False, True):
            fn = torch.randn(24, 16384, device="cuda") * .006
            if packed:
                fn = fn.bfloat16().float()
            mhc = MHC({"fn": fn})
            scale = torch.tensor([.2, .3, .4], device="cuda")
            base = torch.randn(24, device="cuda") * .1
            norm = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
            for rows in (1, 7, 28, 64):
                banks = [[torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                          for _ in range(4)] for _ in range(2)]
                # A local-first fold disagrees across ranks; omitting the BF16
                # boundary also changes the second half of this canary.
                canary = ((2.**24, 1.), (-2.**24, 2.**-8), (1., 2.**-9), (1., 0.))
                for bank in banks:
                    for rank, x in enumerate(bank):
                        x[:, 0] = canary[rank][0]
                        x[:, 1] = canary[rank][1]
                descriptors = [torch.tensor([x.data_ptr() for x in b], device="cuda", dtype=torch.int64)
                               for b in banks]
                descriptor = descriptors[0].clone()
                x = torch.empty_like(banks[0][0])  # deliberately not the summed input
                residual = torch.randn(rows, 4, 4096, device="cuda", dtype=torch.bfloat16)
                post = torch.rand(rows, 4, 1, device="cuda")
                comb = torch.rand(rows, 4, 4, device="cuda")
                def call(value, packets=None):
                    return mhc("fn", value, residual, post, comb, scale, base, norm,
                               1e-5, 1e-6, 2., 20, packets=packets)
                call(x, descriptor)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        actual = call(x, descriptor)
                    for repeat in range(12):
                        bank = banks[repeat % 2]
                        for value in bank:
                            value[:, 2:].normal_().mul_(.1 + repeat / 4)
                        descriptor.copy_(descriptors[repeat % 2])
                        total = bank[0].float() + bank[1].float()
                        total = (total + bank[2].float()) + bank[3].float()
                        expected = call(total.bfloat16())
                        graph.replay()
                        torch.cuda.synchronize()
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
                finally:
                    graph.reset()


if __name__ == "__main__":
    unittest.main()
