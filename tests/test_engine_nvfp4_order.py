"""Dynamic NVFP4 scales must obey their producing kernel's dependency."""
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tests.image_kernels import PRESENT, REASON


@unittest.skipUnless(torch.cuda.is_available() and PRESENT, 'CUDA required; ' + REASON)
class NVFP4OrderingTests(unittest.TestCase):
    def test_dynamic_scale_is_ready_before_quantization(self):
        import flashinfer
        from torch.utils.cpp_extension import load
        from engine.kernels.dense import DenseLinear

        producer = load(name='st_nvfp4_scale_producer',
                        sources=[str(Path(__file__).with_name('nvfp4_scale_producer.cu'))],
                        extra_cuda_cflags=['-O2', '-gencode', 'arch=compute_121a,code=sm_121a'])
        quantize = flashinfer.nvfp4_quantize
        weight = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16) * .02
        layer = DenseLinear(weight)
        x = torch.randn(1024, 512, device='cuda', dtype=torch.bfloat16)
        expected = layer(x).clone()
        torch.cuda.synchronize()

        def delayed_scale(a, scale, **kwargs):
            value = float(scale.item())
            scale.fill_(1e-12)
            producer.publish(scale, value, 5_000_000)
            return quantize(a, scale, **kwargs)

        with patch.object(flashinfer, 'nvfp4_quantize', delayed_scale):
            for _ in range(8):
                actual = layer(x)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
