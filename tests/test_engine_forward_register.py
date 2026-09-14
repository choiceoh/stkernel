"""Changed inputs and TX addresses must survive the register W4 pipeline."""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'requires a reserved GB10 GPU')
class RegisterPipelineGpuTests(unittest.TestCase):
    def test_same_build_bytes_private_scratch_and_rebound_direct_output(self):
        from probes.engine_forward_register import check
        check(lambda *args, **kwargs: None, timing=False)
