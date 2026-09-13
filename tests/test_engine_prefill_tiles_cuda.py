"""Actual CUDA tile readiness/slot reuse; synthetic peers, no NIC claim."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires admitted GB10 GPU")
class PrefillTilesCudaTests(unittest.TestCase):
    def test_bf16_fp8_threshold_tail_and_delayed_slot_reuse(self):
        from engine.kernels.prefill_collectives import PrefillCollectives, BLOCK
        from engine.kernels.dense import FP8Linear
        if torch.cuda.get_device_capability() != (12, 1):
            self.skipTest("requires GB10")
        torch.manual_seed(931302)
        project = FP8Linear((torch.randn(1024, 4096, device="cuda") * .02).bfloat16())
        for local_rows in (256, 576, 1024, 2304):
            peers = [(torch.randn(local_rows, 4096, device="cuda") + r).bfloat16() for r in range(4)]
            cursor = [0]
            modes = []
            def exchange(out, source, *, group=None, async_op=False):
                # Delay the transfer on its real CUDA stream, exposing reads
                # before readiness and writes before the prior slot is released.
                torch.cuda._sleep(100000)
                fp8 = source.dtype == torch.uint8
                rows = source.numel() // 4104 if fp8 else source.shape[0]
                start = cursor[0]
                cursor[0] += rows
                values = [v[start:start+rows].contiguous() for v in peers]
                if fp8:
                    values = [PrefillCollectives.pack(v, v.numel())[0] for v in values]
                out.copy_(torch.cat(values, dim=0))
                modes.append("fp8" if fp8 else "bf16")
                return NS(wait=lambda: None)
            def ordinary(x, dim=0):
                out = torch.empty((x.shape[0]*4, 4096), device=x.device, dtype=x.dtype)
                exchange(out, x)
                return out
            owner = PrefillCollectives(NS(world_size=4, group=None, all_gather=ordinary), project_tiles=True)
            with patch("torch.distributed.all_gather_into_tensor", side_effect=exchange):
                expected_input = owner.all_gather(peers[0])
                expected = project(expected_input)
                torch.cuda.synchronize()
                for repeat in range(3):
                    cursor[0] = 0
                    modes.clear()
                    observed = []
                    def consume(x):
                        observed.append(x.clone())  # must survive two-slot reuse
                        return project(x)
                    actual = owner.gather_project(peers[0], consume)
                    torch.cuda.synchronize()
                    reconstructed = torch.cat([x.view(4, -1, 4096) for x in observed], dim=1).flatten(0, 1)
                    torch.testing.assert_close(reconstructed, expected_input, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(cursor[0], local_rows)
                    self.assertEqual(set(modes), {"fp8" if local_rows >= 1024 else "bf16"})


if __name__ == "__main__":
    unittest.main()
