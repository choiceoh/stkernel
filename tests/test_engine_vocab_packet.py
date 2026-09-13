"""Exact candidate packets and reusable merge storage; GPU or Triton CPU interpreter."""
import importlib.util
import os
import unittest

torch = None
if importlib.util.find_spec('torch'):
    import torch

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
DEVICE = 'cpu' if INTERPRET else 'cuda'


@unittest.skipUnless(torch is not None and (INTERPRET or torch.cuda.is_available()),
                     'requires CUDA or Triton interpreter')
class CandidatePacketTests(unittest.TestCase):
    def test_fused_packets_preserve_exact_keys_order_and_masks(self):
        from engine.kernels.common.vocab_candidates import select_logits
        gen = torch.Generator(device=DEVICE).manual_seed(9146)
        for rows, width in ((3, 17), (2, 2049), (6, 38720), (24, 38720)):
            for dtype in ((torch.float32, torch.float16) if INTERPRET else
                          (torch.float32, torch.float16, torch.bfloat16)):
                x = torch.randn(rows, width*2, generator=gen, device=DEVICE).to(dtype)[:, ::2]
                x[:, :6] = torch.tensor([0., -0., float('nan'), -float('nan'), float('inf'), -float('inf')],
                                        dtype=dtype, device=DEVICE)
                x[:, 8:16] = 4.
                before = x.contiguous().view(torch.uint8).clone()
                for valid, start in ((width, 0), (max(1, width-11), 3*38720)):
                    value = x[:, :valid].float().contiguous()
                    bits = value.view(torch.int32).long()
                    ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
                    ordered = torch.where(torch.isnan(value), 0x7fffffff, ordered)
                    keys = (ordered << 32) | (0xffffffff - (start+torch.arange(valid, device=DEVICE)))
                    k = min(16, valid)
                    with self.subTest(rows=rows, width=width, dtype=dtype, valid=valid):
                        torch.testing.assert_close(select_logits(x, start, valid, k), keys.topk(k).values,
                                                   rtol=0, atol=0)
                self.assertTrue(torch.equal(x.contiguous().view(torch.uint8), before))

    def test_reused_dense_rows_clear_old_candidates_before_overlapping_new_ones(self):
        from engine.kernels.common.vocab_candidates import pack
        from engine.modules.vocab import CandidateBuffer
        workspace = CandidateBuffer(4, 257, 8, DEVICE)
        generator = torch.Generator(device=DEVICE).manual_seed(406)
        for trial, rows in enumerate((4, 1, 3, 4, 2, 4)):
            values = torch.randn(rows, 257, generator=generator, device=DEVICE)
            values[:, :4] = torch.tensor([0., -0., float('inf'), float('nan')], device=DEVICE)
            encoded = pack(values, 0, 257)
            columns = torch.tensor([0, 1, 2, 3, 127, 128, 255, 256], device=DEVICE).roll(trial)
            packet = encoded.index_select(1, columns)
            if trial % 2:
                packet[:, -3:] = -(2**63)
            expected = torch.full_like(values, float('-inf'))
            # Independent unpack/scatter reference, including NaN canonicalization.
            for row in range(rows):
                for key in packet[row].tolist():
                    if key == -(2**63):
                        continue
                    index, ordered = 0xffffffff-(key & 0xffffffff), key >> 32
                    bits = ordered ^ 0x7fffffff if ordered < 0 else ordered
                    expected[row, index] = torch.tensor(bits, dtype=torch.int32).view(torch.float32).item()
            with self.subTest(trial=trial, rows=rows):
                actual = workspace.restore(packet, 257)
                self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires actual CUDA graph')
    def test_changing_packets_replay_with_one_owner_across_batch_shapes(self):
        from engine.modules.vocab import CandidateBuffer, topk
        from engine.kernels.common.vocab_candidates import pack, select
        from types import SimpleNamespace as NS
        width, k = 38720, 16
        workspace = CandidateBuffer(24, width*4, k*4, 'cuda')
        graphs, owners = {}, {}
        try:
            for rows in (24, 6, 12):
                local = torch.zeros(rows, width, dtype=torch.bfloat16, device='cuda')
                packet = torch.empty(rows, 4*k, dtype=torch.int64, device='cuda')
                comm = NS(world_size=4, all_gather=lambda value, dim=-1, packet=packet: packet)
                full = torch.randn(rows, width*4, device='cuda').bfloat16()
                packet.copy_(torch.cat([select(pack(full[:, r*width:(r+1)*width], r*width, width), k)
                                        for r in range(4)], dim=-1))
                topk(local, comm, 0, k, workspace=workspace)  # compile before capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    got = topk(local, comm, 0, k, workspace=workspace)
                graphs[rows], owners[rows] = graph, (local, packet, comm, got)
            for trial, rows in enumerate((24, 6, 12, 24, 6, 24)):
                local, packet, comm, got = owners[rows]
                full = torch.randn(rows, width*4, device='cuda').bfloat16()
                full[:, trial:trial+24] = 5  # cutoff ties and overlapping old/new candidates
                packet.copy_(torch.cat([select(pack(full[:, r*width:(r+1)*width], r*width, width), k)
                                        for r in range(4)], dim=-1))
                expected = topk(local, comm, 0, k)
                graphs[rows].replay()
                for actual, reference in zip(got, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        finally:
            for graph in graphs.values():
                graph.reset()


if __name__ == '__main__':
    unittest.main()
