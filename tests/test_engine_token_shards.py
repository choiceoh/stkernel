"""Ragged TP prefill preserves real rows, final token and recurrent/KV state."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
import torch

from engine.base.comm import LocalTP
from engine.modules.token_shards import TokenShards
from engine.profiles.glm53.execution import prefill_layer_major
from engine.profiles.glm53.net import Step
from tests.test_engine_execution_plans import model
from tests.test_engine_prefill_tiles import OraclePrefill


class TokenShardTests(unittest.TestCase):
    def test_communication_roundtrip_crops_padding_and_keeps_last_token(self):
        torch.set_num_threads(1)
        for rows in (129, 130, 131, 2047, 2048, 2049, 2121, 2672, 31791, 32255):
            with self.subTest(rows=rows):
                def rank(comm):
                    owner = OraclePrefill(comm, True)
                    view = TokenShards(owner, rows, comm.rank)
                    full = torch.arange(rows * 8).reshape(rows, 8).float()
                    shard = view.shard(full)
                    torch.testing.assert_close(view.all_gather(shard), full, rtol=0, atol=0)
                    projected = view.gather_project(shard, lambda x: x * 2 + 3)
                    torch.testing.assert_close(projected, full * 2 + 3, rtol=0, atol=0)
                    reduced = view.reduce_scatter(full + comm.rank)
                    torch.testing.assert_close(view.gather_result(reduced), full * 4 + 6, rtol=0, atol=0)
                    last = comm.all_gather(shard[view.last_local:view.last_local+1], dim=0)[-1:]
                    torch.testing.assert_close(last, full[-1:], rtol=0, atol=0)
                LocalTP(4, timeout_s=30).run(rank)

    def test_packet_projection_callback_survives_padding_and_result_is_cropped(self):
        for rows in (129, 130, 131):
            padded = ((rows + 3) // 4) * 4
            result = torch.arange(padded * 8).reshape(padded, 8)
            owner = NS(comm=NS(world_size=4), project_tiles=True,
                       gather_project=Mock(return_value=result))
            view = TokenShards(owner, rows, 3)
            source, project, packet_project = object(), Mock(), Mock()
            actual = view.gather_project(source, project, packet_project=packet_project)
            owner.gather_project.assert_called_once_with(source, project, packet_project=packet_project)
            torch.testing.assert_close(actual, result[:rows], rtol=0, atol=0)

    def test_model_prefill_and_next_decode_keep_hidden_aux_kda_and_kv(self):
        torch.set_num_threads(1)
        # Three newly admitted remainder cases. Aligned SP/projection behavior
        # is covered by test_engine_prefill_tiles against its existing SP path.
        for rows in (129, 130, 131):
            with self.subTest(rows=rows):
                def rank(comm):
                    net, caches = model(("kda", "dsa", "kda"), comm=comm)
                    slot = caches.slots.take(2)
                    context = 64
                    caches.pool.reserve(2, context + rows + 7)
                    prefix = Step.prefill(torch.arange(context) % net.vp, 0, 2, slot)
                    caches.prepare(prefix)
                    net.forward(prefix, caches)
                    step = Step.prefill(torch.arange(rows) % net.vp, context, 2, slot,
                        patches=((torch.tensor([0, rows-1]), torch.full((2, net.F.hidden), .25).bfloat16()),),
                        marks=((64, 0), (128, 1)))
                    caches.prepare(step)
                    before, paged_before = caches.state.clone(), caches.paged.clone()
                    expected = net.forward(step, caches, aux_layers=[0, 2])
                    after, paged_after = caches.state.clone(), caches.paged.clone()
                    marks_after = {k: v.clone() for k, v in caches._snap.items()}
                    follow = Step.prefill(torch.arange(7) % net.vp, context + rows, 2, slot)
                    caches.prepare(follow)
                    next_hidden = net.forward(follow, caches)
                    final, paged_final = caches.state.clone(), caches.paged.clone()
                    transport = OraclePrefill(comm, True)
                    net.prefill_transport = transport
                    for method in ('ordinary', 'last', 'layer-major'):
                        caches.state.copy_(before); caches.paged.copy_(paged_before)
                        for value in caches._snap.values():
                            value.zero_()
                        caches.prepare(step)
                        if method == 'layer-major':
                            actual = prefill_layer_major(net, step, caches,
                                NS(tile_rows=rows, prefill_tiles=1), [0, 2])
                        else:
                            actual = net.forward(step, caches, aux_layers=[0, 2],
                                                 last_hidden_only=method == 'last')
                        want = (expected[0][-1:], expected[1]) if method == 'last' else expected
                        for a, b in zip(actual, want):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        torch.testing.assert_close(caches.state, after, rtol=0, atol=0)
                        torch.testing.assert_close(caches.paged, paged_after, rtol=0, atol=0)
                        for key in marks_after:
                            torch.testing.assert_close(caches._snap[key][:2], marks_after[key][:2], rtol=0, atol=0)
                        caches.prepare(follow)
                        torch.testing.assert_close(net.forward(follow, caches), next_hidden, rtol=0, atol=0)
                        torch.testing.assert_close(caches.state, final, rtol=0, atol=0)
                        torch.testing.assert_close(caches.paged, paged_final, rtol=0, atol=0)
                    self.assertGreater(transport.calls, 0)
                LocalTP(4, timeout_s=60).run(rank)


if __name__ == '__main__':
    unittest.main()
