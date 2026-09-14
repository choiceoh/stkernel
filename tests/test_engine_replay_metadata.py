"""Real replay fills and CPU sampling with capture replaced by a synchronous runner."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from engine.base.comm import Comm
from engine.base.sampler import sample
from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs, SamplingGraphs
from engine.profiles.glm53.net import Segment, Step


class Copies(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default:
            dst, src = args[:2]
            self.calls.append((dst, src, args[2] if len(args) > 2 else False))
        return func(*args, **(kwargs or {}))


class CpuGraphs:
    def __init__(self, forward, make_inputs, shapes, **kwargs):
        self.forward, self.label = forward, kwargs.get('label')
        self.inputs = {shape: make_inputs(*shape) for shape in shapes}
        self.outputs = {}

    def run(self, shape, fill):
        inputs = self.inputs[shape]
        with Copies() as copies:
            fill(inputs)
        self.copies = copies.calls
        return inputs if self.label == 'target' else self.forward(inputs)

    def close(self):
        pass


class ReplayMetadataTests(unittest.TestCase):
    def setUp(self):
        # Only replace page locking and graph execution. Tensor allocations,
        # aliasing, copy dispatch, host fills and sampling are real CPU torch.
        empty = torch.empty
        def cpu_empty(*args, **kwargs):
            kwargs.pop('pin_memory', None)
            return empty(*args, **kwargs)
        self.addCleanup(patch.stopall)
        patch('torch.empty', cpu_empty).start()
        patch('engine.profiles.glm53.decode_graphs.DecodeGraphs', CpuGraphs).start()
        torch.set_num_threads(1)

    def target(self):
        facts = NS(spec_k=7, block=64, max_position=8192, kpool=4)
        net = NS(F=facts, comm=Comm(), vp=32, lanes=NS(graph_resources=None),
                 head_buffer=lambda n, device: torch.empty(n, 32, device=device))
        caches = NS(F=facts, layout=None, device=torch.device('cpu'),
                    slots=NS(owner=[-1]*5), block_table=torch.empty(4, 128),
                    prepare=Mock(), reset=Mock())
        return Glm53DecodeGraphs(net, caches, 4, 8)

    def assert_transfer(self, copies, count):
        self.assertEqual(len(copies), count)
        for dst, src, non_blocking in copies:
            self.assertTrue(non_blocking)
            self.assertTrue(dst.is_contiguous())
            self.assertTrue(src.is_contiguous())
            self.assertEqual(dst.shape, src.shape)
            self.assertEqual(dst.dtype, src.dtype)

    def test_target_width_churn_preserves_contexts_owners_and_static_addresses(self):
        target = self.target()
        pointers = {s: held.data_ptr() for s, held in target.metadata.items()}
        for turn, n in enumerate((4, 1, 3, 2, 1, 4)):
            capacity = (4096, 8192)[turn % 2]
            shape = n, 8, capacity
            ids = torch.arange(n*8) + turn
            segs = tuple(Segment(2**33+i+turn, n-i, 17*turn+i, i*8, 8) for i in range(n))
            step = Step(ids, segs)
            before = {s: held.clone() for s, held in target.metadata.items() if s != shape}
            result, seqs, slots, caches, _ = target.run(step, shape)
            torch.testing.assert_close(result.ids, ids, rtol=0, atol=0)
            self.assertEqual(result.contexts.tolist(), [s.ctx for s in segs])
            self.assertEqual([int(s.ctx) for s in result.segments], [s.ctx for s in segs])
            self.assertEqual(seqs.tolist(), [s.seq for s in segs])
            self.assertEqual(slots.tolist(), [s.slot for s in segs])
            self.assertIs(caches.sequence_ids, seqs)
            self.assertIs(caches.slots, slots)
            self.assert_transfer(target.graphs.copies, 2)  # ids + one metadata block
            for s, expected in before.items():
                torch.testing.assert_close(target.metadata[s], expected, rtol=0, atol=0)
            self.assertEqual({s: held.data_ptr() for s, held in target.metadata.items()}, pointers)

    def test_device_pipeline_keeps_independent_inputs_and_does_not_touch_host_staging(self):
        target = self.target()
        for n in (1, 3, 4, 2):
            shape = n, 8, 4096
            target.staging[n].fill_(-97)
            inputs = (torch.arange(n*8), torch.arange(n)+400, torch.arange(n+2)+9, torch.arange(n+2)+1)
            result, seqs, slots, _, _ = target.run_inputs(shape, *inputs)
            for actual, expected in zip((result.ids, result.contexts, seqs, slots), inputs):
                torch.testing.assert_close(actual, expected[:actual.numel()], rtol=0, atol=0)
            self.assertTrue(bool((target.staging[n] == -97).all()))

    def test_packed_sampling_matches_independent_fields_across_widths_and_modes(self):
        from engine.base import draws
        shared = {(n, 8): torch.zeros(n*8, 64) for n in range(1, 5)}
        outputs = {(n, t, cap): (None, None, logits) for (n, t), logits in shared.items() for cap in (4096, 8192)}
        target = NS(tokens=8, graphs=NS(outputs=outputs), net=NS(comm=Comm(), rank=0, vp=64))
        graphs = SamplingGraphs(target, 61, .8)
        self.assertTrue(all(len(inputs) == 1 for inputs in graphs.greedy.inputs.values()))
        pointers = {s: held.data_ptr() for s, held in graphs.device_policy.items()}
        for turn, n in enumerate((4, 1, 3, 2, 1, 4)):
            rows, shape = n*8, (n, 8, (4096, 8192)[turn % 2])
            logits = shared[n, 8]
            logits.copy_(torch.randn(rows, 64, generator=torch.Generator().manual_seed(37+turn)))
            logits[:, 61:] = 1000.  # undecodable tail stays excluded
            temps = [0., .7, 1., 1.2] * (rows//4)
            # A numeric float conversion would corrupt the larger int32 values.
            ks = [0, 1, 17, 2**24+1] * (rows//4)
            ps = [1., .6, .8, .9] * (rows//4)
            uniforms = draws.uniforms(draws.row_key(19, turn+1, 7), draws.PICK, rows)
            actual = graphs.run(shape, temps, ks, ps, uniforms)
            fields = (torch.tensor(temps), torch.tensor(ks, dtype=torch.int32), torch.tensor(ps), torch.tensor(uniforms))
            expected = sample(logits, fields[0], fields[2], fields[3], top_k=fields[1], valid=61)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for got, want in zip(graphs.stochastic.inputs[n, 8][1:], fields):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            self.assert_transfer(graphs.stochastic.copies, 1)
            held = graphs.device_policy[n, 8].clone()
            greedy = graphs.run(shape, [0.]*rows)
            torch.testing.assert_close(greedy, logits[:, :61].argmax(-1), rtol=0, atol=0)
            self.assertEqual(graphs.greedy.copies, [])
            torch.testing.assert_close(graphs.device_policy[n, 8], held, rtol=0, atol=0)
            graphs.run(shape, [1.]*rows, uniforms=uniforms)
            _, _, k, p, _ = graphs.stochastic.inputs[n, 8]
            self.assertEqual(k.tolist(), [0]*rows)
            torch.testing.assert_close(p, torch.full((rows,), .8), rtol=0, atol=0)
        self.assertEqual({s: held.data_ptr() for s, held in graphs.device_policy.items()}, pointers)
        with self.assertRaisesRegex(ValueError, 'captured shape'):
            graphs.run((1, 8), [1.]*16, uniforms=[.5]*16)


if __name__ == '__main__':
    unittest.main()
