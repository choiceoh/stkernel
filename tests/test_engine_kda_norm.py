"""KDA output fusion preserves the BF16 gate and normalization contract."""
import importlib.util
import unittest
from dataclasses import replace
from types import SimpleNamespace

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class KdaOutputNormTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kda.output import kda_output_norm
        from engine.modules.linear_attention import kda_output_norm as reference
        self.run,self.ref=kda_output_norm,reference
        torch.manual_seed(62819)

    def exact(self, actual, expected):
        self.assertEqual(actual.shape,expected.shape)
        self.assertEqual(actual.dtype,expected.dtype)
        self.assertTrue(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))

    def test_glm128_rows_magnitudes_and_final_rounding(self):
        from engine.profiles.glm53.net import O_NORM_EPS
        for rows in (0,1,16,96,384,1024,4096,16384,65536,110592):
            for magnitude in (1e-20,1e-3,1.,1000.,1e12):
                with self.subTest(rows=rows,magnitude=magnitude):
                    x=(torch.randn(rows,128,device="cuda")*magnitude).bfloat16()
                    gate=torch.randn_like(x)*5
                    weight=torch.randn(128,device="cuda",dtype=torch.bfloat16)
                    self.exact(self.run(x,gate,weight,O_NORM_EPS),self.ref(x,gate,weight,O_NORM_EPS))

    def test_finite_bf16_patterns_and_saturated_gates(self):
        bits=torch.arange(65536,device="cuda",dtype=torch.int32).to(torch.int16)
        values=bits.view(torch.bfloat16)
        values=torch.where(torch.isfinite(values),values,0.)
        patterns=values.reshape(-1,128).contiguous()
        for gate_value in (-100.,-88.,-20.,-1.,-0.,0.,1.,20.,88.,100.):
            gate=torch.full_like(patterns,gate_value)
            weight=torch.linspace(-2,2,128,device="cuda").bfloat16()
            self.exact(self.run(patterns,gate,weight),self.ref(patterns,gate,weight))
        for value in (-0.,0.,1.,-1.):
            x=torch.full((512,128),value,device="cuda",dtype=torch.bfloat16)
            self.exact(self.run(x,patterns,weight),self.ref(x,patterns,weight))

    def test_other_dimensions_and_fp32_weight(self):
        for d in (1,8,17,33,64,127,128,129,511,512):
            x=torch.randn(37,d,device="cuda",dtype=torch.bfloat16)
            gate=torch.randn_like(x)
            weight=torch.randn(d,device="cuda")
            for eps in (1e-6,1e-5,1e-4):
                with self.subTest(dim=d,eps=eps):
                    actual,expected=self.run(x,gate,weight,eps),self.ref(x,gate,weight,eps)
                    error=(actual.float()-expected.float()).abs().max()/expected.float().abs().max()
                    self.assertLess(error.item(),.008)
                    if d==128:self.exact(actual,expected)

    def test_graph_reads_changed_values_and_does_not_mutate_inputs(self):
        from engine.profiles.glm53.lanes import served
        run=served(reference_for=("expert",)).kda_output_norm
        x=torch.randn(6,16,128,device="cuda",dtype=torch.bfloat16)
        gate=torch.randn_like(x)
        weight=torch.randn(128,device="cuda",dtype=torch.bfloat16)
        run(x,gate,weight)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=run(x,gate,weight)
        try:
            for _ in range(8):
                for tensor in (x,gate,weight):tensor.normal_()
                saved=[t.clone() for t in (x,gate,weight)]
                graph.replay()
                self.exact(out,self.ref(x,gate,weight))
                for actual,expected in zip((x,gate,weight),saved):self.exact(actual,expected)
                self.assertNotIn(out.data_ptr(),[t.data_ptr() for t in (x,gate,weight)])
        finally:graph.reset()

    def test_independent_streams_and_invalid_layout(self):
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        inputs=[(torch.randn(rows,128,device="cuda",dtype=torch.bfloat16),
                 torch.randn(rows,128,device="cuda",dtype=torch.bfloat16),
                 torch.randn(128,device="cuda",dtype=torch.bfloat16)) for rows in (96,4096)]
        expected=[self.ref(*args) for args in inputs]
        for args in inputs:self.run(*args)
        for stream in streams:stream.wait_stream(torch.cuda.current_stream())
        outputs=[]
        for stream,args in zip(streams,inputs):
            with torch.cuda.stream(stream):outputs.append(self.run(*args))
        torch.cuda.synchronize()
        for actual,ref in zip(outputs,expected):self.exact(actual,ref)
        x,g,w=inputs[0]
        for args in ((x.float(),g,w),(x,g.float(),w),(x[::2],g[::2],w),
                     (x,g[:1],w),(x,g,w[:-1]),(x,g,w.cpu()),(x,g,w[::2]),
                     (x.cpu(),g.cpu(),w.cpu())):
            with self.assertRaises(ValueError):self.run(*args)
        for eps in (0.,-1.,float("nan"),float("inf")):
            with self.assertRaises(ValueError):self.run(x,g,w,eps)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "requires PyTorch and the vendored norm module's Triton imports")
class KdaOutputNormContractTests(unittest.TestCase):
    def test_composition_matches_default_glm_gated_norm_at_small_core_magnitudes(self):
        import torch
        from engine.kernels.kda.kda import FusedRMSNormGated
        from engine.profiles.glm53 import lanes
        from engine.profiles.glm53.net import Glm53Net, Step
        from test_engine_glm53 import tiny_facts

        F = tiny_facts()
        comm = SimpleNamespace(rank=0, world_size=4, all_reduce=lambda x: x)
        net = Glm53Net(F, comm, lanes.reference(), layers=[0])
        net.p = {s.name: torch.zeros(s.shape, dtype=s.dtype) for s in net.specs()
                 if s.name.startswith("L0.kda.")}
        width = F.kda_heads_local * F.kda_dim
        net.p["L0.kda.o_proj"] = torch.eye(F.hidden, width, dtype=torch.bfloat16)
        net.p["L0.kda.o_norm"] = torch.linspace(.1, .8, F.kda_dim).bfloat16()
        # This is the class GLM constructs without specifying an epsilon.
        oracle = FusedRMSNormGated(F.kda_dim, activation="sigmoid", dtype=torch.bfloat16)
        oracle.weight.data.copy_(net.p["L0.kda.o_norm"])
        for tokens in (1, 6, 7):  # decode, speculative verify, chunk prefill
            for magnitude in (1e-5, 1e-3, 1e-1):
                with self.subTest(tokens=tokens, magnitude=magnitude):
                    core = (torch.linspace(-magnitude, magnitude, F.kda_dim)
                            .repeat(1, tokens, F.kda_heads_local, 1).bfloat16())
                    state = torch.zeros(F.kda_heads_local, F.kda_dim, F.kda_dim)
                    net.lanes = replace(lanes.reference(),
                        kda_chunk=lambda *args: (core, state[None]),
                        kda_recurrent=lambda *args: (core, state.repeat(tokens, 1, 1, 1)))
                    conv = torch.zeros(3*width, net.conv_ring, dtype=torch.bfloat16)
                    rec = torch.zeros(net.rec_ring, *state.shape)
                    cache = SimpleNamespace(kda=lambda *args: (conv, rec))
                    x = torch.zeros(tokens, F.hidden, dtype=torch.bfloat16)
                    step = Step.prefill(torch.zeros(tokens, dtype=torch.int64), 0, 0, 1)
                    got = net._kda(0, x, step, cache)
                    normed = oracle.forward_native(core, torch.zeros_like(core))
                    expected = torch.nn.functional.linear(normed.reshape(tokens, width), net.p["L0.kda.o_proj"])
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)


if __name__=="__main__":unittest.main()
