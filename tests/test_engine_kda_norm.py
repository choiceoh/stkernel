"""KDA output fusion preserves the BF16 gate and normalization contract."""
import importlib.util
import unittest

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
        for rows in (0,1,16,96,384,1024,4096,16384,65536,110592):
            for magnitude in (1e-20,1e-3,1.,1000.,1e12):
                with self.subTest(rows=rows,magnitude=magnitude):
                    x=(torch.randn(rows,128,device="cuda")*magnitude).bfloat16()
                    gate=torch.randn_like(x)*5
                    weight=torch.randn(128,device="cuda",dtype=torch.bfloat16)
                    self.exact(self.run(x,gate,weight),self.ref(x,gate,weight))

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
            for eps in (1e-6,1e-4):
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


if __name__=="__main__":unittest.main()
