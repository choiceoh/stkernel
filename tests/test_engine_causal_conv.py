"""Single-sequence conv: original GPU arithmetic and independent history oracle."""
import importlib.util
from types import SimpleNamespace
import unittest

torch = None
if importlib.util.find_spec("torch"):
    import torch


def legacy_conv(x, w, state):
    """Frozen batching adapter from b7a6e44c; the generic kernel is unchanged."""
    from engine.kernels.causal_conv import causal_conv1d_fn
    t, c = x.shape
    table = torch.zeros(2, c, w.shape[1]-1, device=x.device, dtype=x.dtype)
    if state is not None:
        table[1] = state
    programs = -(-t // 8)
    batch = torch.zeros(programs, device=x.device, dtype=torch.int32)
    offsets = torch.arange(programs, device=x.device, dtype=torch.int32)
    metadata = SimpleNamespace(batch_ptr=batch, token_chunk_offset_ptr=offsets,
        nums_dict={8: dict(tot=programs, mlist=None, mlist_len=programs,
                          offsetlist=None, batch_ptr=batch, token_chunk_offset_ptr=offsets)})
    y = causal_conv1d_fn(x.T, w, None, table,
        torch.arange(2, device=x.device, dtype=torch.int32)*t,
        cache_indices=torch.ones(1, device=x.device, dtype=torch.int32),
        has_initial_state=torch.full((1,), state is not None, device=x.device, dtype=torch.bool),
        activation="silu", metadata=metadata)
    return y.T, table[1]


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class SingleConvTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.causal_conv_single import causal_conv1d_single
        self.run = causal_conv1d_single
        torch.manual_seed(12861)

    def exact(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        self.assertTrue(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))

    def check(self, x, w, state):
        saved = [v.clone() for v in (x, w, state) if v is not None]
        y, final = self.run(x, w, state)
        expected, old_final = legacy_conv(x, w, state)
        self.exact(y, expected)
        self.exact(final, old_final)
        history = (torch.zeros(w.shape[0], w.shape[1]-1, device=x.device, dtype=x.dtype)
                   if state is None else state.to(x.dtype))
        oracle = torch.cat((history, x.T), dim=1)[:, -(w.shape[1]-1):]
        self.exact(final, oracle)
        for v, original in zip((v for v in (x, w, state) if v is not None), saved):
            self.exact(v, original)
        self.assertTrue(y.is_contiguous() and final.is_contiguous())
        return y, final

    def test_projection_stride_lengths_and_history_boundaries(self):
        for t in (1,2,3,6,7,8,9,24,64,256,1024,4096,6912):
            x = torch.randn(t,6416,device="cuda",dtype=torch.bfloat16)[:, :6144]
            w = torch.randn(6144,4,device="cuda")
            for initialized in (False, True):
                with self.subTest(tokens=t, initialized=initialized):
                    state = torch.randn(6144,3,device="cuda",dtype=x.dtype) if initialized else None
                    self.check(x,w,state)

    def test_widths_channel_tails_dtypes_and_strided_weights_history(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for k in (2,3,4):
                for t in (1,3,9):
                    with self.subTest(dtype=dtype, width=k, tokens=t):
                        x = torch.randn(t,134,device="cuda",dtype=dtype)[:, ::2]
                        w = torch.randn(134,2*k,device="cuda")[::2, ::2]
                        state = torch.randn(134,2*(k-1),device="cuda")[::2, ::2]
                        self.check(x,w,state)
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(6,67,device="cuda",dtype=dtype)
            self.check(x,torch.randn(67,4,device="cuda",dtype=dtype),None)

    def test_chunk_carry_and_torch_convolution(self):
        from engine.modules.causal_conv import causal_conv1d
        x = torch.randn(29,129,device="cuda",dtype=torch.bfloat16)*.1
        w = torch.randn(129,4,device="cuda")
        expected, expected_state = self.run(x,w,None)
        outputs, state = [], None
        start = 0
        for length in (1,2,6,3,9,8):
            y,state = self.run(x[start:start+length],w,state)
            outputs.append(y); start += length
        self.exact(torch.cat(outputs),expected)
        self.exact(state,expected_state)
        ref, ref_state = causal_conv1d(x,w)
        torch.testing.assert_close(expected.float(),ref.float(),atol=2e-3,rtol=.008)
        self.exact(expected_state,ref_state.to(x.dtype))

    def test_bf16_patterns_and_saturated_silu(self):
        bits = torch.arange(65536,device="cuda",dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        x = torch.where(torch.isfinite(values),values,0.).reshape(16,4096)
        for scale in (.001, .125, 1., 100.):
            w = torch.tensor([1.,-.5,.25,-.125],device="cuda").expand(4096,4).contiguous()*scale
            self.check(x,w,None)
        w = torch.ones(4096,4,device="cuda")
        for value in (-100.,-88.,-20.,-0.,0.,20.,88.,100.):
            self.check(torch.full_like(x,value),w,None)

    def test_graph_replay_changed_inputs_state_and_independent_streams(self):
        from engine.profiles.glm53.lanes import served
        run = served(reference_for=("expert",)).conv_prefill
        for t in (1,6,9):
            x = torch.randn(t,6416,device="cuda",dtype=torch.bfloat16)[:, :6144]
            w = torch.randn(6144,4,device="cuda")
            s = torch.randn(6144,3,device="cuda",dtype=x.dtype)
            run(x,w,s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): out,final = run(x,w,s)
            try:
                for _ in range(4):
                    for v in (x,w,s): v.normal_()
                    expected,history = self.check(x,w,s)
                    graph.replay()
                    self.exact(out,expected); self.exact(final,history)
            finally: graph.reset()
        streams = [torch.cuda.Stream(),torch.cuda.Stream()]
        inputs = [(torch.randn(t,129,device="cuda",dtype=torch.bfloat16),
                   torch.randn(129,4,device="cuda")) for t in (1,17)]
        for x,w in inputs: run(x,w,None)
        for stream in streams: stream.wait_stream(torch.cuda.current_stream())
        outputs=[]
        for stream,(x,w) in zip(streams,inputs):
            with torch.cuda.stream(stream): outputs.append(run(x,w,None))
        torch.cuda.synchronize()
        for (x,w),(y,f) in zip(inputs,outputs):
            expected,state=legacy_conv(x,w,None)
            self.exact(y,expected);self.exact(f,state)

    def test_empty_and_invalid_arguments(self):
        x = torch.empty(0,67,device="cuda",dtype=torch.bfloat16)
        w = torch.randn(67,4,device="cuda")
        s = torch.randn(67,3,device="cuda")
        for state in (None,s):
            y,final=self.run(x,w,state)
            self.assertEqual(y.shape,x.shape)
            self.exact(final,torch.zeros_like(s,dtype=x.dtype) if state is None else s.to(x.dtype))
            if state is not None:self.assertNotEqual(final.data_ptr(),state.data_ptr())
        for args in ((x.cpu(),w.cpu(),None),(x,w.cpu(),None),(x,w[:3],None),
                     (x,w[:, :1],None),(x,w,s.cpu()),(x,w,s[:, :2]),
                     (x.to(torch.int32),w,None),(x,w.to(torch.int32),None)):
            with self.assertRaises(ValueError):self.run(*args)


if __name__ == "__main__": unittest.main()
