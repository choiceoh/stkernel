"""Direct recurrent loads preserve the logical tensor and state contracts."""
import importlib.util
import unittest

from tests.image_kernels import PRESENT, REASON

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and PRESENT, "requires CUDA; " + REASON)
class KdaStridesTests(unittest.TestCase):
    def setUp(self):
        from engine.kernels.kda import fused_recurrent_kda
        from engine.profiles.glm53.lanes import served
        self.kernel = fused_recurrent_kda
        self.run = served(reference_for=("expert",)).kda_recurrent
        torch.manual_seed(129522)

    def exact(self, actual, expected):
        for x,y in zip(actual,expected):
            self.assertEqual(x.shape,y.shape)
            self.assertEqual(x.dtype,y.dtype)
            self.assertTrue(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)))
            self.assertTrue(x.is_contiguous())

    def args(self,t):
        x=torch.randn(t,6144,device="cuda",dtype=torch.bfloat16)
        q,k,v=(p.reshape(1,t,16,128) for p in x.split(2048,dim=-1))
        beta=torch.randn(t,6416,device="cuda",dtype=torch.bfloat16)[:,6144:6160][None]
        g=torch.randn(1,t,16,128,device="cuda",dtype=torch.bfloat16)
        a=torch.randn(16,device="cuda")*.2
        bias=torch.randn(2048,device="cuda")*.1
        state=torch.randn(1,16,128,128,device="cuda")*.1
        return q,k,v,g,beta,a,bias,state,-5.

    def test_conv_and_beta_views_match_materialized_values(self):
        for t in (1,2,3,4,5,6,7,12):
            for magnitude in (.001,1.,10.):
                with self.subTest(tokens=t,magnitude=magnitude):
                    args=list(self.args(t))
                    for x in args[:5]:x.mul_(magnitude)
                    for initialized in (True,False):
                        if not initialized:args[7]=None
                        saved=[x.clone() if isinstance(x,torch.Tensor) else x for x in args]
                        dense=[x.contiguous() if isinstance(x,torch.Tensor) else x for x in args]
                        self.exact(self.run(*args),self.run(*dense))
                        for x,y in zip(args,saved):
                            if isinstance(x,torch.Tensor):
                                self.assertTrue(torch.equal(x,y))

    def test_channel_head_token_strides_and_vector_beta(self):
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            t,h,hv,k,v=6,2,4,33,17
            def transposed(heads,dim):
                return torch.randn(1,t*2,dim*2,heads,device="cuda",dtype=dtype)[:,::2,::2].transpose(-1,-2)
            q,kk=transposed(h,k),transposed(h,k)
            vv,g=transposed(hv,v),-transposed(hv,k).abs()
            state=torch.randn(1,hv,k,v,device="cuda")*.1
            for vector in (False,True):
                beta=(transposed(hv,v) if vector else torch.randn(1,t*2,hv*2,device="cuda",dtype=dtype)[:,::2,::2]).sigmoid()
                if vector:beta=beta.transpose(-1,-2).contiguous().transpose(-1,-2)
                inputs=(q,kk,vv,g,beta)
                kwargs=dict(initial_state=state,inplace_final_state=False,state_layout="kv")
                actual=self.kernel(*inputs,**kwargs)
                expected=self.kernel(*(x.contiguous() for x in inputs),**kwargs)
                self.exact(actual,expected)
        args=list(self.args(6))
        args[0]=args[0][:,:1,:1,:1].expand(1,6,16,128)
        args[2]=args[2][:,:1].expand(1,6,16,128)
        self.exact(self.run(*args),self.run(*(x.contiguous() if isinstance(x,torch.Tensor) else x for x in args)))

    def test_graph_replay_mutable_inputs_and_accepted_prefix(self):
        for t in (1,6):
            args=list(self.args(t))
            self.run(*args)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):actual=self.run(*args)
            try:
                for i in range(8):
                    for x in args[:5]:x.normal_()
                    original=args[7].clone()
                    dense=[x.contiguous() if isinstance(x,torch.Tensor) else x for x in args]
                    expected=self.run(*dense)
                    graph.replay();self.exact(actual,expected)
                    self.assertTrue(torch.equal(args[7],original))
                    args[7].copy_(actual[1][i%t:i%t+1])
            finally:graph.reset()

    def test_separate_destination_and_reject_alias_or_malformed_inputs(self):
        q,k,v,g,beta,a,bias,state,lb=self.args(6)
        kwargs=dict(initial_state=state,inplace_final_state=False,state_layout="kv",
                    sigmoid_beta=True,a_log=a,g_bias=bias,compute_gate=True,lower_bound=lb)
        out=torch.empty(v.shape,device=v.device,dtype=v.dtype)
        result=self.kernel(q,k,v,g,beta,out=out,**kwargs)
        self.assertEqual(result[0].data_ptr(),out.data_ptr())
        self.exact(result,self.run(q,k,v,g,beta,a,bias,state,lb))
        # A contiguous gate aliases the output shape; it must be rejected
        # before direct input loads race stores from other value tiles.
        for invalid in (g,out.cpu(),out.float(),out[:,:2],out.transpose(-1,-2)):
            with self.assertRaises(ValueError):self.kernel(q,k,v,g,beta,out=invalid,**kwargs)
        for args in ((q[:,:,:2],k,v,g,beta),(q,k,v,g[:,:,:2],beta),
                     (q,k,v,g,beta[:,:,:2]),(q.cpu(),k,v,g,beta),
                     (q,k,v,g,beta.to(torch.int32)),(q,k,v[:,:,:,None,:],g,beta)):
            with self.assertRaises(ValueError):self.kernel(*args,**kwargs)

    def test_legacy_scalar_decay_kernel_against_independent_reference(self):
        from engine.kernels.kda.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
        from engine.modules.linear_attention import gated_delta_rule
        for t in (1,6):
            q,k,v=[torch.randn(1,t,3,d,device="cuda",dtype=torch.bfloat16) for d in (64,64,32)]
            g=-torch.rand(1,t,3,device="cuda")
            beta=torch.rand(1,t,3,device="cuda")
            state=torch.randn(1,3,32,64,device="cuda")*.1
            original=state.clone()
            actual,states=fused_recurrent_gated_delta_rule_fwd(
                q,k,v,g,beta,64**-.5,state,inplace_final_state=False,use_qk_l2norm_in_kernel=True)
            expected,final=gated_delta_rule(q,k,v,g,beta,state.transpose(-1,-2))
            for x,y,limit in ((actual,expected,.008),(states[-1:].transpose(-1,-2),final,2e-6)):
                error=(x.float()-y.float()).abs().max()/y.float().abs().max().clamp_min(1e-6)
                self.assertLess(error.item(),limit)
            self.assertTrue(torch.equal(state,original))


if __name__=="__main__":unittest.main()
