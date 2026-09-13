"""Packet bytes, BF16 rounding, transport padding and the real MoE handoff."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
torch=None
if importlib.util.find_spec('torch'):
    import torch


class SumHandoffTests(unittest.TestCase):
    def test_real_moe_hands_both_outputs_to_fusion_without_an_intermediate_add(self):
        source=ROOT/'engine/profiles/glm53/net.py'
        node=copy.deepcopy(next(n for n in ast.walk(ast.parse(source.read_text()))
                                if isinstance(n,ast.FunctionDef) and n.name=='_moe'))
        node.decorator_list=[];node.returns=None
        for a in node.args.args: a.annotation=None
        calls=[]
        class Output:
            def __add__(self,other): calls.append(('add',self,other));return 'joined'
        out,shared=Output(),Output()
        net=NS(F=NS(spec_k=6,swiglu_limit=10.),p={},shared_overlap=None,
               route=lambda l,x:(None,None),_experts={3:lambda *a:out},
               linear=lambda x,name:shared if name.endswith('sh_down') else NS(chunk=lambda *a,**k:(None,None)),
               _activation=lambda *a:None,comm=NS(all_reduce=lambda x:('ordinary',x)))
        scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        fused=lambda a,b: calls.append(('pair',a,b)) or 'fused'
        self.assertEqual(scope['_moe'](net,3,NS(shape=(2675,4096)),reduce_pair=fused),'fused')
        self.assertEqual(calls,[('pair',out,shared)])
        calls.clear()
        self.assertEqual(scope['_moe'](net,3,NS(shape=(2675,4096))),('ordinary','joined'))
        self.assertEqual(calls,[('add',out,shared)])

    def test_shards_forward_real_inputs_and_only_describe_transport_padding(self):
        from engine.modules.token_shards import TokenShards
        for rows in (129,130,131,2672,2675,32255):
            calls=[]
            owner=NS(comm=NS(world_size=4),project_tiles=False,fuse_sum=True,
                     reduce_scatter_pair=lambda a,b,**kw:calls.append((a,b,kw)) or 'packets')
            view=TokenShards(owner,rows,0)
            a,b=NS(shape=(rows,4096)),NS(shape=(rows,4096))
            self.assertEqual(view.reduce_scatter_pair(a,b),'packets')
            self.assertEqual(calls,[(a,b,dict(padded_rows=((rows+3)//4)*4))])
            with self.assertRaises(ValueError): view.reduce_scatter_pair(a,NS(shape=(rows+1,4096)))


@unittest.skipUnless(torch is not None,'requires CPU torch')
class SumPacketTests(unittest.TestCase):
    def test_actual_fused_body_matches_materialized_bf16_sum_packet_bytes(self):
        class Pointer:
            def __init__(self,data,offset=0): self.data,self.offset=data.reshape(-1),offset
            def __add__(self,offset): return Pointer(self.data,self.offset+offset)
        program=[0]
        def load(p,mask,other=0):
            indices=p.offset
            selected=indices[mask]
            self.assertTrue(bool(((selected>=0)&(selected<p.data.numel())).all()))
            out=torch.full(indices.shape,other,dtype=p.data.dtype)
            out[mask]=p.data[selected]
            return out
        def store(p,value,mask=None):
            indices=torch.as_tensor(p.offset)
            if mask is None: mask=torch.ones_like(indices,dtype=torch.bool)
            indices,mask=torch.broadcast_tensors(indices,mask)
            self.assertTrue(bool(((indices[mask]>=0)&(indices[mask]<p.data.numel())).all()))
            value=torch.as_tensor(value).to(p.data.dtype).expand(indices.shape).contiguous()
            if p.data.dtype==torch.float8_e4m3fn:
                p.data.view(torch.uint8)[indices[mask]]=value.view(torch.uint8)[mask]
            else: p.data[indices[mask]]=value[mask]
        tl=NS(program_id=lambda d:program[d],arange=torch.arange,load=load,store=store,
              float32=torch.float32,bfloat16=torch.bfloat16,float8e4nv=torch.float8_e4m3fn,
              abs=torch.abs,max=lambda x,axis:torch.max(x),exp2=torch.exp2,ceil=torch.ceil,log2=torch.log2,
              maximum=lambda a,b:torch.maximum(a,torch.as_tensor(b)))
        source=ROOT/'engine/kernels/prefill_collectives/sum_pack.py'
        node=copy.deepcopy(next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)))
        node.decorator_list=[]
        for arg in node.args.args: arg.annotation=None
        scope=dict(tl=tl);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        run=scope[node.name]
        torch.manual_seed(773)
        for rows,hidden,block in ((65,128,64),(67,128,64),(64,4096,2048)):
            x=torch.randn(rows,hidden).bfloat16();y=torch.randn_like(x)
            # Rounding before quantization changes this FP8 tie. Force scale=1.
            x.reshape(-1)[:4]=torch.tensor([448.,1.,-0.,0.]).bfloat16()
            y.reshape(-1)[:4]=torch.tensor([0.,.06640625,-0.,0.]).bfloat16()
            padded=((rows+3)//4)*4;local=padded*hidden//4
            stride=((local+4*(local//block)+127)//128)*128
            actual=torch.full((stride*4,),0xcd,dtype=torch.uint8)
            for pid in range(padded*hidden//block):
                program[0]=pid
                run(Pointer(x),Pointer(y),Pointer(actual.view(torch.float8_e4m3fn)),
                    Pointer(actual.view(torch.float32)),x.numel(),local,stride,block)
            summed=torch.zeros(padded,hidden,dtype=torch.bfloat16);summed[:rows]=x+y
            expected=torch.zeros_like(actual)
            for rank in range(4):
                values=summed.reshape(4,local)[rank].float().reshape(-1,block)
                scales=torch.exp2(torch.ceil(torch.log2(values.abs().amax(1).clamp_min(1e-30)/448.)))
                quantized=(values/scales[:,None]).to(torch.float8_e4m3fn)
                expected[rank*stride:rank*stride+local]=quantized.reshape(-1).view(torch.uint8)
                expected.view(torch.float32)[rank*(stride//4)+local//4:rank*(stride//4)+local//4+len(scales)]=scales
            self.assertTrue(torch.equal(actual,expected))
            self.assertNotEqual((x.float()+y.float())[0,1].to(torch.float8_e4m3fn).view(torch.uint8).item(),
                                (x+y)[0,1].to(torch.float8_e4m3fn).view(torch.uint8).item())


if __name__=='__main__': unittest.main()
