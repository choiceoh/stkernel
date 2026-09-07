"""CPU contracts for actual-call routing and separately sized M64 storage."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'overlay/modules/glm53_moe/b12x_moe.py'
DISPATCH=ROOT/'overlay/modules/glm53_moe/moe_dispatch.py'

def functions(path, names, ns):
    tree=ast.parse(path.read_text())
    selected=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in selected:
        n.decorator_list=[]
        # Imports are replaced by the explicit fakes in ns.
        for parent in ast.walk(n):
            for field in ('body','orelse','finalbody'):
                body=getattr(parent,field,None)
                if isinstance(body,list):setattr(parent,field,[s for s in body if not isinstance(s,(ast.Import,ast.ImportFrom))])
    code=ast.Module(body=selected,type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(path),'exec'),ns)
    return ns

class M64Tests(unittest.TestCase):
    def setUp(self):
        self.torch=SimpleNamespace(bfloat16='bf16',device=lambda x:x,
            empty=Mock(return_value='output'),cuda=SimpleNamespace(
                get_device_capability=Mock(return_value=(12,1)),
                is_current_stream_capturing=Mock(return_value=False)))
        self.ns=functions(WRAPPER,{'_glm53_prefill_m64_geometry','_workspace_for_prefill','_allocate_buffers'}, {'torch':self.torch})
        self.exact=dict(enabled=True,num_experts=288,num_local_experts=288,hidden_size=4096,
            intermediate_size=512,top_k=8,quant_mode='nvfp4',activation='swigluoai_uninterleave',
            swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,capability=(12,1),output_dtype='bf16')

    def test_other_geometry_dtype_quant_activation_arch_are_excluded(self):
        gate=self.ns['_glm53_prefill_m64_geometry']
        self.assertTrue(gate(**self.exact))
        for field,value in dict(enabled=False,num_experts=72,num_local_experts=72,hidden_size=2048,
            intermediate_size=1024,top_k=4,quant_mode='mxfp4',activation='silu',swiglu_alpha=1.702,
            swiglu_beta=1.,swiglu_limit=None,capability=(12,0),output_dtype='fp16').items():
            with self.subTest(field=field):self.assertFalse(gate(**dict(self.exact,**{field:value})))

    def test_actual_rows_and_capture_choose_separate_workspace(self):
        stock=object();candidate=object();static=object()
        obj=SimpleNamespace(_dynamic_workspace=stock,_prefill_m64_workspace=candidate)
        choose=self.ns['_workspace_for_prefill']
        for rows in (1,2048,4096,6143,6144,6912,8192,8193,32768):
            with self.subTest(rows=rows):
                self.assertIs(choose(obj,stock,rows),candidate if 6144<=rows<=8192 else stock)
                self.assertIs(choose(obj,static,rows),static)
                self.assertIs(choose(obj,None,rows),None)
        self.torch.cuda.is_current_stream_capturing.return_value=True
        self.assertIs(choose(obj,stock,8192),stock)
        self.torch.cuda.is_current_stream_capturing.side_effect=RuntimeError('CUDA unavailable')
        self.assertIs(choose(obj,stock,8192),stock)
        obj._prefill_m64_workspace=None
        self.assertIs(choose(obj,stock,8192),stock)

    def test_allocation_is_bounded_and_does_not_replace_stock_geometry(self):
        for enabled,capacity,extra in ((False,8192,False),(True,4096,False),(True,6144,True),(True,8192,True),(True,32768,True)):
            with self.subTest(enabled=enabled,capacity=capacity):
                stock=Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
                candidate=Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
                self.ns.update(_GLM53_PREFILL_M64=enabled,
                    allocate_sm120_moe_workspace=stock,allocate_sm120_dynamic_workspace=candidate,
                    select_sm120_moe_backend=lambda **kw:'dynamic',
                    _get_static_compact_cutover_pairs=lambda *a:640,
                    _effective_glm53_static_cutover=lambda *a,**kw:640)
                fields={k:v for k,v in self.exact.items() if k not in ('enabled','capability')}
                obj=SimpleNamespace(**fields,max_num_tokens=capacity,activation_precision='fp4',device='cuda',_prefill_m64_workspace=None)
                self.ns['_allocate_buffers'](obj)
                self.assertEqual(obj._dynamic_workspace.routed_rows,capacity*8)
                self.assertNotIn('tile_m',vars(obj._dynamic_workspace))
                self.assertEqual(candidate.call_count,int(extra))
                if extra:
                    self.assertEqual(obj._prefill_m64_workspace.tile_m,64)
                    self.assertEqual(obj._prefill_m64_workspace.routed_rows,min(capacity,8192)*8)

    def test_override_drives_storage_and_rejects_invalid_tiles_before_allocation(self):
        class Allocated(Exception):pass
        torch=SimpleNamespace(device=object,uint8='u8',empty=Mock(side_effect=Allocated))
        ns=dict(torch=torch,Sm120DynamicMoEWorkspace=object,
            _normalize_activation_precision=lambda x:x,_normalize_quant_mode=lambda *a:'nvfp4',
            _sf_params_for_quant_mode=lambda x:(16,'sf'),_select_dynamic_tile_m=lambda *a:128,
            is_gated_activation=lambda a:a!='relu2',_level_tile_n=lambda a:128,
            _LEVEL_TILE_M=128,_LEVEL_TILE_N=128,_DYNAMIC_SLICE_CHUNK=1,_align_up=lambda n,m:((n+m-1)//m)*m,
            _check_memref_limit=lambda *a:None)
        functions(DISPATCH,{'allocate_sm120_dynamic_workspace','_dynamic_task_geometry'},ns)
        allocate=ns['allocate_sm120_dynamic_workspace']
        args=dict(state_E=288,weight_E=288,routed_rows=8192*8,k=4096,n=512,num_topk=8,device='cuda')
        for tile in (None,64):
            with self.assertRaises(Allocated):allocate(**args,tile_m=tile)
            expected_tile=128 if tile is None else tile
            padded=(65536//expected_tile+287)*expected_tile
            self.assertEqual(torch.empty.call_args.args,(1,padded,2048))
        for tile in (0,-64,63,65,256,True,64.0):
            with self.subTest(tile=tile),self.assertRaises(ValueError):allocate(**args,tile_m=tile)
        with self.assertRaises(ValueError):allocate(**args,tile_m=64,activation='relu2')

if __name__=='__main__':unittest.main()
