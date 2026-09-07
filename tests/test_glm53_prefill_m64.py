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
                    m64_stock_contract_matches=lambda:True,static_v2_weights_layout=lambda **kw:True,
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

class PortTests(unittest.TestCase):
    def test_port_changes_m_dimensions_after_stock_initialization(self):
        source=ROOT/'overlay/modules/glm53_moe/moe_dynamic_gated_tiled.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MoEGatedDynamicKernelM64Tiled')
        # Execute the host-side constructor/layout override, leaving the GPU
        # DSL bodies to the real image compiler and numerical probes.
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef)
                  and n.name in ('__init__','_setup_attributes')]
        class Base:
            def __init__(self,**kw):
                self.stock_kwargs=kw
                self.tile_shape_mnk=(128,128,128)
                self.fc1_tile_shape_mnk=(128,64,128)
                self.epi_tile=(128,128)
                self.num_mma_warps=8
                self.fc1_sfb_tile_shape_nk=(128,128)
            def _setup_attributes(self,hidden_size):
                self.setup_hidden=hidden_size
                for name in ('a_dtype','a_layout','b_dtype','b_layout','c_layout','epi_stage','sf_vec_size','tiled_mma'):
                    setattr(self,name,name)
                self.b_smem_layout_staged='stock_b'
                self.sfb_smem_layout_staged='stock_sfb'
                self._dense_cls=SimpleNamespace(_make_smem_layouts=Mock(
                    side_effect=[('physical_a5','discard','discard','discard','discard'),
                                 ('discard','discard','physical_sfa4','discard','discard')]))
        ns=dict(MoEGatedDynamicKernelTiled=Base,m64_stock_contract_matches=lambda:True,
                cutlass=SimpleNamespace(BFloat16='bf16'))
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
        args=dict(sf_vec_size=16,mma_tiler_mn=(64,128),hidden_size=4096,intermediate_size=512,
                  num_topk=8,activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.)
        obj=ns['MoEGatedDynamicKernelM64Tiled'](**args)
        self.assertEqual(obj.stock_kwargs['mma_tiler_mn'],(128,128))
        self.assertEqual((obj.tile_shape_mnk,obj.fc1_tile_shape_mnk,obj.epi_tile),((64,128,128),(64,64,128),(64,128)))
        self.assertEqual((obj.num_mma_warps,obj.fc1_sfb_tile_shape_nk),(8,(128,128)))
        self.assertEqual((obj.sa_tile_shape_mk,obj.sfa_tile_shape_mk),((128,128),(128,128)))
        self.assertEqual((obj.sa_tiles_per_block,obj.sfa_tiles_per_block),(2,2))
        obj._setup_attributes(4096)
        self.assertEqual(obj.setup_hidden,4096)
        self.assertEqual((obj.a_smem_layout_staged,obj.sfa_smem_layout_staged),('physical_a5','physical_sfa4'))
        self.assertEqual((obj.b_smem_layout_staged,obj.sfb_smem_layout_staged),('stock_b','stock_sfb'))
        for call,stages in zip(obj._dense_cls._make_smem_layouts.call_args_list,(5,4)):
            self.assertEqual(call.args[:2],((128,128,128),(64,128)))
            self.assertEqual(call.args[6],stages)
            self.assertEqual(call.args[-1],'tiled_mma')
        with self.assertRaises(ValueError):obj._setup_attributes(8192)
        for field,value in dict(hidden_size=8192,intermediate_size=1024,num_topk=4,sf_vec_size=32,
                                mma_tiler_mn=(32,128),swiglu_limit=None).items():
            with self.subTest(field=field),self.assertRaises(ValueError):ns['MoEGatedDynamicKernelM64Tiled'](**dict(args,**{field:value}))
        ns['m64_stock_contract_matches']=lambda:False
        with self.assertRaises(ValueError):ns['MoEGatedDynamicKernelM64Tiled'](**args)

    def test_source_hash_is_checked_and_missing_or_changed_source_is_rejected(self):
        import hashlib,tempfile
        source=ROOT/'overlay/modules/glm53_moe/moe_dynamic_gated_tiled.py'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stock=root/'_moe_dynamic/gated.py';stock.parent.mkdir()
            stock.write_bytes(b'pinned body')
            ns=functions(source,{'m64_stock_contract_matches'},dict(Path=Path,hashlib=hashlib,
                __file__=str(root/'moe_dynamic_gated_tiled.py'),_M64_GATED_SHA256=hashlib.sha256(stock.read_bytes()).hexdigest()))
            self.assertTrue(ns['m64_stock_contract_matches']())
            stock.write_bytes(b'new body');self.assertFalse(ns['m64_stock_contract_matches']())
            stock.unlink();self.assertFalse(ns['m64_stock_contract_matches']())

if __name__=='__main__':unittest.main()
