"""EP tiled canary CPU contracts; no Torch/CUDA imports or GPU work."""
import ast
import copy
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'overlay/modules/glm53_moe/glm53_ep_tiled_selftest.py'
PACKAGE = '_ep_tiled_canary_cpu'
package = ModuleType(PACKAGE)
package.__path__ = [str(SOURCE.parent)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE+'.glm53_ep_tiled_selftest', SOURCE)
canary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = canary
spec.loader.exec_module(canary)


def runtime():
    return dict(torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
        md=SimpleNamespace(_WORKSPACE_CACHE={'old': object()}, _WEIGHT_CACHE={}),
        device='cuda:0', provenance=dict(source='fixed'))


def packed_owner():
    class Tensor:
        dtype='torch.uint8'
        device='cuda:0'
        def __init__(self,shape,address):self.shape,self.address,self.digest=shape,address,'original'
        def is_contiguous(self):return True
        def untyped_storage(self):return self
        def data_ptr(self):return self.address
    first,second=Tensor((72,512,1552),100),Tensor((72,256,1552),200)
    scales=SimpleNamespace(enabled=True,fc1=first,fc2=second)
    views=SimpleNamespace(tiled=True,packed_only=True,reform_scales=scales,
                          sfb1_packed=first,sfb2_packed=second)
    raw1,raw2=Tensor((75497472,),10),Tensor((37748736,),20)
    owner=SimpleNamespace(_ep_tiled_weight_views=views,_ep_tiled_workspace=object(),
        w1_scale=raw1,w2_scale=raw2,w1_sf_mma=raw1,w2_sf_mma=raw2)
    layer=SimpleNamespace(w13_weight=SimpleNamespace(device='cuda:0'),
                          w13_weight_scale=raw1,w2_weight_scale=raw2)
    return owner,layer


def tensor_identity(value):
    return dict(shape=list(value.shape),dtype=value.dtype,data_ptr=value.address,sha256=value.digest)


def native_cache_fixture(decode_opt=False):
    """Execute the real compiler's pure key construction, without CuTe imports."""
    path = SOURCE.with_name('moe_static_ep_tiled.py')
    tree = ast.parse(path.read_text())
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    factory = functions['ep_tiled_compile_spec']
    start = next(i for i,n in enumerate(factory.body) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0]) == 'key')
    end = next(i for i in range(start,len(factory.body)) if isinstance(factory.body[i],ast.Return))
    key_fn = ast.parse('''def native_key(m, sf6=True, route_mode="local",
                                    expert_map_len=None, expert_map_dtype=None,
                                    local_expert_offset=0, decode_opt=None):
    max_rows, mac, topk_ids_dtype = 256, 48, "torch.int32"
    input_scales_are_reciprocal, fast_math, reform_sf_pack = False, True, sf6
    geometry = ep_tiled_geometry(m, max_rows, mac)
    scale_mode = ep_tiled_scale_mode(reform_sf_pack)
    decode_opt = ep_tiled_decode_opt(m, reform_sf_pack, decode_opt)
    route_key = ep_tiled_route_key(route_mode=route_mode, expert_map_len=expert_map_len,
        expert_map_dtype=expert_map_dtype, local_expert_offset=local_expert_offset)
''').body[0]
    selected = next(n for n in factory.body if isinstance(n,ast.Assign)
                    and ast.unparse(n.targets[0])=='scatter_bf16')
    key_statements = factory.body[start:end]
    # Only key construction executes here. The separate map fake operand
    # belongs to real CuTe compilation, covered by the normal CPU probe.
    key_statements = [n for n in key_statements if not (
        isinstance(n, ast.If) and ast.unparse(n.test) == "route_mode == 'global'")]
    result = factory.body[end].value
    assert isinstance(result, ast.Tuple) and len(result.elts) == 3
    key_fn.body += [copy.deepcopy(selected)] + copy.deepcopy(key_statements) + [
        ast.Return(copy.deepcopy(result.elts[-1]))]
    constants = [copy.deepcopy(n) for n in tree.body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name)
                 and n.targets[0].id in ('EP_TILED_CACHE_TAG','EP_TILED_A_RING_CACHE_TAG',
                                        'EP_TILED_SF6_WORD_CACHE_TAG','EP_TILED_BF16_SCATTER_CACHE_TAG',
                                        'EP_TILED_ROUTE_CACHE_TAG','EP_TILED_DECODE_OPT_CACHE_TAG')]
    module = ast.Module(body=constants + [copy.deepcopy(functions[name]) for name in
        ('ep_tiled_geometry','ep_tiled_scale_mode','ep_tiled_route_metadata',
         'ep_tiled_route_key','ep_tiled_decode_opt','ep_tiled_decode_opt_enabled')] + [key_fn], type_ignores=[])
    ns = {'_EP_TILED_DECODE_OPT': decode_opt}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    def native_key(*args, **kwargs):
        with patch.dict(sys.modules, {'torch':SimpleNamespace(int32='torch.int32',int64='torch.int64')}):
            return ns['native_key'](*args, **kwargs)
    return SimpleNamespace(ep_tiled_geometry=ns['ep_tiled_geometry'],
                           ep_tiled_decode_opt_enabled=ns['ep_tiled_decode_opt_enabled'],
                           native_key=native_key,_EP_TILED_KERNEL_CACHE={})


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        canary._STATES.clear()
        self.owner, self.layer, self.rt = object(), object(), runtime()

    def capture(self, context, owner, layer, receipt):
        return dict(references=['CPU B1/B2'], buffers=['shared input'], original={})

    def test_two_hooks_publish_once_after_validation_sync_and_release(self):
        output = io.StringIO()
        events = []
        self.rt['torch'].cuda.synchronize.side_effect = lambda _: events.append('sync')
        with redirect_stdout(output), patch.object(canary,'_runtime',return_value=self.rt), patch.object(
                canary,'_memory',return_value={}), patch.object(
                canary,'_capture_references',side_effect=self.capture) as capture, patch.object(
                canary,'_validate_candidate',side_effect=lambda *a: events.append('validate')) as validate:
            handle = canary.before_relayout(self.owner,self.layer)
            self.assertEqual(output.getvalue(),'')
            self.assertEqual(handle['receipt']['verdict'],'RUNNING')
            receipt = canary.after_relayout(self.owner,self.layer,handle)
            cached = canary.before_relayout(object(),object())
            self.assertIs(canary.after_relayout(object(),object(),cached),receipt)
        self.assertEqual((capture.call_count,validate.call_count),(1,1))
        self.assertEqual(events,['validate','sync'])
        self.assertNotIn('references',handle)
        self.assertNotIn('buffers',handle)
        lines=output.getvalue().splitlines()
        self.assertEqual(len(lines),1)
        self.assertEqual(json.loads(lines[0].removeprefix('[ep-tiled-selftest] PASS ')),receipt)
        self.assertFalse(receipt['performance_acceptance'])
        self.assertFalse(receipt['full_sanitizer_acceptance'])

    def test_first_candidate_failure_retained_when_cleanup_also_fails(self):
        original=AssertionError('original peak numerical rejection')
        with redirect_stdout(io.StringIO()) as output, patch.object(canary,'_runtime',return_value=self.rt), patch.object(
                canary,'_memory',return_value={}), patch.object(canary,'_capture_references',side_effect=self.capture), patch.object(
                canary,'_validate_candidate',side_effect=original) as validate:
            handle=canary.before_relayout(self.owner,self.layer)
            self.rt['torch'].cuda.synchronize.side_effect=RuntimeError('cleanup sync failure')
            with self.assertRaisesRegex(RuntimeError,'readiness refused') as caught:
                canary.after_relayout(self.owner,self.layer,handle)
            with self.assertRaisesRegex(RuntimeError,'previously failed'):
                canary.before_relayout(self.owner,self.layer)
        self.assertIs(caught.exception.__cause__,original)
        self.assertEqual(validate.call_count,1)
        self.assertIn('original peak',handle['receipt']['error'])
        self.assertIn('cleanup sync',handle['receipt']['cleanup_error'])
        self.assertEqual(handle['receipt']['verdict'],'FAIL')
        self.assertNotIn('] PASS ',output.getvalue())

    def test_reference_failure_blocks_relayout_and_retry(self):
        with redirect_stdout(io.StringIO()), patch.object(canary,'_runtime',return_value=self.rt), patch.object(
                canary,'_memory',return_value={}), patch.object(
                canary,'_capture_references',side_effect=AssertionError('stock repeat unstable')) as capture:
            with self.assertRaisesRegex(RuntimeError,'stock reference failed'):
                canary.before_relayout(self.owner,self.layer)
            with self.assertRaisesRegex(RuntimeError,'previously failed'):
                canary.before_relayout(self.owner,self.layer)
        self.assertEqual(capture.call_count,1)
        self.assertEqual(next(iter(canary._STATES.values()))['verdict'],'FAIL')

    def test_pending_handle_cannot_be_reused_for_another_owner(self):
        with patch.object(canary,'_runtime',return_value=self.rt), patch.object(canary,'_memory',return_value={}), patch.object(
                canary,'_capture_references',side_effect=self.capture), patch.object(canary,'_validate_candidate') as validate:
            handle=canary.before_relayout(self.owner,self.layer)
            with self.assertRaisesRegex(RuntimeError,'pending owner'):
                canary.after_relayout(object(),self.layer,handle)
            with self.assertRaisesRegex(RuntimeError,'remains in progress'):
                canary.before_relayout(self.owner,self.layer)
        validate.assert_not_called()


class AdmissionTests(unittest.TestCase):
    def test_packed_planes_are_complete_separate_and_raw_release_not_claimed(self):
        owner,layer=packed_owner()
        with patch.object(canary,'_tensor_identity',side_effect=tensor_identity):
            receipt=canary._packed_identity(owner,layer)
        self.assertTrue(receipt['raw_sources_retained'])
        self.assertFalse(receipt['raw_release_acceptance'])
        coverage=receipt['preparation_contract']
        self.assertEqual((coverage['fc1_stages'],coverage['fc2_stages']),(36864,18432))
        self.assertEqual((coverage['raw_bytes'],coverage['packed_bytes']),(113246208,85819392))
        self.assertIn('not an additional canary roundtrip',coverage['scope'])
        mutations=(lambda o,l:setattr(o._ep_tiled_weight_views,'packed_only',False),
                   lambda o,l:setattr(o._ep_tiled_weight_views.reform_scales,'enabled',False),
                   lambda o,l:setattr(o._ep_tiled_weight_views,'sfb1_packed',object()),
                   lambda o,l:setattr(o._ep_tiled_weight_views.sfb1_packed,'shape',(72,256,1552)),
                   lambda o,l:setattr(o._ep_tiled_weight_views.sfb1_packed,'address',10),
                   lambda o,l:setattr(o,'w1_sf_mma',None))
        for mutation in mutations:
            owner,layer=packed_owner();mutation(owner,layer)
            with self.subTest(mutation=mutation),patch.object(canary,'_tensor_identity',side_effect=tensor_identity):
                with self.assertRaises(AssertionError):canary._packed_identity(owner,layer)
        for name in ('w1_scale_storage','w2_scale_storage','_w13_sf_storage','_down_sf_storage','sfb_w13_ptr','sfb_down_ptr'):
            owner,layer=packed_owner();setattr(owner._ep_tiled_weight_views,name,object())
            with self.subTest(raw_alias=name),patch.object(canary,'_tensor_identity',side_effect=tensor_identity):
                with self.assertRaisesRegex(AssertionError,'retains raw'):canary._packed_identity(owner,layer)

    def test_candidate_completion_rejects_mutated_packed_bytes(self):
        owner,layer=packed_owner()
        class Mapping:
            def __setitem__(self,key,value):pass
        context=dict(torch=SimpleNamespace(full=lambda *a,**k:Mapping(),arange=lambda *a,**k:[],int32='i32'),
                     device='cuda:0',tiled=object())
        handle=dict(original={},offset=0,rng=None,receipt={'cases':[]})
        calls=[]
        def observe(value):
            result=tensor_identity(value)
            calls.append(value)
            if len(calls)==3:result['sha256']='mutated packed bytes'
            return result
        with patch.object(canary,'_identities',return_value={}),patch.object(
                canary,'_tensor_identity',side_effect=observe),patch.object(canary,'_check_rng'):
            with self.assertRaisesRegex(AssertionError,'changed its actual packed SF6'):
                canary._validate_candidate(context,owner,layer,handle)
        self.assertNotIn('actual_packed_owner',handle['receipt'])

    def test_source_bound_roundtrip_contract_checks_both_whole_planes(self):
        spec=importlib.util.spec_from_file_location(PACKAGE+'.moe_reform_sf_pack',SOURCE.with_name('moe_reform_sf_pack.py'))
        packing=importlib.util.module_from_spec(spec);sys.modules[spec.name]=packing;spec.loader.exec_module(packing)
        self.assertEqual((packing.REFORM_SF_BLOCK,packing.REFORM_SF_STAGE),(2048,1552))
        raw1,raw2,packed1,packed2=object(),object(),object(),object()
        with patch.object(packing,'pack_plane',side_effect=[(packed1,None),(packed2,None)]) as pack:
            owner=packing.prepare_reform_scales(raw1,raw2,experts=72,n=2048,k=4096)
        self.assertTrue(owner.enabled)
        self.assertEqual([(c.kwargs['rows'],c.kwargs['k'],c.kwargs['kind']) for c in pack.call_args_list],
                         [(4096,4096,'fc1'),(4096,2048,'fc2')])
        with patch.object(packing,'pack_plane',side_effect=[(packed1,None),(None,'unrepresentable')]):
            rejected=packing.prepare_reform_scales(raw1,raw2,experts=72,n=2048,k=4096)
        self.assertFalse(rejected.enabled);self.assertIsNone(rejected.fc1)
        tree=ast.parse(SOURCE.with_name('moe_reform_sf_pack.py').read_text())
        body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pack_plane')
        text=ast.unparse(body)
        self.assertIn('for first in range(0, count, REFORM_SF_CHUNK):',text)
        self.assertIn('back = unpack_sf_inline(target[first:last], REFORM_SF_BLOCK).view_as(raw)',text)
        self.assertIn('valid &= (back == raw).all()',text)
        self.assertIn('if not bool(valid.item()):',text)

    def test_runtime_accepts_new_owner_without_old_micro_workspaces_or_old_knob(self):
        owner=SimpleNamespace(_use_ep=True,_ep_no_dummy=True,global_num_experts=288,
            num_local_experts=72,hidden_dim=4096,intermediate_size_per_partition=2048,
            topk=8,_kernel_num_experts=72,_activation_str='swigluoai_uninterleave',
            _swiglu_alpha=1.,_swiglu_beta=0.,_swiglu_limit=10.)
        torch=SimpleNamespace(__file__=str(SOURCE),__version__='test',cuda=SimpleNamespace(
            get_device_capability=lambda d:(12,1),is_current_stream_capturing=lambda:False))
        md=SimpleNamespace(__file__=str(SOURCE),_GLM53_EP_TILED=True,_GLM53_EP_PREFILL_LOCAL=False)
        local=SimpleNamespace(__file__=str(SOURCE),_stock=SimpleNamespace(__file__=str(SOURCE)),stock_contract_matches=lambda:True)
        decode=SimpleNamespace(__file__=str(SOURCE),ep_tiled_source_contract=Mock())
        prefix='flashinfer.fused_moe.cute_dsl.blackwell_sm12x.'
        modules={prefix+name:SimpleNamespace(__file__=str(SOURCE)) for name in
                 ('glm53_ep_tiled','moe_static_kernel_v4','moe_static_kernel_v5',
                  'moe_reform_sf_pack','moe_sf_pack','moe_dynamic_gated_sf6')}
        modules.update({prefix+'moe_static_ep_tiled':decode,'torch':torch,
            'flashinfer':SimpleNamespace(__file__=str(SOURCE),__version__='test'),
            'cuda.bindings':SimpleNamespace(__file__=None)})
        with tempfile.TemporaryDirectory() as temp:
            relative=Path('cuda_bindings-13.3.1.dist-info/METADATA')
            metadata=Path(temp)/relative;metadata.parent.mkdir();metadata.write_text('Version: 13.3.1\n')
            distribution=SimpleNamespace(files=[relative],version='13.3.1',locate_file=lambda p:Path(temp)/p)
            with patch.dict(sys.modules,{'torch':torch,PACKAGE+'.moe_dispatch':md,PACKAGE+'.moe_dynamic_ep_local':local}), patch.object(
                    canary.importlib,'import_module',side_effect=lambda name:modules[name]), patch.object(
                    canary.importlib.metadata,'distribution',return_value=distribution), patch.object(canary.inspect,'getfile',return_value=str(SOURCE)):
                context=canary._runtime(owner,SimpleNamespace(w13_weight=SimpleNamespace(device=SimpleNamespace(type='cuda'))))
                self.assertEqual(context['provenance']['versions']['cuda.bindings']['version'],'13.3.1')
                self.assertIsNone(context['provenance']['versions']['cuda.bindings']['path'])
                self.assertEqual(len(context['provenance']['source']),13)
                md._GLM53_EP_TILED=False
                with self.assertRaisesRegex(RuntimeError,'selection/source'):
                    canary._runtime(owner,SimpleNamespace(w13_weight=SimpleNamespace(device=SimpleNamespace(type='cuda'))))
        decode.ep_tiled_source_contract.assert_called_once()

    def test_routes_all_ranks_remote_changed_bounds_duplicates_and_full_cases(self):
        original=(("mixed6",6,"mixed"),("balanced12",12,"balanced"),
                  ("concentrated24",24,"concentrated"),("zeros32",32,"zeros"),
                  ("remote33",33,"remote"),("balanced2128",2128,"balanced"),
                  ("balanced4096",4096,"balanced"),("concentrated6912",6912,"concentrated"),
                  ("balanced8192",8192,"balanced"))
        self.assertEqual(canary.CASES[:9],original)  # Original RNG indices stay fixed.
        self.assertEqual(canary.CASES[9:],(("mixed4",4,"mixed"),("balanced8",8,"balanced"),
                                        ("concentrated16",16,"concentrated")))
        self.assertEqual(canary.SEED,905329)
        for offset in (0,72,144,216):
            for changed in (False,True):
                remote=canary.route_rows(33,'remote',offset,changed)
                self.assertTrue(all(len(r)==8 and all(0<=e<288 and not offset<=e<offset+72 for e in r) for r in remote))
                concentrated=canary.route_rows(6912,'concentrated',offset,changed)
                self.assertTrue(all(offset<=e<offset+72 for r in concentrated for e in r))
                mixed=canary.route_rows(6,'mixed',offset,changed)[0]
                self.assertEqual(mixed[-1],mixed[-2]);self.assertIn(-1,mixed);self.assertIn(288,mixed)
                for _,rows,kind in canary.CASES[9:]:
                    routes=canary.route_rows(rows,kind,offset,changed)
                    self.assertEqual(len(routes),rows)
                    self.assertTrue(all(len(row)==8 for row in routes))
                    self.assertNotEqual(routes,canary.route_rows(rows,kind,offset,not changed))
                    if kind=='mixed':
                        self.assertEqual(routes[0][-1],routes[0][-2])
                        self.assertIn(-1,routes[0]);self.assertIn(288,routes[0])
                    elif kind=='concentrated':
                        self.assertTrue(all(offset<=expert<offset+72 for row in routes for expert in row))
                    else:
                        self.assertTrue(all(0<=expert<288 for row in routes for expert in row))
        with self.assertRaises(ValueError):canary.route_rows(6,'unknown',0)

    def test_only_weight_byte_permutation_allowed_without_storage_replacement(self):
        before={name:dict(shape=[4],dtype='uint8',data_ptr=i,sha256='old')
                for i,name in enumerate(('w13','w2','sf1','fc1_alpha'))}
        after={name:dict(value) for name,value in before.items()}
        after['w13']['sha256']='permuted';after['w2']['sha256']='permuted'
        canary._same_weights_and_scales(before,after,relayout=True)
        for name,field,value in [('w13','data_ptr',999),('sf1','sha256','bad'),('fc1_alpha','shape',[8])]:
            changed={n:dict(v) for n,v in after.items()};changed[name][field]=value
            with self.assertRaises(AssertionError):canary._same_weights_and_scales(before,changed,relayout=True)
        with self.assertRaises(AssertionError):canary._same_weights_and_scales(before,after,relayout=False)

    def test_cache_namespace_and_native_shape_must_match(self):
        decode=native_cache_fixture()
        # Execute the receipt's pure admission too: a matching key cannot
        # stand in for the actual constructor's selected implementation.
        probe_path=ROOT/'probes/glm53_ep_tiled_compile.py'
        probe_tree=ast.parse(probe_path.read_text())
        probe_nodes=[copy.deepcopy(n) for n in probe_tree.body
                     if (isinstance(n,ast.FunctionDef) and n.name=='static_specialization')
                     or (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                         and n.targets[0].id=='STATIC_ROWS')]
        probe={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=probe_nodes,type_ignores=[])),
                     str(probe_path),'exec'),probe)
        owner=SimpleNamespace(_ep_tiled_workspace=SimpleNamespace(static=SimpleNamespace(max_rows=256),scratch=SimpleNamespace(max_active_clusters=48)))
        context=dict(decode=decode,md=SimpleNamespace(_DYNAMIC_KERNEL_CACHE={}))
        for rows in range(1,33):
            with self.subTest(rows=rows):
                key=decode.native_key(rows)
                self.assertEqual(len(key),19 if rows<=8 else 16)
                if rows<=8:
                    self.assertEqual(key[-4:],('bf16_scatter','glm53_ep_static_sf6_a_ring_v1',
                        'glm53_ep_static_sf6_word_unpack_v1','glm53_ep_static_bf16_scatter_v1'))
                else:
                    self.assertEqual(key[-1],'fp32_scatter')
                fused = decode.native_key(rows, route_mode='global', expert_map_len=288,
                                         expert_map_dtype='torch.int32', local_expert_offset=216)
                route_tail = ('glm53_ep_static_fused_route_v1', 288, 'torch.int32', 0)
                self.assertEqual(fused, key + route_tail)
                self.assertEqual(len(fused), 23 if rows <= 8 else 20)
                decode._EP_TILED_KERNEL_CACHE={key:object(), fused:object()}
                self.assertEqual(canary._cache_evidence(context,owner,rows)['keys'],[repr(fused)])
                wrong_ring=key[:-3]+key[-2:] if rows<=8 else key+('glm53_ep_static_sf6_a_ring_v1',)
                wrong_word=key[:-2]+key[-1:] if rows<=8 else key+('glm53_ep_static_sf6_word_unpack_v1',)
                wrong_tag=key[:-1]+('glm53_ep_static_bf16_scatter_v0',)
                wrong_abi=key[:15]+(('fp32_scatter' if rows<=8 else 'bf16_scatter'),)+key[16:]
                mutations=(wrong_ring,wrong_word,wrong_tag,decode.native_key(rows,False),
                                 wrong_abi,key[:-1] if rows<=8 else key+('glm53_ep_static_bf16_scatter_v1',),
                                 key[:1]+(rows+1,)+key[2:],key[:4]+('torch.int64',)+key[5:],
                                 key[:5]+(True,)+key[6:],key+('extra',))
                for mutation in mutations:
                    decode._EP_TILED_KERNEL_CACHE={key:object(), mutation+route_tail:object()}
                    with self.assertRaises(AssertionError):canary._cache_evidence(context,owner,rows)
                if rows in probe['STATIC_ROWS']:
                    selected=rows<=8
                    check=probe['static_specialization']
                    dtype='bfloat16' if selected else 'float32'
                    self.assertEqual(check(rows,key,selected,selected,selected,dtype),dict(a_ring=selected,
                        word_unpack=selected,scatter_bf16=selected,output_dtype=dtype,
                        scale_mode='sf6_v1',cache_tag=key[-1]))
                    for ring,word in ((selected,not selected),(not selected,selected),
                                      (selected,int(selected)),(int(selected),selected)):
                        with self.assertRaises(AssertionError):check(rows,key,ring,word,selected,dtype)
                    for scatter,output in ((not selected,dtype),(int(selected),dtype),
                                           (selected,'float32' if selected else 'bfloat16'),
                                           (selected,None)):
                        with self.assertRaises(AssertionError):check(rows,key,selected,selected,scatter,output)
                    for mutation in mutations[:6]+mutations[-1:]:
                        with self.assertRaises(AssertionError):check(rows,mutation,selected,selected,selected,dtype)
        dynamic=('dynamic','fp4','nvfp4',72,4096,2048,8,48,(128,128),'torch.int32',False,True,'swigluoai_uninterleave',1.,0.,10.,False,True,'glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1')
        context['md']._DYNAMIC_KERNEL_CACHE[dynamic]=object()
        self.assertEqual(canary._cache_evidence(context,owner,8192)['keys'],[repr(dynamic)])
        context['md']._DYNAMIC_KERNEL_CACHE={dynamic[:17]+(False,)+dynamic[18:]:object()}
        with self.assertRaises(AssertionError):canary._cache_evidence(context,owner,8192)
        context['md']._DYNAMIC_KERNEL_CACHE={dynamic[:-1]:object()}
        with self.assertRaises(AssertionError):canary._cache_evidence(context,owner,8192)

    def test_native_canary_rejects_local_only_or_wrong_global_map_namespace(self):
        decode = native_cache_fixture()
        owner = SimpleNamespace(_ep_tiled_workspace=SimpleNamespace(
            static=SimpleNamespace(max_rows=256), scratch=SimpleNamespace(max_active_clusters=48)))
        context = dict(decode=decode)
        for rows in (4, 6, 8, 12, 16, 24, 32):
            local = decode.native_key(rows)
            fused = decode.native_key(rows, route_mode='global', expert_map_len=288,
                                      expert_map_dtype='torch.int32')
            wrong = (local, fused[:-4], fused[:-4] + ('glm53_ep_static_fused_route_v0',) + fused[-3:],
                     fused[:-3] + (287,) + fused[-2:],
                     fused[:-2] + ('torch.int64', 0), fused[:-1] + (72,),
                     decode.native_key(rows, route_mode='global', local_expert_offset=216),
                     decode.native_key(rows, route_mode='global', expert_map_len=0,
                                       expert_map_dtype='torch.int32'))
            for key in wrong:
                decode._EP_TILED_KERNEL_CACHE = {key: object()}
                with self.subTest(rows=rows, key=key), self.assertRaises(AssertionError):
                    canary._cache_evidence(context, owner, rows)
            decode._EP_TILED_KERNEL_CACHE = {fused: object()}
            self.assertEqual(canary._cache_evidence(context, owner, rows)['keys'], [repr(fused)])

    def test_optimized_canary_requires_new_small_shape_keys_and_preserves_large_shapes(self):
        decode = native_cache_fixture(decode_opt=True)
        owner = SimpleNamespace(_ep_tiled_workspace=SimpleNamespace(
            static=SimpleNamespace(max_rows=256), scratch=SimpleNamespace(max_active_clusters=48)))
        context = dict(decode=decode)
        for rows in (4,6,8,12,16,24,32):
            kwargs = dict(route_mode='global',expert_map_len=288,expert_map_dtype='torch.int32')
            baseline = decode.native_key(rows,decode_opt=False,**kwargs)
            selected = decode.native_key(rows,**kwargs)
            self.assertEqual(selected, baseline + (('glm53_ep_static_sf6_q1_register_max_v5',)
                                                   if rows <= 8 else ()))
            decode._EP_TILED_KERNEL_CACHE = {baseline:object()}
            if rows <= 8:
                with self.assertRaises(AssertionError):
                    canary._cache_evidence(context,owner,rows)
            decode._EP_TILED_KERNEL_CACHE[selected] = object()
            evidence = canary._cache_evidence(context,owner,rows)
            self.assertEqual(evidence['keys'],[repr(selected)])
            self.assertIs(evidence['decode_opt'],rows<=8)
            if rows <= 8:
                for stale_tag in ('glm53_ep_static_sf6_q1_pair_v4',
                                  'glm53_ep_static_sf6_fc1_register_v2',
                                  'glm53_ep_static_sf6_fc2_out_of_place_v1'):
                    decode._EP_TILED_KERNEL_CACHE = {baseline + (stale_tag,): object()}
                    with self.assertRaises(AssertionError):
                        canary._cache_evidence(context,owner,rows)

    def test_source_uses_established_numeric_contract_and_bounded_graph_flow(self):
        old=sys.modules[PACKAGE+'.glm53_ep_local_selftest']
        self.assertIs(canary.compare,old.compare);self.assertIs(canary.check_control,old.check_control)
        self.assertEqual((canary.ROW_L2_FLOOR,canary.ROW_PEAK_FLOOR),(.02,.04))
        tree=ast.parse(SOURCE.read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for forbidden in ('torch.manual_seed','torch.cuda.manual_seed','torch.cuda.reset_peak_memory_stats','md.clear_sm120_moe_caches'):
            self.assertNotIn(forbidden,calls)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and ast.unparse(node.func) in ('torch.rand','torch.randn'):
                self.assertIn('generator',[kw.arg for kw in node.keywords])
                self.assertIn('dtype',[kw.arg for kw in node.keywords])
        text=SOURCE.read_text()
        self.assertNotIn('from probes',text)
        self.assertIn('graph.replay() if graph is not None else call()',text)
        self.assertIn('identity != cell["inputs"][int(changed)]',text)
        # Run the actual candidate scheduler with CPU fakes: appended shapes
        # must receive both phases and all three comparisons, including replay
        # of the original graph after the input bytes change.
        class Tensor:
            def fill_(self,value):return self
            def to(self,device):return self
            def __setitem__(self,key,value):pass
            def __eq__(self,value):return self
            def all(self):return True
        class Stream:
            def wait_stream(self,other):pass
        class Graph:
            def __init__(self):self.replays=0
            def replay(self):self.replays+=1
        graphs=[]
        def graph_factory():
            graph=Graph();graphs.append(graph);return graph
        cuda=SimpleNamespace(Stream=lambda **kw:Stream(),CUDAGraph=graph_factory,
            current_stream=lambda device:Stream(),graph=lambda *a,**kw:nullcontext(),
            stream=lambda side:nullcontext(),synchronize=Mock())
        tensors=tuple(Tensor() for _ in range(4))
        owner=SimpleNamespace(_ep_tiled_weight_views=object(),_ep_tiled_workspace=object())
        layer=SimpleNamespace(w13_weight=object(),w2_weight=object())
        launch=Mock(side_effect=lambda owner,out,*args:out)
        context=dict(torch=SimpleNamespace(cuda=cuda,full=lambda *a,**kw:Tensor(),
                     arange=lambda *a,**kw:Tensor(),int32='i32'),device='cpu-fake',
                     tiled=SimpleNamespace(launch_ep_tiled=launch))
        # Evaluate the actual reference-side cell expression, including its
        # graph admission, rather than hardcoding a second scheduler.
        capture=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_capture_references')
        cell_expr=next(n.value for n in ast.walk(capture) if isinstance(n,ast.Assign)
                      and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='cell')
        cell_code=compile(ast.Expression(cell_expr),str(SOURCE),'eval')
        cells=[]
        for name,rows,kind in canary.CASES:
            cell=eval(cell_code,dict(name=name,rows=rows,kind=kind,time=SimpleNamespace(time=lambda:0)))
            cell['inputs']=[{'phase':False},{'phase':True}];cells.append(cell)
        current={}
        def fill(context,buffers,index,changed,offset):
            current.update(index=index,changed=changed);return tensors
        handle=dict(receipt={'cases':cells},original={},offset=0,rng=None,buffers=tensors,
                    references=[[(Tensor(),Tensor()),(Tensor(),Tensor())] for _ in cells])
        with patch.object(canary,'_identities',return_value={}),patch.object(
                canary,'_packed_identity',return_value={}),patch.object(canary,'_fill',side_effect=fill),patch.object(
                canary,'_inputs',side_effect=lambda *a:{'phase':current['changed']}),patch.object(
                canary,'_scales',return_value={}),patch.object(canary,'compare',return_value={'bad_rows':0}) as compare,patch.object(
                canary,'_cache_evidence',return_value={}),patch.object(canary,'_check_rng'):
            canary._validate_candidate(context,owner,layer,handle)
        self.assertEqual(compare.call_count,72)
        self.assertEqual(len(graphs),9)
        self.assertTrue(all(graph.replays==4 for graph in graphs))
        for cell in cells:
            self.assertEqual((cell['verdict'],len(cell['candidate'])),('PASS',6))
        for cell in cells[9:]:
            self.assertTrue(cell['graph_replay'])
            self.assertEqual([r['phase'] for r in cell['candidate']],
                [phase+'-'+label for phase in ('initial','changed')
                 for label in ('C1-eager','C2-graph-current','C3-graph-side')])
        labels=next(n.iter for n in ast.walk(capture) if isinstance(n,ast.For)
                    and isinstance(n.target,ast.Name) and n.target.id=='label')
        self.assertEqual(ast.literal_eval(labels),('B1','B2','B3'))
        self.assertEqual(len(cells)*2*len(ast.literal_eval(labels)),72)


if __name__=='__main__':unittest.main()
