"""EP2/TP2 CPU contracts. No numerical, real lowering or collective claims."""
import ast, copy, io, os, sys, unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import test_glm53_ep_tiled_owner as old_owner
import test_glm53_ep_tiled_selftest as old_canary

C=old_canary.canary
G=sys.modules['_ep_tiled_canary_cpu.glm53_ep_shard_geometry']
H=G.ep_shard_geometry(144,1024)
ROOT=Path(__file__).resolve().parents[1]

def identity(p):
 ep,tp=p//2,p%2
 return ('glm53_ep2_tp2_loader_v1',288,144,4096,2048,1024,p,ep,tp,
         ep*144,(ep+1)*144,tp*1024,(tp+1)*1024)

def rank_owner(p=0):
 return SimpleNamespace(_use_ep=True,_ep_no_dummy=True,global_num_experts=288,
     num_local_experts=144,hidden_dim=4096,intermediate_size_per_partition=1024,topk=8,
     local_expert_offset=(p//2)*144,_glm53_hybrid_loader_identity=identity(p),
     _glm53_ep_parallelism=(2,p%2,2,p//2))

def hybrid_harness(test,p=0):
 h=old_owner.Harness(test)
 env=patch.dict(os.environ,{'VLLM_GLM53_EP_HYBRID_TP2':'1','VLLM_GLM53_EP_DECODE_OPT':'0'})
 env.start();test.addCleanup(env.stop)
 for key,value in vars(rank_owner(p)).items():setattr(h.owner,key,value)
 h.layer.w13_weight=old_owner.Tensor(H['raw_w13'],'u8')
 h.layer.w2_weight=old_owner.Tensor(H['raw_down'],'u8')
 for side,shape in ((1,(144,2048,256)),(2,(144,4096,64))):
  tensor=old_owner.Tensor(shape,'u8');name='w13_weight_scale' if side==1 else 'w2_weight_scale'
  setattr(h.owner,f'w{side}_scale',tensor);setattr(h.owner,f'w{side}_sf_mma',tensor)
  setattr(h.layer,name,tensor);h.layer._parameters[name]=tensor
  getattr(h.owner.quant_config,f'_w{side}').scale=tensor
 h.owner.g1_alphas=old_owner.Tensor((144,));h.owner.g2_alphas=old_owner.Tensor((144,))
 h.owner._fc2_input_scale=old_owner.Tensor((144,))
 h.workspace.shard=H
 h.workspace.static.state_E=h.workspace.static.weight_E=144;h.workspace.static.n=1024
 for field in ('packed_input','packed_input_scale','row_counts','token_map','token_weights','weight_expert_ids','global_to_local_expert'):
  old=getattr(h.workspace.static,field)
  setattr(h.workspace.static,field,old_owner.Tensor((144,*old.shape[1:]),old.dtype))
 original=h.views
 def views(*args,**kwargs):
  got=original(*args,**kwargs)
  got.sfb1_packed=got.reform_scales.fc1=old_owner.Tensor(H['sf6_fc1'],'u8')
  got.sfb2_packed=got.reform_scales.fc2=old_owner.Tensor(H['sf6_fc2'],'u8')
  return got
 h.md._get_weight_views.side_effect=views
 return h

def actual_native_key_fixture():
 """Only pure key statements from the actual compile factory."""
 path=ROOT/'overlay/modules/glm53_moe/moe_static_ep_tiled.py'
 tree=ast.parse(path.read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 factory=functions['ep_tiled_compile_spec']
 start=next(i for i,n in enumerate(factory.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='key')
 end=next(i for i in range(start,len(factory.body)) if isinstance(factory.body[i],ast.Return))
 fn=ast.parse('''def native_key(m, e=144, i=1024, sf6=True, decode_opt=False):
    max_rows, mac, topk_ids_dtype = 256, 48, "torch.int32"
    input_scales_are_reciprocal, fast_math, reform_sf_pack = False, True, sf6
    shard = ep_shard_geometry(e, i)
    require_hybrid_mode(shard, reform_sf_pack, decode_opt)
    geometry = ep_tiled_geometry(m, max_rows, mac)
    scale_mode = ep_tiled_scale_mode(reform_sf_pack)
    decode_opt = ep_tiled_decode_opt(m, reform_sf_pack, decode_opt)
    route_key = ("glm53_ep_static_fused_route_v1", 288, "torch.int32", 0)
''').body[0]
 selected=next(n for n in factory.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='scatter_bf16')
 statements=[n for n in factory.body[start:end] if not (isinstance(n,ast.If) and ast.unparse(n.test)=="route_mode == 'global'")]
 fn.body += [copy.deepcopy(selected)] + copy.deepcopy(statements) + [ast.Return(ast.Name('key',ast.Load()))]
 constants=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('EP_TILED_') and 'CACHE_TAG' in n.targets[0].id]
 ns=dict(ep_shard_geometry=G.ep_shard_geometry,require_hybrid_mode=G.require_hybrid_mode,
         ep_shard_cache_suffix=G.ep_shard_cache_suffix,_EP_TILED_DECODE_OPT=False)
 functions_selected=[copy.deepcopy(functions[name]) for name in ('ep_tiled_geometry','ep_tiled_scale_mode','ep_tiled_decode_opt','ep_tiled_decode_opt_enabled')]
 exec(compile(ast.fix_missing_locations(ast.Module(body=constants+functions_selected+[fn],type_ignores=[])),str(path),'exec'),ns)
 return SimpleNamespace(native_key=ns['native_key'],ep_tiled_geometry=ns['ep_tiled_geometry'],
     ep_tiled_decode_opt_enabled=ns['ep_tiled_decode_opt_enabled'],_EP_TILED_KERNEL_CACHE={})

class HybridContracts(unittest.TestCase):
 def test_exact_loader_identity_all_ranks_and_mutation_before_relayout(self):
  with patch.dict(os.environ,{'VLLM_GLM53_EP_HYBRID_TP2':'1','VLLM_GLM53_EP_DECODE_OPT':'0'}):
   for p in range(4):
    owner=rank_owner(p);self.assertEqual(G.owner_contract(owner),H)
    rank=G.hybrid_rank_contract(owner)
    self.assertEqual((rank['ep_rank'],rank['tp_rank']),(p//2,p%2))
    self.assertFalse(rank['collective_execution_verified'])
    for index in range(13):
     for replacement in (None,False,99):
      bad=list(identity(p));bad[index]=replacement
      owner._glm53_hybrid_loader_identity=tuple(bad)
      with self.subTest(rank=p,index=index,replacement=replacement),self.assertRaises(ValueError):G.owner_contract(owner)
    owner._glm53_hybrid_loader_identity=identity(p)
    for value in ([2,p%2,2,p//2],(4,p,1,0),(True,p%2,2,p//2)):
     owner._glm53_ep_parallelism=value
     with self.assertRaises(ValueError):G.owner_contract(owner)
   for flags in ({'VLLM_GLM53_EP_HYBRID_TP2':'0'},{'VLLM_GLM53_EP_DECODE_OPT':'1'},{'VLLM_GLM53_EP_HYBRID_TP2':'true'}):
    with patch.dict(os.environ,flags),self.assertRaises(ValueError):G.owner_contract(rank_owner())
  h=hybrid_harness(self);h.owner.local_expert_offset=72
  with self.assertRaises(ValueError):h.prepare()
  h.md.tile_expert_weights_inplace.assert_not_called()

 def test_baseline_flag_identity_separation_preserves_geometry_and_cache_suffix(self):
  h=old_owner.Harness(self)
  self.assertEqual(G.owner_contract(h.owner),G.ep_shard_geometry())
  self.assertEqual(G.ep_shard_cache_suffix(G.ep_shard_geometry()),())
  h.owner._glm53_hybrid_loader_identity=identity(0)
  with self.assertRaises(ValueError):h.prepare()
  h.md.tile_expert_weights_inplace.assert_not_called()
  self.assertEqual(G.memory_contract(H)['routed_weight_bytes'],G.memory_contract(G.ep_shard_geometry())['routed_weight_bytes'])
  self.assertEqual(G.memory_contract(H)['sf6_bytes'],G.memory_contract(G.ep_shard_geometry())['sf6_bytes'])
  self.assertEqual(G.memory_contract(H)['static_q0_bytes'],2*G.memory_contract(G.ep_shard_geometry())['static_q0_bytes'])

 def test_actual_owner_hybrid_shapes_release_dispatch_alias_and_rank_seal(self):
  h=hybrid_harness(self,3).prepare()
  self.assertEqual([e[0] for e in h.events[:3]],['before','relayout','views'])
  self.assertEqual(h.md._get_weight_views.call_args.kwargs['n'],1024)
  self.assertEqual(h.owner._ep_tiled_scale_receipt['packed_bytes'],85819392)
  self.assertEqual(h.owner._ep_tiled_scale_receipt['raw_bytes_released'],113246208)
  with redirect_stdout(io.StringIO()) as output:
   for rows in (1,4,6,8,12,16,24,32,33,2128,4096,6912,8192):
    args=h.inputs(rows);self.assertIs(h.module.launch_ep_tiled(h.owner,**args),args['output'])
    if rows<=32:
     self.assertIs(h.static.launch_ep_tiled_decode.call_args.kwargs['decode_opt'],False)
    else:
     self.assertEqual(h.remap.try_remap_ep_local.call_args.kwargs['num_local_experts'],144)
     self.assertEqual(h.md.launch_sm120_dynamic_moe.call_args.kwargs['num_experts'],144)
     self.assertEqual(h.md.launch_sm120_dynamic_moe.call_args.kwargs['n'],1024)
  self.assertIn('[ep-hybrid] LAUNCHED decode E144/H4096/I1024/top8 T=1',output.getvalue())
  self.assertNotIn('[ep-tiled] LAUNCHED',output.getvalue())
  warm=h.static.warm_ep_tiled_decode.call_args.kwargs
  self.assertEqual((warm['num_local_experts'],warm['intermediate_size'],warm['decode_opt']),(144,1024,False))
  # A different internally consistent rank must not reuse the prepared owner seal.
  h.owner._glm53_hybrid_loader_identity=identity(2);h.owner._glm53_ep_parallelism=(2,0,2,1)
  h.static.launch_ep_tiled_decode.reset_mock()
  with self.assertRaisesRegex(RuntimeError,'identity changed'):h.module.launch_ep_tiled(h.owner,**h.inputs(6))
  h.static.launch_ep_tiled_decode.assert_not_called()

 def test_workspace_isolated_by_shard_and_prewarms_explicit_baseline_math(self):
  h=old_owner.Harness(self)
  h.md.allocate_sm120_static_workspace=Mock(side_effect=lambda **kw:SimpleNamespace(**kw))
  h.md.allocate_sm120_dynamic_workspace=Mock(side_effect=lambda **kw:SimpleNamespace(device=kw['device'],max_rows=kw['routed_rows']))
  h.static.allocate_ep_tiled_decode_scratch=Mock(return_value=SimpleNamespace(scatter_fp32=old_owner.Tensor((32,4096))))
  h.md._get_dynamic_kernel=Mock()
  old=h.real_shared_workspace('cuda:0',8192)
  new=h.real_shared_workspace('cuda:0',8192,shard=H)
  self.assertIsNot(old,new);self.assertIs(new,h.real_shared_workspace('cuda:0',8192,shard=H))
  self.assertEqual((new.static.state_E,new.static.n),(144,1024))
  self.assertEqual((old.static.state_E,old.static.n),(72,2048))
  self.assertEqual(len(h.module._WORKSPACES),2)
  self.assertEqual(h.md._get_dynamic_kernel.call_args.args[:5],(144,8192,4096,1024,8))
  self.assertEqual(h.static.warm_ep_tiled_decode.call_args.kwargs,dict(reform_sf_pack=True,num_local_experts=144,intermediate_size=1024,decode_opt=False))

 def test_routes_and_stock_reference_borrow_all_shard_weights_without_copy(self):
  for offset in (0,144):
   for changed in (False,True):
    rows=C.route_rows(33,'remote',offset,changed,num_local_experts=144)
    self.assertTrue(all(0<=e<288 and not offset<=e<offset+144 for row in rows for e in row))
    mixed=C.route_rows(6,'mixed',offset,changed,num_local_experts=144)
    self.assertIn(offset+143,mixed[0]);self.assertIn(-1,mixed[0]);self.assertIn(288,mixed[0])
    self.assertEqual(mixed[0][-2],mixed[0][-1])
   with self.assertRaises(ValueError):C.route_rows(6,'mixed',offset+72,num_local_experts=144)
  class Owner(SimpleNamespace):
   def _apply_ep_compact(self,out,x,w1,w2,ids,weights):
    self.captured=(self.num_local_experts,self._kernel_num_experts,w1,w2,self.w1_sf_mma,self.w2_sf_mma)
    return out
  fields=dict(w1_sf_mma=object(),w2_sf_mma=object(),g1_alphas=object(),g2_alphas=object(),_fc2_input_scale=object(),
              _activation_str='swigluoai_uninterleave',_swiglu_alpha=1.,_swiglu_beta=0.,_swiglu_limit=10.,num_local_experts=144)
  original=Owner(**fields);borrow=C._reference_owner(original)
  w1,w2,out=object(),object(),object()
  self.assertIs(borrow._apply_ep_compact(out,None,w1,w2,None,None),out)
  self.assertEqual(borrow.captured,(144,144,w1,w2,original.w1_sf_mma,original.w2_sf_mma))
  context=dict(torch=object(),md=object(),device='cuda:0',shard=H)
  wrong=SimpleNamespace(w13_weight=SimpleNamespace(shape=(72,4096,2048)),w2_weight=SimpleNamespace(shape=(72,4096,1024)))
  with self.assertRaisesRegex(RuntimeError,'original row-major'):C._capture_references(context,original,wrong,{})

 def test_hybrid_packed_shape_same_byte_budget_and_no_early_release(self):
  owner,layer=old_canary.packed_owner();owner.num_local_experts=144;owner.intermediate_size_per_partition=1024
  owner._ep_tiled_weight_views.sfb1_packed.shape=H['sf6_fc1'];owner._ep_tiled_weight_views.sfb2_packed.shape=H['sf6_fc2']
  with patch.object(C,'_tensor_identity',side_effect=old_canary.tensor_identity):
   proof=C._packed_identity(owner,layer)
   self.assertEqual((proof['preparation_contract']['raw_bytes'],proof['preparation_contract']['packed_bytes']),(113246208,85819392))
   self.assertFalse(proof['raw_release_acceptance'])
   owner._ep_tiled_weight_views.sfb1_packed.shape=(72,512,1552)
   with self.assertRaises(AssertionError):C._packed_identity(owner,layer)

 def test_real_native_key_statements_hybrid_namespace_and_dynamic_rejections(self):
  decode=actual_native_key_fixture();context=dict(decode=decode,shard=H,md=SimpleNamespace(_DYNAMIC_KERNEL_CACHE={}))
  owner=SimpleNamespace(_ep_tiled_workspace=SimpleNamespace(static=SimpleNamespace(max_rows=256),scratch=SimpleNamespace(max_active_clusters=48)))
  for rows in range(1,33):
   key=decode.native_key(rows);self.assertEqual(len(key),26 if rows<=8 else 23)
   self.assertEqual(key[-3:],(G.HYBRID_TAG,144,1024))
   decode._EP_TILED_KERNEL_CACHE={key:object()}
   evidence=C._cache_evidence(context,owner,rows);self.assertEqual(evidence['keys'],[repr(key)]);self.assertIs(evidence['decode_opt'],False)
   for wrong in (key[:-3],key[:-3]+(G.HYBRID_TAG,72,2048),key[:-3]+('glm53_ep_static_sf6_q1_register_max_v5',)+key[-3:],key[:15]+('fp32_scatter' if rows<=8 else 'bf16_scatter',)+key[16:]):
    decode._EP_TILED_KERNEL_CACHE={wrong:object()}
    with self.assertRaises(AssertionError):C._cache_evidence(context,owner,rows)
  dynamic=('dynamic','fp4','nvfp4',144,4096,1024,8,48,(128,128),'torch.int32',False,True,'swigluoai_uninterleave',1.,0.,10.,False,True,'glm53_ep_prefill_local_fp32_v2','glm53_ep_tiled_sf6_v1',G.HYBRID_TAG)
  context['md']._DYNAMIC_KERNEL_CACHE={dynamic:object()};self.assertEqual(C._cache_evidence(context,owner,8192)['keys'],[repr(dynamic)])
  for wrong in (dynamic[:-1],dynamic[:3]+(72,4096,2048)+dynamic[6:],dynamic[:17]+(False,)+dynamic[18:]):
   context['md']._DYNAMIC_KERNEL_CACHE={wrong:object()}
   with self.assertRaises(AssertionError):C._cache_evidence(context,owner,8192)

 def test_schema_rank_source_once_cache_and_sticky_failure_separate(self):
  C._STATES.clear();rt=old_canary.runtime();rt.update(shard=H,loader_identity=identity(0),rank_geometry={'physical_rank':0})
  output=io.StringIO()
  def capture(*args):return dict(references=[],buffers=[],original={})
  with redirect_stdout(output),patch.object(C,'_runtime',side_effect=lambda *a:rt),patch.object(C,'_memory',return_value={}),patch.object(C,'_capture_references',side_effect=capture) as capture_mock,patch.object(C,'_validate_candidate'):
   owner,layer=object(),object();h=C.before_relayout(owner,layer);r=C.after_relayout(owner,layer,h)
   self.assertEqual(r['schema'],2);self.assertEqual(r['geometry'],dict(E=144,K=4096,I=1024,top8=8))
   self.assertEqual(r['loader_identity'],list(identity(0)))
   self.assertTrue(C.before_relayout(object(),object())['cached'])
   rt=dict(rt,loader_identity=identity(1),rank_geometry={'physical_rank':1})
   owner,layer=object(),object();other=C.before_relayout(owner,layer)
   self.assertNotIn('cached',other)
   with patch.object(C,'_validate_candidate',side_effect=AssertionError('unchanged numeric threshold')):
    with self.assertRaises(RuntimeError):C.after_relayout(owner,layer,other)
   with self.assertRaisesRegex(RuntimeError,'previously failed'):C.before_relayout(object(),object())
   rt=old_canary.runtime();base=C.before_relayout(object(),object());self.assertEqual(base['receipt']['schema'],1)
   self.assertEqual(capture_mock.call_count,3)
  self.assertIn('[ep-hybrid-selftest] PASS ',output.getvalue());self.assertIn('[ep-hybrid-selftest] FAIL ',output.getvalue())
  C._STATES.clear()

 def test_hybrid_runs_unchanged_complete_numeric_graph_scheduler(self):
  original=C._validate_candidate
  def hybrid(context,owner,layer,handle):
   context=dict(context,shard=H)
   return original(context,owner,layer,handle)
  with patch.object(C,'_validate_candidate',side_effect=hybrid):
   old_canary.AdmissionTests.test_source_uses_established_numeric_contract_and_bounded_graph_flow(self)
