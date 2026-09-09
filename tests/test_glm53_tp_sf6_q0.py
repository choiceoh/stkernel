"""Actual-source CPU contracts for a Q0-only TP SF6 candidate; no accelerator work."""
import ast,copy,gzip,hashlib,struct,sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import test_glm53_ep_route_scale_cache as cache_oracle

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'overlay/modules/glm53_moe/moe_dynamic_gated_sf6_q0.py'
DISPATCH=ROOT/'overlay/modules/glm53_moe/moe_dispatch.py'
SF6=ROOT/'overlay/modules/glm53_moe/moe_dynamic_gated_sf6.py'
STOCK=ROOT/'measurements/glm53_ep_local_20260908/cpu13/stock-gated.py.gz'
EXACT=dict(enabled=True,E=288,m=8192,k=4096,n=512,num_topk=8,tile_m=128,
 quant_mode='nvfp4',tiled=True,reform_sf_pack=True,activation='swigluoai_uninterleave',
 swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.,share_input_across_experts=False)

def function(path,name):
 return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(nodes):return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
def assigned(node,name):return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
def loop(node):return isinstance(node,ast.While) and ast.unparse(node.test)=='produce_active > Int32(0)'
def execute(nodes,ns):
 tree=ast.Module(body=copy.deepcopy(nodes),type_ignores=[])
 exec(compile(ast.fix_missing_locations(tree),str(MODULE),'exec'),ns)
def load(path,name,ns):
 execute([ast.parse('from __future__ import annotations').body[0],function(path,name)],ns);return ns[name]

class Cache288(cache_oracle.SourceCache):
 def __init__(self):
  with patch.object(cache_oracle,'KERNEL',MODULE):super().__init__()
  self.row_counts=[0]*288
  def atomic(pointer,value):
   name,expert=pointer
   if name!='expert_rows' or not 0<=expert<288 or value!=1:raise AssertionError('invalid TP allocation')
   old=self.row_counts[expert];self.row_counts[expert]+=1;return old
  self.base.update(num_experts=288,expert_tile_base=[i*64 for i in range(288)],atomic_add_global_i32=atomic)

class SourceContracts(unittest.TestCase):
 def test_stock_bf16_initialization_histogram_prefix_and_task_publication_preserved(self):
  raw=gzip.decompress(STOCK.read_bytes())
  self.assertEqual(hashlib.sha256(raw).hexdigest(),'993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445')
  stock=next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.FunctionDef) and n.name=='initialize_route_q0_and_publish')
  actual=function(MODULE,stock.name)
  si=next(i for i,n in enumerate(stock.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='tidx == Int32(0)' and any(isinstance(k,ast.Name) and k.id=='q0_bulk_barrier_init' for k in ast.walk(n)))
  ai=next(i for i,n in enumerate(actual.body) if assigned(n,'expert_scales_addr'))
  self.assertEqual(dump(stock.body[:si]),dump(actual.body[:ai]))
  sl=next(i for i,n in enumerate(stock.body) if loop(n));al=next(i for i,n in enumerate(actual.body) if loop(n))
  self.assertEqual(dump(stock.body[sl+1:]),dump(actual.body[al+1:]))
  self.assertEqual(ast.unparse(next(n.value for n in actual.body if assigned(n,'cols_u32'))),'cols // Int32(2)')
  # Nine-warp E288 retains the stock serial prefix; do not reuse E256 scan.
  self.assertIn('num_experts != Int32(256)',ast.unparse(actual))

 def test_bf16_zero_writes_cover_only_full_output_and_never_guard_bytes(self):
  fn=function(MODULE,'initialize_route_q0_and_publish')
  begin=next(i for i,n in enumerate(fn.body) if assigned(n,'scatter_total_u32'))
  end=next(i for i,n in enumerate(fn.body[begin:],begin) if isinstance(n,ast.If) and ast.unparse(n.test)=='flat_tid == Int32(0)')
  for tokens in (4096,4097,8192,16384):
   for ctas in (1,48):
    writes=[]
    def store(addr,*values):
     self.assertEqual(values,(0,0,0,0));writes.append(addr)
    class Tail:
     def __setitem__(self,key,value):raise AssertionError('aligned H4096 cannot need scalar zero tail')
    for flat in range(ctas*288):
     execute(fn.body[begin:end],dict(Int32=int,Int64=int,Uint32=int,num_tokens=tokens,
      cols_u32=2048,flat_tid=flat,flat_stride=ctas*288,scatter_base=0,
      st_global_v4_u32=store,scatter_output_u32=Tail()))
    self.assertEqual(sorted(writes),list(range(0,tokens*4096*2,16)))

 def test_sf6_inherited_source_is_pinned_and_only_q0_plus_guards_are_overridden(self):
  tree=ast.parse(MODULE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
  self.assertEqual(ast.unparse(cls.bases[0]),'_sf6.MoEGatedDynamicKernelSF6')
  self.assertEqual({n.name for n in cls.body if isinstance(n,ast.FunctionDef)},
   {'_setup_attributes','_check_sf6_shapes','initialize_route_q0_and_publish'})
  constant=next(n.value.value for n in tree.body if assigned(n,'SF6_SOURCE_SHA256'))
  self.assertEqual(hashlib.sha256(SF6.read_bytes()).hexdigest(),constant)
  for forbidden in ('scatter_sC_to_gmem','load_fc1_tma_slice','load_fc2_tma_tile','kernel','__call__'):
   self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name==forbidden for n in cls.body))

 def test_exact_optin_gate_boundaries_and_missing_scale_or_tile_contract(self):
  gate=load(DISPATCH,'_tp_sf6_q0_eligible',{})
  self.assertTrue(gate(**EXACT))
  for m in (4096,4097,8192):self.assertTrue(gate(**(EXACT|dict(m=m))))
  mutations=dict(enabled=[False],E=[72,287,289],m=[4095,8193,True,8192.0],k=[4095,8192],
   n=[511,513,2048],num_topk=[1,7,9],tile_m=[16,32,64],quant_mode=['mxfp4'],
   tiled=[False],reform_sf_pack=[False],activation=['silu'],swiglu_alpha=[1.1],
   swiglu_beta=[1.],swiglu_limit=[None,9.],share_input_across_experts=[True])
  for field,values in mutations.items():
   for value in values:
    with self.subTest(field=field,value=value):self.assertFalse(gate(**(EXACT|{field:value})))

 def test_private_override_selects_exact_two_handles_and_rejects_invalid_requests(self):
  fn=function(DISPATCH,'_get_dynamic_kernel')
  begin=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If)
             and '_tp_sf6_q0_override is not None' in ast.unparse(n.test))
  end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If)
           and ast.unparse(n.test)=='tp_sf6_q0')
  gate=load(DISPATCH,'_tp_sf6_q0_eligible',{})
  def select(override,enabled=True,**changes):
   ns=dict(EXACT,_tp_sf6_q0_override=override,_TP_SF6_Q0_ENABLED=enabled,
           _tp_sf6_q0_eligible=gate)
   ns.update(changes);execute(fn.body[begin:end],ns);return ns['tp_sf6_q0']
  for enabled in (False,True):
   self.assertEqual(select(None,enabled),enabled)
   self.assertFalse(select(False,enabled));self.assertTrue(select(True,enabled))
  for override in (0,1,'1',[],{}):
   with self.subTest(override=override),self.assertRaises(TypeError):select(override)
  for change in (dict(E=72),dict(m=2048),dict(m=8193),dict(n=2048),
                 dict(reform_sf_pack=False),dict(tiled=False),dict(share_input_across_experts=True)):
   with self.subTest(change=change),self.assertRaises(ValueError):select(True,**change)
   self.assertFalse(select(None,**change));self.assertFalse(select(False,**change))

 def test_actual_launcher_forwards_private_selection_without_runtime_abi_changes(self):
  actual=function(DISPATCH,'launch_sm120_dynamic_moe')
  argument=next(a for a in actual.args.kwonlyargs if a.arg=='_tp_sf6_q0_override')
  self.assertIsNone(actual.args.kw_defaults[actual.args.kwonlyargs.index(argument)].value)
  call=next(n for n in ast.walk(actual) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Name) and n.func.id=='_get_dynamic_kernel')
  self.assertEqual(ast.unparse(next(k.value for k in call.keywords
                  if k.arg=='_tp_sf6_q0_override')),'_tp_sf6_q0_override')
  calls=[n for n in ast.walk(actual) if isinstance(n,ast.Call)
         and isinstance(n.func,ast.Name) and n.func.id=='compiled']
  self.assertEqual(len(calls),1)
  self.assertEqual([ast.unparse(arg) for arg in calls[0].args],['*runtime_args'])
  self.assertFalse(calls[0].keywords)
  self.assertEqual(ast.unparse(actual.body[-1]),'return scatter_output')
  runtime=next(n.value for n in actual.body if isinstance(n,ast.AnnAssign)
               and isinstance(n.target,ast.Name) and n.target.id=='runtime_args')
  self.assertEqual(len(runtime.elts),34)
  self.assertEqual(ast.unparse(runtime.elts[25]),'accumulator.data_ptr()')
  self.assertEqual(ast.unparse(runtime.elts[14]),'weights.w13_fp4')
  self.assertEqual(ast.unparse(runtime.elts[16]),'weights.down_fp4')

 def test_cache_isolation_preserves_all_original_sf6_key_fields(self):
  key=load(DISPATCH,'_dynamic_kernel_cache_key',{})
  args=dict(activation_precision='fp4',quant_mode='nvfp4',E=288,k=4096,n=512,num_topk=8,
   mac=48,mma_tiler_mn=(128,128),topk_ids_dtype='i32',input_scales_are_reciprocal=False,
   fast_math=True,activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,
   swiglu_limit=10.,share_input_across_experts=False,tiled=True,reform_sf_pack=True)
  expected=('dynamic','fp4','nvfp4',288,4096,512,8,48,(128,128),'i32',False,True,
   'swigluoai_uninterleave',1.,0.,10.,False,True,'sf6_direct_prefill_v1')
  self.assertEqual(key(**args),expected)
  self.assertEqual(key(**(args|dict(tp_sf6_q0=True))),expected+('glm53_tp_sf6_q0_v1',))

class ProducerContracts(unittest.TestCase):
 def test_all_eight_tp_routes_including_zero_weights_match_stock_global_allocation(self):
  stock=next(n for n in ast.walk(ast.parse(gzip.decompress(STOCK.read_bytes()))) if isinstance(n,ast.FunctionDef) and n.name=='initialize_route_q0_and_publish')
  branch=next(n for n in ast.walk(stock) if isinstance(n,ast.If) and ast.unparse(n.test)=='cutlass.const_expr(self.share_input_across_experts)' and n.orelse)
  producer=next(n for n in ast.walk(ast.Module(body=branch.orelse,type_ignores=[])) if isinstance(n,ast.If) and ast.unparse(n.test)=='lane_id == Int32(0)')
  for ids in ([0,31,32,71,72,255,256,287],[287]*8,[0]*8):
   new=Cache288();old=Cache288();new.experts([0x3f800000]*288)
   weights=[cache_oracle.f32(v) for v in (0.,1.,0.,-0.,2.,0.,3.,0.)]
   new.allocate(0,ids,weights)
   ns=dict(old.base,lane_id=0,token_idx=0,route_slot_base=0,
    topk_ids=[cache_oracle.ExpertId(i) for i in ids],topk_weights=weights)
   execute([producer],ns)
   self.assertEqual(new.row_counts,old.row_counts);self.assertEqual(sum(new.row_counts),8)
   self.assertEqual(new.global_writes,old.global_writes)
   self.assertEqual(new.state(0)&15,8)

 def test_scale_cache_and_equal_state_refresh_every_invocation_in_every_warp_lane(self):
  cache=Cache288()
  cases=[[0x3f800000]*8,[0,0x80000000]*4,[0x3f800000+i*0x10000 for i in range(8)],
   [0x7fc00001]+[0x3f800000]*7,[0x3f800000]*7+[0x7fc00002]]
  ids=[0,31,32,71,72,255,256,287]
  for selected in cases:
   all_bits=[0x40000000]*288
   for i,bits in zip(ids,selected):all_bits[i]=bits
   cache.experts(all_bits)
   for warp in range(4):
    cache.allocate(warp,ids)
    for lane in range(32):
     self.assertEqual(cache.consume(warp,lane=lane),cache_oracle.old_quantizer_inputs(selected))
   self.assertEqual([cache.memory[cache.EXPERTS+i*4] for i in range(288)],all_bits)

 def test_scale_offsets_and_publication_order_preserve_old_m128_contract(self):
  cache=Cache288();cache.experts([0x3f800000]*288)
  producer,barriers,consumer=cache.publication_order
  self.assertTrue(any(producer<b<consumer for b in barriers))
  batch_barriers,dispatch=cache.batch_publication_order
  self.assertTrue(any(b<dispatch for b in batch_barriers))
  for warp in range(4):
   cache.allocate(warp,[0,31,32,71,72,255,256,287])
   rows=[cache.memory[cache.ROWS+(warp*32+s)*4] for s in range(8)]
   for sf in range(256):
    self.assertEqual(cache.consume_offsets(warp,sf),[cache_oracle.scalar_scale_offset(r,sf) for r in rows])
  for tokens in (4096,4097,8192,8193,16384):
   base=tokens//4*4
   for warp in range(4):self.assertEqual(cache.active(warp,base,tokens),base+warp<tokens)

if __name__=='__main__':unittest.main()
