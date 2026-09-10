"""Pure CPU contracts for native FC1 SF6 register restore; no CuTe/GPU import.

The real CuTe host validator is present and tested with adversarial synthetic
layouts here. Only a later actual CuTe lowering can prove its real lane mapping.
"""
import ast
import copy
import gzip
import hashlib
import os
from pathlib import Path
import random
import types
import unittest

ROOT=Path(os.environ.get('GLM53_EP_FC1_TEST_ROOT',Path(__file__).resolve().parents[1]))
SOURCE=Path(os.environ.get('GLM53_EP_FC1_TEST_SOURCE',ROOT/'overlay/modules/glm53_moe/moe_static_ep_tiled.py'))
ORACLE=ROOT/'measurements/glm53_ep_tiled_20260909/ep76_onepass1/source/moe_static_ep_tiled.py.gz'
BASE_SHA='f975bbf6faea6899a8d7af0f1a182548da784b238209c2580ffd2b562a4be952'
BASE=gzip.decompress(ORACLE.read_bytes()).decode()
assert hashlib.sha256(BASE.encode()).hexdigest()==BASE_SHA
TEXT=SOURCE.read_text()
def fn(name,text=TEXT):
 return copy.deepcopy(next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name))
def run_function(name,ns):
 node=fn(name);node.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom('__future__',[ast.alias('annotations')],0),node],type_ignores=[])),str(SOURCE),'exec'),ns)
 return ns[name]
def dump(node):return ast.dump(node,include_attributes=False)
class Select(ast.NodeTransformer):
 def __init__(self,opt):self.opt=opt
 def visit_If(self,n):
  if ast.unparse(n.test)=='cutlass.const_expr(self.ep_decode_opt)':
   out=[]
   for item in n.body if self.opt else n.orelse:
    value=self.visit(item);out.extend(value if isinstance(value,list) else [value])
   return out
  return self.generic_visit(n)
 def visit_IfExp(self,n):
  if ast.unparse(n.test)=='self.ep_decode_opt':return self.visit(n.body if self.opt else n.orelse)
  return self.generic_visit(n)
def u32(x):return int(x)&0xffffffff
class ByteRegister(list):
 element_type='FP8-bits'
 @property
 def shape(self):return (len(self),1)

class DecodeOptimizationTests(unittest.TestCase):
 def test_actual_register_loader_exhausts_base_delta_lanes_and_preserves_bits(self):
  memory=bytearray(4096);reads=[]
  def load(addr):
   reads.append(addr);slot=addr//2048;rel=addr%2048
   self.assertIn(slot,(0,1));self.assertEqual(addr%4,0)
   self.assertTrue(0<=rel<=1020 or 1024<=rel<=1532)
   return int.from_bytes(memory[addr:addr+4],'little')
  def recast(reg,dtype):
   self.assertIsInstance(reg,ByteRegister);self.assertIs(dtype,u8)
   return reg
  u8=lambda x:int(x)&255
  ns=dict(Int32=int,cutlass=types.SimpleNamespace(Uint32=u32,Uint8=u8,range_constexpr=range),
          cute=types.SimpleNamespace(size=len,recast_tensor=recast),_ld_shared_i32_volatile=load)
  run_function('_sf6_unpack_word',ns);call=run_function('_sf1_load_register_words',ns)
  rng=random.Random(72)
  # Exhaust every base and each delta at each output byte position. Vary
  # aligned source word and stage; neighbours deliberately differ.
  for base in range(256):
   for delta in range(64):
    for lane in range(4):
     e=4*((base*17+delta*5+lane)%512);slot=(base+delta+lane)%2;addr=slot*2048
     ds=[(delta+13*i+7)&63 for i in range(4)];ds[lane]=delta
     lo=sum((d&15)<<(4*i) for i,d in enumerate(ds));hi=sum((d>>4)<<(2*i) for i,d in enumerate(ds))
     memory[addr+e//2:addr+e//2+2]=lo.to_bytes(2,'little');memory[addr+1024+e//4]=hi
     before=bytes(memory);out=ByteRegister([0]*4);reads.clear()
     call(None,addr,list(range(e,e+4)),out,(base&127)*0x01010101,(base&128)*0x01010101)
     self.assertEqual(out,[(base+d)&255 for d in ds]);self.assertEqual(bytes(memory),before)
     self.assertEqual(len(reads),2)
  # All physical byte offsets in both stages with random packed planes.
  for stage in range(2):
   memory[:]=rng.randbytes(4096);payload=memory[stage*2048:stage*2048+1552];base=payload[1536]
   for e in range(0,2048,4):
    out=ByteRegister([0]*4)
    call(None,stage*2048,list(range(e,e+4)),out,(base&127)*0x01010101,(base&128)*0x01010101)
    expected=[(base+((payload[i//2]>>(4*(i%2)))&15)+16*((payload[1024+i//4]>>(2*(i%4)))&3))&255 for i in range(e,e+4)]
    self.assertEqual(out,expected)
  source=ast.unparse(fn('_sf1_load_register_words'))
  self.assertNotIn('Float8',source);self.assertNotIn('_st_shared',source)
  self.assertNotIn('arrive_and_wait',source)

 def test_actual_host_coordinate_guard_rejects_unsafe_synthetic_layouts(self):
  offset=run_function('_ep_sf1_static_offset',{})
  owner=types.SimpleNamespace(decode_reform=True,reform_sf_pack=True,fc1_tile_n=128,fc1_tile_k=256,
       sf1_packed_blocks=1,sf1_block_bytes=2048,num_mma_warps=4,num_k_blocks1=4,sf_dtype=object(),
       tiled_mma1=types.SimpleNamespace(permutation_mnk=(16,128,256)),sfb1_smem_layout_staged='actual-layout',fc1_tile_shape_mnk=(16,128,256))
  owner._ep_sf1_static_offset=lambda t,i:offset(owner,t,i)
  class Group:
   def __init__(self,tid,kb,mode):
    self.iterator=(tid%32)*16+(tid//32)*4+kb*512
    self.layout=types.SimpleNamespace(shape=((2,2),1),stride=((1,2),0))
    if mode=='unaligned':self.iterator+=1
    if mode=='gap':self.layout.stride=((1,3),0)
    if mode=='outside':self.iterator+=2048
    if mode=='symbolic':self.iterator=object()
  class Slot:
   shape=(4,1,4)
   def __init__(self,tid,mode):self.tid=tid;self.mode=mode
   def __getitem__(self,key):return Group(self.tid,key[2],self.mode)
  class Partition:
   def __init__(self,tid,mode):self.tid=tid;self.mode=mode
   def __getitem__(self,key):return Slot(self.tid,self.mode)
  class Offsets:
   def __getitem__(self,key):return types.SimpleNamespace(iterator=key[-1]*2048)
  mode=['ok'];events=[]
  def make_tensor(engine,layout):
   events.append((engine,layout));return Offsets()
  copier=types.SimpleNamespace(get_slice=lambda tid:types.SimpleNamespace(partition_S=lambda offsets:Partition(tid,mode[0])))
  owner._dense_cls=types.SimpleNamespace(_get_layoutSFB_TV=lambda *args:'original-SFB-TV')
  cute=types.SimpleNamespace(make_copy_atom=lambda *a:None,nvgpu=types.SimpleNamespace(CopyUniversalOp=lambda:None),
    make_tiled_copy=lambda *a:copier,make_tensor=make_tensor,local_tile=lambda x,*a:x,slice_=lambda *a:None,
    filter_zeros=lambda x:x,size=lambda x,mode=None:(4 if mode else 16) if isinstance(x,Slot) else (4 if isinstance(x,Group) else x))
  check=run_function('_check_ep_sf1_register_layout',dict(cute=cute))
  check(owner);self.assertTrue(owner.ep_sf1_register_layout_proven)
  self.assertEqual(events,[(0,'actual-layout')])
  self.assertEqual(owner.ep_sf1_register_layout_receipt,dict(proven=True,threads=128,raw_stage_bytes=2048,num_k_blocks=4,word_coverage_bytes=2048,stages=2,stage_stride_bytes=2048,copy_shape='(4, 1, 4)',words_per_thread=[4],offset_engine='static_scalar_physical_layout',slot_zero_relative_offsets=True))
  for bad in ('unaligned','gap','outside','symbolic'):
   mode[0]=bad
   with self.assertRaises(ValueError):check(owner)
  # Explicit nested/zero-stride arithmetic; no coercion of symbolic MLIR values.
  t=types.SimpleNamespace(iterator=9,layout=types.SimpleNamespace(shape=((2,3),2),stride=((1,8),0)))
  self.assertEqual([offset(owner,t,i) for i in range(12)],[9,10,17,18,25,26]*2)
  with self.assertRaises(ValueError):offset(owner,t,12)
  guard=run_function('_check_ep_sf1_register_copy',{})
  dtype=types.SimpleNamespace(width=8);owner.sf_dtype=dtype
  guard(owner,types.SimpleNamespace(shape=(4,1,4)),types.SimpleNamespace(shape=(4,1,4),element_type=dtype))
  with self.assertRaises(ValueError):guard(owner,types.SimpleNamespace(shape=(8,1,4)),types.SimpleNamespace(shape=(4,1,4),element_type=dtype))

 def test_complete_baseline_kernel_matches_pinned_b41_and_only_fc1_changes_on(self):
  old=Select(False).visit(fn('kernel',BASE));off=Select(False).visit(fn('kernel'))
  self.assertEqual(dump(off),dump(old))
  # Reconstruct the precise stock FC1 copies/expand from their replacements.
  original_expand=next(n for n in ast.walk(old) if isinstance(n,ast.If) and ast.unparse(n.test)=='cutlass.const_expr(self.reform_sf_pack)' and 'sfb1_base_addr' in ast.unparse(n) and any(isinstance(x,ast.Call) and ast.unparse(x.func)=='self._sf_expand_stage' for x in ast.walk(n)))
  removed={'sf1_offset_tensor','sf1_offset_partition','sf1_register_offsets'}
  class Restore(ast.NodeTransformer):
   def visit_If(self,n):
    if ast.unparse(n.test)=='cutlass.const_expr(self.reform_sf_pack)' and any(isinstance(x,ast.Name) and x.id=='sf1_packed_addr' for x in ast.walk(n)):
     n.body=copy.deepcopy(original_expand.body)
    return self.generic_visit(n)
   def visit_Assign(self,n):
    if any(isinstance(t,ast.Name) and t.id in removed for t in n.targets):return None
    return self.generic_visit(n)
   def visit_Expr(self,n):
    if isinstance(n.value,ast.Call):
     c=n.value
     if ast.unparse(c.func) in ('self._check_ep_storage','self._check_ep_sf1_register_copy'):return None
     if ast.unparse(c.func)=='self._sf1_load_register_words':
      dst=copy.deepcopy(c.args[2]);src=copy.deepcopy(dst);src.value=ast.Name('fz_csSFB_p',ast.Load())
      return ast.Expr(ast.Call(ast.Attribute(ast.Name('cute',ast.Load()),'copy',ast.Load()),[ast.Name('smem_copy_SFB1',ast.Load()),src,dst],[]))
    return self.generic_visit(n)
  restored=Restore().visit(Select(True).visit(fn('kernel')))
  self.assertEqual(dump(restored),dump(old))
  for name in ('__init__','__call__','_call_global','launch_ep_tiled_decode','_global_route_id','_sf_expand_stage'):
   self.assertEqual(dump(fn(name)),dump(fn(name,BASE)),name)

 def test_storage_metadata_fc2_and_actual_capacity_guard_use_baseline(self):
  old=Select(False).visit(fn('kernel',BASE));new=Select(True).visit(fn('kernel'))
  struct=lambda node:next(n for n in ast.walk(node) if isinstance(n,ast.ClassDef) and n.name=='Storage')
  self.assertEqual(dump(struct(old)),dump(struct(new)))
  guard=run_function('_check_ep_storage',{})
  owner=types.SimpleNamespace(smem_bytes=98304,smem_capacity=101376,threads_per_cta=160)
  guard(owner,types.SimpleNamespace(size_in_bytes=lambda:98304));self.assertEqual(owner.ep_storage_bytes,98304)
  for n in (97280,100352,101376):
   with self.assertRaises(ValueError):guard(owner,types.SimpleNamespace(size_in_bytes=lambda n=n:n))
  owner.smem_capacity=99327
  with self.assertRaises(ValueError):guard(owner,types.SimpleNamespace(size_in_bytes=lambda:98304))
  self.assertEqual(ast.unparse(fn('_smem_bytes_estimate').body[0]),'return super()._smem_bytes_estimate()')
  source=ast.unparse(new)
  self.assertNotIn('sf2_source_base_addr',source);self.assertNotIn('self._sf_expand_fc2_out_of_place(',source)
  self.assertIn('sf2_dest = sfb2_base_addr + fc2_prod_state.index * Int32(2048)',source)
  self.assertIn('_COMPACT_STATIC_TILE_M',source)
  self.assertFalse(any(isinstance(n,ast.Raise) for n in ast.walk(fn('kernel'))))

 def test_fc1_pipeline_lifetime_and_all_non_expansion_barriers_are_preserved(self):
  old=Select(False).visit(fn('kernel',BASE));new=Select(True).visit(fn('kernel'))
  calls=lambda node:[ast.unparse(x) for x in ast.walk(node) if isinstance(x,ast.Call)]
  for suffix in ('consumer_wait','consumer_release','producer_acquire','producer_commit','producer_tail','arrive_and_wait','_resident_grid_barrier'):
   self.assertEqual([x for x in calls(old) if suffix in x.split('(',1)[0]],[x for x in calls(new) if suffix in x.split('(',1)[0]],suffix)
  stage=next(x for x in ast.walk(new) if isinstance(x,ast.For) and ast.unparse(x.target)=='gu' and any(isinstance(y,ast.Name) and y.id=='sf1_packed_addr' for y in ast.walk(x)))
  text=ast.unparse(stage)
  self.assertLess(text.index('fc1_pipeline.consumer_wait'),text.index('_ld_shared_i32_volatile'))
  self.assertLess(text.index('_sf1_load_register_words'),text.index('cute.gemm'))
  self.assertLess(text.index('cute.gemm'),text.index('fc1_pipeline.consumer_release'))
  self.assertEqual(text.count('_sf1_load_register_words'),2)
  # Only the selected FC1 SF6 branch avoids the old expansion call. Generic
  # sf_pack and raw/ineligible branches remain for the existing source contract.
  self.assertEqual(sum('self._sf_expand_stage(' in x for x in calls(old)),sum('self._sf_expand_stage(' in x for x in calls(new))+1)

 def test_exact_selection_and_cache_abi_remain_isolated(self):
  ns={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(TEXT).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
  ns['_EP_TILED_DECODE_OPT']=False
  resolve=run_function('ep_tiled_decode_opt',ns);run_function('ep_tiled_decode_opt_enabled',ns)
  for latched in (False,True):
   ns['_EP_TILED_DECODE_OPT']=latched
   for m in range(1,33):
    for sf6 in (False,True):
     self.assertEqual(resolve(m,sf6,None),latched and sf6 and m<=8)
     self.assertEqual(resolve(m,sf6,True),sf6 and m<=8);self.assertFalse(resolve(m,sf6,False))
  for value in (0,1,'1',[],object()):
   with self.assertRaises(TypeError):resolve(6,True,value)
  self.assertEqual(ns['EP_TILED_DECODE_OPT_CACHE_TAG'],'glm53_ep_static_sf6_fc1_register_v2')
  for name in ('ep_tiled_compile_spec','get_ep_tiled_decode_kernel','warm_ep_tiled_decode'):
   self.assertEqual(dump(fn(name)),dump(fn(name,BASE)))
  setup=fn('_setup_attributes')
  self.assertEqual(ast.unparse(setup.body[0]),'super()._setup_attributes(hidden_size)')
  self.assertEqual(ast.unparse(setup.body[1].test),'self.ep_decode_opt')
  self.assertEqual(ast.unparse(setup.body[1].body[0]),'self._check_ep_sf1_register_layout()')

if __name__=='__main__':unittest.main(verbosity=2)
