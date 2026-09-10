"""Pure register-max Q1 bit-flow and ownership tests. No CuTe import/lowering or GPU.

The common conversion stubs model IEEE boundaries for comparison; their results
are not evidence of hardware FP4 conversion. Exact helper/operand reuse is also
checked against the pinned source, so device conversion remains the same call.
"""
import ast,copy,gzip,hashlib,itertools,json,math,os,random,struct,types,unittest
from pathlib import Path
ROOT=Path(os.environ.get('GLM53_EP_Q1_ROOT',Path(__file__).resolve().parents[1]))
SOURCE=Path(os.environ.get('GLM53_EP_Q1_SOURCE',ROOT/'overlay/modules/glm53_moe/moe_static_ep_tiled.py'))
BASE=gzip.decompress((ROOT/'measurements/glm53_ep_tiled_20260909/ep76_onepass4/source/moe_static_ep_tiled.py.gz').read_bytes()).decode()
NEW=SOURCE.read_text()
ORACLE=gzip.decompress((ROOT/'measurements/glm53_ep_local_20260908/micro-stock-oracle/fp4_common.py.gz').read_bytes()).decode()
BASE_SHA='7c610fa7659e85be4201907c08125e35f818d130113cbacfe26828c377649074'
ORACLE_SHA='a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1'
def bits(x):return struct.unpack('<I',struct.pack('<f',float(x)))[0]
def frombits(x):return struct.unpack('<f',struct.pack('<I',x))[0]
class F32(float):
 def __new__(cls,x=0):
  try:x=struct.unpack('<f',struct.pack('<f',float(x)))[0]
  except OverflowError:x=math.copysign(math.inf,x)
  return float.__new__(cls,x)
 def __mul__(self,x):return F32(float(self)*float(x))
 def __rmul__(self,x):return F32(float(x)*float(self))
 def __truediv__(self,x):
  x=float(x)
  if x==0:
   return F32(math.nan if float(self)==0 or math.isnan(self) else math.copysign(math.inf,math.copysign(1,self)*math.copysign(1,x)))
  return F32(float(self)/x)
def u8(v):return int(v)&255
def u32(v):return int(v)&0xffffffff
def u64(v):return int(v)&0xffffffffffffffff
def mx(a,b):
 if math.isnan(a):return F32(b)
 if math.isnan(b):return F32(a)
 return F32(max(a,b))
def mn(a,b):
 if math.isnan(a):return F32(b)
 if math.isnan(b):return F32(a)
 return F32(min(a,b))
def absolute(a):return F32(frombits(bits(a)&0x7fffffff))
FP8=[m/512 if e==0 else (1+m/8)*2**(e-7) for e in range(16) for m in range(8)][:-1]
def cvt8(x):
 if math.isnan(x):return 127
 sign=bits(x)>>31;v=min(abs(float(x)),448)
 q=min(range(127),key=lambda i:(abs(FP8[i]-v),i&1))
 return q|(sign<<7)
def from8(x):
 x=int(x)&255
 if x&127==127:return F32(math.nan)
 return F32(math.copysign(FP8[x&127],-1 if x&128 else 1))
def recip(x):
 if bits(x)&0x7f800000==0:x=F32(math.copysign(0,x))
 return F32(1)/F32(x)
def from8rcp(x):
 x=from8(x);return F32(0) if x==0 else recip(x)
def cvt4x8(*xs):
 table=(0,.5,1,1.5,2,3,4,6);word=0
 for i,x in enumerate(xs):
  q=7 if math.isnan(x) else min(range(8),key=lambda q:(abs(table[q]-min(abs(float(x)),6)),q&1))
  word|=(q|((bits(x)>>31)<<3))<<(4*i)
 return word
def fn(name,source=NEW):return copy.deepcopy(next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name))
def runfn(n,ns):
 n=copy.deepcopy(n);n.decorator_list=[]
 module=ast.Module(body=[ast.ImportFrom('__future__',[ast.alias('annotations')],0),n],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(ROOT/n.name),'exec'),ns);return ns[n.name]
def shape_n(shape):return math.prod(shape)
def ns_base():return dict(Float32=F32,Uint8=u8,Uint32=u32,Uint64=u64,Int32=int,FLOAT4_E2M1_MAX=6.,FLOAT8_E4M3_MAX=448.,fmax_f32=mx,fmin_f32=mn,fabs_f32=absolute,cvt_f32_to_e4m3=cvt8,fp8_e4m3_to_f32=from8,cvt_e2m1x8_f32=cvt4x8,rcp_approx_ftz=recip,fp8_e4m3_to_f32_and_rcp=from8rcp,cutlass=types.SimpleNamespace(Float32=F32,range_constexpr=range,const_expr=lambda b:b),cute=types.SimpleNamespace(make_rmem_tensor=lambda shape,dtype:[F32(0)]*shape_n(shape)))
class OldOnly(ast.NodeTransformer):
 def visit_If(self,n):
  if ast.unparse(n.test)=='cutlass.const_expr(self.ep_decode_opt)':
   result=[]
   for x in n.orelse:
    x=self.visit(x);result.extend(x if isinstance(x,list) else [x])
   return result
  return self.generic_visit(n)
def coord(tid,i):
 lane,warp=tid%32,tid//32;k,rem=divmod(i,4);j,e=divmod(rem,2)
 return lane//4+8*j,2*(lane%4)+16*(warp%2)+8*(warp//2)+32*k+e,0
class RegisterMaxTests(unittest.TestCase):
 def test_all_bf16_patterns_and_nonfinite_reduction_bits(self):
  def reduce(xs):
   m=F32(0)
   for x in xs:m=mx(m,absolute(x))
   return m
  for raw in range(65536):
   v=F32(frombits(raw<<16));xs=[F32(0)]*16;xs[raw%16]=v
   xs[(raw+7)%16]=F32(frombits(0x7fc10000))
   self.assertEqual(bits(reduce(xs)),bits(mx(reduce(xs[:8]),reduce(xs[8:]))))
  for triple in itertools.product((0.,-0.,math.nan,-math.nan,math.inf,-math.inf,1.,-1.,2**-133),repeat=3):
   xs=list(map(F32,triple))*5+[F32(math.nan)]
   self.assertEqual(bits(reduce(xs)),bits(mx(reduce(xs[:8]),reduce(xs[8:]))))
 def test_scale_helpers_and_x2_instruction_are_pinned_exactly(self):
  self.assertEqual(hashlib.sha256(BASE.encode()).hexdigest(),BASE_SHA)
  for name in ('_ep_q1_scale','_ep_q1_scale_fast'):
   self.assertEqual(ast.dump(fn(name),include_attributes=False),ast.dump(fn(name,BASE),include_attributes=False))
  self.assertEqual(hashlib.sha256(ORACLE.encode()).hexdigest(),ORACLE_SHA)
  op=fn('_ep_q1_cvt_pair');call=next(n for n in ast.walk(op) if isinstance(n,ast.Call) and ast.unparse(n.func)=='llvm.inline_asm')
  asm=ast.literal_eval(call.args[2]);pinned=fn('cvt_e2m1x8_f32',ORACLE)
  pinasm=ast.literal_eval(next(n for n in ast.walk(pinned) if isinstance(n,ast.Call) and ast.unparse(n.func)=='llvm.inline_asm').args[2])
  self.assertIn('cvt.rn.satfinite.e2m1x2.f32 pair, $2, $1;',asm)
  self.assertIn('cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;',pinasm)
  self.assertIn('cvt.u32.u8 $0, pair;',asm)
  load=next(n for n in ast.walk(fn('_ep_q1_ld_peer_max')) if isinstance(n,ast.Call) and ast.unparse(n.func)=='llvm.inline_asm')
  self.assertEqual(ast.literal_eval(load.args[2]),'ld.volatile.shared.f32 $0, [$1];')
  self.assertIs(ast.literal_eval(next(k.value for k in load.keywords if k.arg=='has_side_effects')),True)
  self.assertNotIn('_ld_shared_f32',ast.unparse(fn('_ep_q1_register_quantize')))
  self.assertEqual(ast.unparse(call.args[1]),'[Float32(v0).ir_value(loc=loc, ip=ip), Float32(v1).ir_value(loc=loc, ip=ip)]')
 def simulate(self,r,gs,data,fast,storage=None):
  ns=ns_base();scratch=storage if storage is not None else {};written_max=set();writes={};readmax=[];readregs=[];scale_calls=[]
  def idx(crd,outer):row,col,stage=crd;assert stage==0;return row*128+col
  def store(addr,v):self.assertNotIn(addr,writes);writes[addr]=v
  def maxstore(addr,v):self.assertNotIn(addr,written_max);written_max.add(addr);self.assertTrue(8192<=addr<8704 and addr%4==0);scratch[addr]=v
  def maxload(addr):self.assertIn(addr,scratch);readmax.append(addr);return scratch[addr]
  ns.update(_st_shared_f32=maxstore,_ep_q1_ld_peer_max=maxload,st_shared_u8=store,_ep_q1_cvt_pair=lambda a,b:cvt4x8(a,b,*([F32(0)]*6))&255)
  ns['cute'].crd2idx=idx
  owner=types.SimpleNamespace(fast_math=fast)
  for name in ('_ep_q1_scale','_ep_q1_scale_fast'):
   call=runfn(fn(name),ns)
   setattr(owner,name,lambda m,g,call=call:(scale_calls.append((bits(m),bits(g))) or call(owner,m,g)))
  class Registers:
   def __init__(self,tid):self.tid=tid
   def __getitem__(self,i):
    row,col,stage=coord(self.tid,i);self_outer.assertTrue(0<=row<r and 0<=col<128 and i%4<2)
    readregs.append((self.tid,i,row,col));return data[row][col]
  self_outer=self;regs=[Registers(t) for t in range(128)];maxima=[[F32(0)]*4 for _ in range(128)]
  class Shuffles(ast.NodeTransformer):
   def visit_Call(self,n):
    if ast.unparse(n.func)=='cute.arch.shuffle_sync':return ast.Yield(ast.Tuple(n.args,ast.Load()))
    return self.generic_visit(n)
  def phase(name,args):
   f=runfn(Shuffles().visit(fn(name)),ns);gen=[f(owner,*args(t)) for t in range(128)];req=[next(g) for g in gen]
   for stage in range(8):
    following=[]
    for t,g in enumerate(gen):
     value,peer=req[t];self.assertIn(peer,range(32));self.assertEqual(peer//4,(t%32)//4)
     answer=req[t//32*32+peer][0]
     try:following.append(g.send(answer))
     except StopIteration:self.assertEqual(stage,7)
    if stage<7:self.assertEqual(len(following),128)
    req=following
  # This phased execution models the existing CTA barrier; loads cannot run
  # until every producer has returned. Every subgroup shuffle participates.
  if r:phase('_ep_q1_register_max',lambda t:(regs[t],maxima[t],r,t,8192))
  self.assertEqual(len(written_max),r*16)
  phase('_ep_q1_register_quantize',lambda t:(regs[t],maxima[t],r,t,F32(gs),8192,0,types.SimpleNamespace(outer='rowmajor'),4096))
  self.assertEqual(len(readmax),r*16);self.assertEqual(sorted(readmax),sorted(written_max));self.assertEqual(len(scale_calls),r*16)
  self.assertEqual(len(writes),r*72)
  expected={};runfn(fn('quantize_and_pack_16',ORACLE),ns);runfn(fn('quantize_and_pack_16_fast',ORACLE),ns)
  quant=runfn(fn('quantize_block_fp4_fast' if fast else 'quantize_block_fp4',ORACLE),ns)
  for row in range(r):
   for block in range(8):
    vals=data[row][block*16:(block+1)*16];maximum=F32(0)
    for v in vals:maximum=mx(maximum,absolute(v))
    word,sf=quant(vals,maximum,F32(gs))
    for i in range(8):
     b=row*64+block*8+i;b^=(b>>3)&0x30;expected[b]=(word>>(8*i))&255
    expected[4096+(block//4)*512+row*16+block%4]=sf
  self.assertEqual(writes,expected)
 def test_actual_two_phases_match_scalar_bits_ownership_and_scratch(self):
  rng=random.Random(76)
  for r in range(9):
   data=[[F32(frombits(rng.randrange(65536)<<16)) for _ in range(128)] for _ in range(r)]
   for gs,fast in itertools.product((0.,-0.,.125,1.,math.inf,math.nan),(False,True)):
    with self.subTest(r=r,gs=gs,fast=fast):self.simulate(r,gs,data,fast)
  # Preserve the same scratch dictionary/addresses over changing persistent
  # items, including inactive rows and shrinking/expanding valid ranges.
  storage={}
  for item,r in enumerate((8,1,0,8)):
   data=[[F32((item+1)*(col+1)/32) for col in range(128)] for _ in range(r)]
   self.simulate(r,.125,data,True,storage)
 def test_complete_accepted_body_fallback_and_all_barriers_remain(self):
  old=OldOnly().visit(fn('kernel',BASE));new=OldOnly().visit(fn('kernel'))
  self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
  barriers=lambda n:[ast.unparse(x) for x in ast.walk(n) if isinstance(x,ast.Call) and ('barrier.arrive_and_wait' in ast.unparse(x.func) or ast.unparse(x.func)=='cute.arch.fence_proxy')]
  self.assertEqual(barriers(fn('kernel')),barriers(fn('kernel',BASE)))
  for method in ('_sf_expand_stage','_ep_q1_scale','_ep_q1_scale_fast'):
   self.assertEqual(ast.dump(fn(method),include_attributes=False),ast.dump(fn(method,BASE),include_attributes=False))
  body=fn('kernel');self.assertNotIn('_ep_q1_pair',ast.unparse(body))
  branches=[n for n in ast.walk(body) if isinstance(n,ast.If) and ast.unparse(n.test) in ('epi_rows <= Int32(8)','epi_m_valid <= Int32(8)')]
  self.assertEqual(len(branches),2)
  scalar=next(n for n in ast.walk(old) if isinstance(n,ast.While) and ast.unparse(n.test)=='quant_idx < epi_rows * sf_blocks_per_half')
  quant=next(n for n in branches if ast.unparse(n.test)=='epi_rows <= Int32(8)')
  self.assertEqual(ast.dump(quant.orelse[1],include_attributes=False),ast.dump(scalar,include_attributes=False))
  text=ast.unparse(body);self.assertLess(text.index('self._ep_q1_register_max('),text.index('self._ep_q1_register_quantize('))
 def test_actual_host_guard_mocks_bind_fragment_and_reject_mismatch(self):
  state={'mode':'valid'}
  class Tensor:
   shape=(4,1,4)
   def __init__(self,t):self.t=t
   def __getitem__(self,i):
    if isinstance(i,tuple):return self
    v=coord(self.t,i)
    if state['mode']=='ownership' and self.t==0 and i==0:return (0,8,0)
    if state['mode']=='symbolic':return (object(),0,0)
    return v
  class Thread:
   def __init__(self,t):self.t=t
   def partition_D(self,x):return Tensor(self.t)
   def partition_S(self,x):return types.SimpleNamespace(shape=(4,1,4,1))
  copier=types.SimpleNamespace(get_slice=lambda t:Thread(t))
  def layout(kind,shape,stride,extent):return types.SimpleNamespace(kind=kind,shape=shape,stride=stride,extent=extent)
  def comp(outer,sw):return types.SimpleNamespace(outer=outer,inner=types.SimpleNamespace(num_bits=sw[0],num_base=sw[1],num_shift=sw[2]),offset=0)
  def args():return [comp(layout('sc',((8,2),(64,2),(1,1)),((64,512),(1,1024),(0,0)),2048),(3,4,3)),comp(layout('a',(16,128,1),(128,1,0),2048),(2,4,3)),layout('sf',(16,128,1),(0,1,0),1024)]
  def idx(c,l):
   if l.kind=='dense':return c if state['mode']!='rmem' else c+1
   r,col,stage=c
   if l.kind=='sc':return (r%8)*64+(r//8)*512+col%64+(col//64)*1024
   if l.kind=='sf':return (col//64)*512+r*16+(col//16)%4
   return (0 if state['mode']=='a_alias' else r)*128+col
  cute=types.SimpleNamespace(make_copy_atom=lambda *a:object(),make_tiled_copy_S=lambda *a:copier,make_tiled_copy_C_atom=lambda *a:object(),
   make_identity_tensor=lambda s:object(),make_layout=lambda s:layout('dense',s,(),16),shape=lambda t:t.shape,size=lambda t:16,
   crd2idx=idx,cosize=lambda l:l.extent,nvgpu=types.SimpleNamespace(CopyUniversalOp=lambda:object(),warp=types.SimpleNamespace(StMatrix8x8x16bOp=lambda *a:object())))
  guard=runfn(fn('_check_ep_q1_register_layout'),dict(cute=cute,cutlass=types.SimpleNamespace(BFloat16=object())))
  owner=lambda:types.SimpleNamespace(ep_q1_register_geometry_proven=True,a_dtype=types.SimpleNamespace(width=4),sf_dtype=types.SimpleNamespace(width=8),buffer_align_bytes=1024,fast_math=True,c_layout=types.SimpleNamespace(is_m_major_c=lambda:False))
  ob=owner();guard(ob,object(),*args());receipt=ob.ep_q1_register_layout_receipt
  self.assertTrue(receipt['proven']);self.assertEqual(receipt['scratch_bytes'],512)
  for r,row in enumerate(receipt['rows']):self.assertEqual((row['max_store_bytes'],row['max_load_bytes'],row['a2_bytes'],row['sfa2_bytes']),(64*r,64*r,64*r,8*r))
  for mode in ('ownership','symbolic','rmem','a_alias'):
   state['mode']=mode
   with self.subTest(mode=mode),self.assertRaises(ValueError):guard(owner(),object(),*args())
  state['mode']='valid';sc,a,sf=args();sc.outer.extent=255
  with self.assertRaises(ValueError):guard(owner(),object(),sc,a,sf)
  sc,a,sf=args();sf.extent=1
  with self.assertRaises(ValueError):guard(owner(),object(),sc,a,sf)
 def test_geometry_selection_and_exact_scratch_budget(self):
  check=runfn(fn('_check_ep_q1_geometry'),{})
  attrs=dict(decode_reform=True,reform_sf_pack=True,tile_m=16,fc1_tile_n=128,sf_vec_size=16,fc1_halves=1,num_mma_warps=4,num_threads_per_warp=32)
  ob=types.SimpleNamespace(**attrs);check(ob);self.assertTrue(ob.ep_q1_register_geometry_proven)
  for k in attrs:
   changed=attrs|{k:False if isinstance(attrs[k],bool) else attrs[k]+1}
   with self.subTest(k=k),self.assertRaises(ValueError):check(types.SimpleNamespace(**changed))
  self.assertIn('glm53_ep_static_sf6_q1_register_max_v5',NEW)
  for r in range(1,9):self.assertEqual((4096+256*r)-(64*r+64*r),4096+128*r)
if __name__=='__main__':unittest.main(verbosity=2)
