"""Pure Q1 bit-flow and ownership tests. No CuTe import/lowering or GPU.

The common conversion stubs model IEEE boundaries for comparison; their results
are not evidence of hardware FP4 conversion. Exact helper/operand reuse is also
checked against the pinned source, so device conversion remains the same call.
"""
import ast,copy,gzip,hashlib,itertools,json,math,os,random,struct,types,unittest
from pathlib import Path
ROOT=Path(os.environ.get('GLM53_EP_Q1_ROOT',Path(__file__).resolve().parents[1]))
SOURCE=Path(os.environ.get('GLM53_EP_Q1_SOURCE',ROOT/'overlay/modules/glm53_moe/moe_static_ep_tiled.py'))
BASE=gzip.decompress((ROOT/'measurements/glm53_ep_tiled_20260909/ep76_onepass3/source/moe_static_ep_tiled.py.gz').read_bytes()).decode()
NEW=SOURCE.read_text()
ORACLE=gzip.decompress((ROOT/'measurements/glm53_ep_local_20260908/micro-stock-oracle/fp4_common.py.gz').read_bytes()).decode()
BASE_SHA='f4435778e3248d0fee3cf373fb95800475c38e2354e59600943574ccff6a4583'
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
class SelectOld(ast.NodeTransformer):
 def visit_If(self,n):
  test=ast.unparse(n.test)
  if test in ('cutlass.const_expr(self.ep_decode_opt)','cutlass.const_expr(self.ep_decode_opt and not self.fast_math)'):
   out=[]
   for x in n.orelse:
    v=self.visit(x);out.extend(v if isinstance(v,list) else [v])
   return out
  return self.generic_visit(n)
class Q1PairTests(unittest.TestCase):
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
 def test_scale_and_converter_formulas_are_the_actual_pinned_expressions(self):
  old=fn('quantize_block_fp4',ORACLE);new=fn('_ep_q1_scale')
  ob=[n for n in old.body if not isinstance(n,ast.Expr)];nb=[n for n in new.body if not isinstance(n,ast.Expr)]
  dump=lambda n:ast.dump(n,include_attributes=False)
  self.assertEqual([dump(n) for n in ob[:5]],[dump(n) for n in nb[:5]])
  oi=next(n for n in ob if isinstance(n,ast.If));ni=next(n for n in nb if isinstance(n,ast.If))
  self.assertEqual(dump(oi.test),dump(ni.test))
  self.assertEqual(dump(oi.body[0].value.args[1]),dump(ni.body[0].value))
  q16=fn('quantize_and_pack_16',ORACLE);q8=fn('_ep_q1_quantize_eight')
  multiply=lambda n:next(x.value for x in ast.walk(n) if isinstance(x,ast.Assign) and isinstance(x.value,ast.BinOp) and isinstance(x.value.op,ast.Mult))
  m16=multiply(q16);m16.left.value.id='values'
  self.assertEqual(dump(m16),dump(multiply(q8)))
  self.assertEqual(ast.unparse(q8.body[-1].value),'cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])')
  ns=ns_base();runfn(fn('quantize_and_pack_16',ORACLE),ns);stock=runfn(old,ns);scale=runfn(new,ns);eight=runfn(q8,ns)
  fast=fn('_ep_q1_scale_fast');stockfast=fn('quantize_block_fp4_fast',ORACLE)
  class RestoreFast(ast.NodeTransformer):
   def visit_Assign(self,n):
    name=ast.unparse(n.targets[0])
    if name=='enabled':return None
    if name=='inv_scale':
     if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='Float32':
      return ast.parse('packed64 = Uint64(0)').body[0]
     return ast.Assign([ast.Name('packed64',ast.Store())],ast.Call(ast.Name('quantize_and_pack_16_fast',ast.Load()),[ast.Name('values',ast.Load()),n.value],[]))
    return self.generic_visit(n)
   def visit_Return(self,n):return ast.parse('return packed64, scale_byte').body[0]
  restored=RestoreFast().visit(fast)
  bodies=lambda n:[ast.dump(x,include_attributes=False) for x in n.body if not isinstance(x,ast.Expr)]
  self.assertEqual(bodies(restored),bodies(stockfast))
  runfn(fn('quantize_and_pack_16_fast',ORACLE),ns);fast_stock=runfn(stockfast,ns);fast_scale=runfn(fn('_ep_q1_scale_fast'),ns)
  rng=random.Random(76)
  vals=[0.,-0.,math.inf,-math.inf,math.nan,1.,-1.,2**-133,3.3895313892515355e38]
  for gs in (0.,-0.,1.,.125,-1.,math.inf,math.nan,1e-38,1e38):
   for i in range(40):
    xs=[F32(rng.choice(vals) if i<20 else rng.uniform(-10,10)) for _ in range(16)]
    maximum=F32(0)
    for v in xs:maximum=mx(maximum,absolute(v))
    oldword,oldscale=stock(xs,maximum,F32(gs));inv,sf,enabled=scale(None,maximum,F32(gs))
    newword=(eight(None,xs[:8],inv)|(eight(None,xs[8:],inv)<<32)) if enabled else 0
    self.assertEqual((newword,sf),(oldword,oldscale))
    oldword,oldscale=fast_stock(xs,maximum,F32(gs));inv,sf,enabled=fast_scale(None,maximum,F32(gs))
    newword=(eight(None,xs[:8],inv)|(eight(None,xs[8:],inv)<<32)) if enabled else 0
    self.assertEqual((newword,sf),(oldword,oldscale))
 def simulate(self,rows,gs,data,paired,fast=False):
  ns=ns_base();writes={};reads=[];coords=[];scale_calls=[]
  class C:
   def __getitem__(self,key):
    row,col,stage=key;assert stage==0 and 0<=row<rows and 0<=col<128
    reads.append((row,col));return data[row][col]
  def idx(crd,outer):
   row,col,stage=crd;coords.append(crd);return row*128+col
  def store(addr,value):
   self.assertNotIn(addr,writes);writes[addr]=value
  ns.update(st_shared_u8=store);ns['cute'].crd2idx=idx
  owner=types.SimpleNamespace(fast_math=fast,decode_reform=True,sf_vec_size=16,tile_m=16,num_mma_warps=4,num_threads_per_warp=32)
  sc=runfn(fn('_ep_q1_scale'),ns);owner._ep_q1_scale=lambda mx,gs:(scale_calls.append((bits(mx),bits(gs))) or sc(owner,mx,gs))
  sc_fast=runfn(fn('_ep_q1_scale_fast'),ns);owner._ep_q1_scale_fast=lambda mx,gs:(scale_calls.append((bits(mx),bits(gs))) or sc_fast(owner,mx,gs))
  q=runfn(fn('_ep_q1_quantize_eight'),ns);owner._ep_q1_quantize_eight=lambda vals,inv:q(owner,vals,inv)
  args=(C(),rows,8,0,None,F32(gs),0,types.SimpleNamespace(outer='synthetic-row-major'),4096)
  if paired:
   class YieldShuffle(ast.NodeTransformer):
    def visit_Call(self,n):
     if ast.unparse(n.func)=='cute.arch.shuffle_sync':return ast.Yield(ast.Tuple(n.args,ast.Load()))
     return self.generic_visit(n)
   f=runfn(YieldShuffle().visit(fn('_ep_q1_pair')),ns)
   generators=[f(owner,*args[:4],tid,*args[5:]) for tid in range(128)]
   requests=[next(g) for g in generators]
   for stage in range(3):
    next_requests=[]
    for tid,g in enumerate(generators):
     value,src=requests[tid];self.assertIn(src,range(32));self.assertEqual(src//2,(tid%32)//2)
     reply=requests[(tid//32)*32+src][0]
     try:next_requests.append(g.send(reply))
     except StopIteration:self.assertEqual(stage,2)
    if stage<2:self.assertEqual(len(next_requests),128)
    requests=next_requests
   self.assertEqual(len(scale_calls),rows*8)
  else:
   runfn(fn('quantize_and_pack_16',ORACLE),ns);runfn(fn('quantize_block_fp4',ORACLE),ns)
   runfn(fn('quantize_and_pack_16_fast',ORACLE),ns);runfn(fn('quantize_block_fp4_fast',ORACLE),ns)
   kernel=SelectOld().visit(fn('kernel',BASE))
   loop=next(n for n in ast.walk(kernel) if isinstance(n,ast.While) and ast.unparse(n.test)=='quant_idx < epi_rows * sf_blocks_per_half')
   names=['self','sC1','epi_rows','sf_blocks_per_half','h','tidx','gs_value','a2_base_addr','a2_smem_layout','sfa2_base_addr']
   init=ast.parse('quant_idx = Int32(tidx)\na2_rows = Int32(self.tile_m)').body
   f=ast.FunctionDef(name='scalar',args=ast.arguments(posonlyargs=[],args=[ast.arg(n) for n in names],kwonlyargs=[],kw_defaults=[],defaults=[]),body=init+[loop],decorator_list=[])
   call=runfn(f,ns)
   for tid in range(128):call(owner,*args[:4],tid,*args[5:])
  return writes,sorted(reads),sorted(coords)
 def test_actual_pair_helper_matches_scalar_ownership_and_bits_rows_zero_to_eight(self):
  rng=random.Random(8)
  for rows in range(9):
   data=[[F32(frombits(rng.randrange(65536)<<16)) for _ in range(128)] for _ in range(rows)]
   for gs,fast in itertools.product((0.,-0.,.125,1.,math.inf,math.nan),(False,True)):
    with self.subTest(rows=rows,gs=gs,fast=fast):
     a=self.simulate(rows,gs,data,False,fast);b=self.simulate(rows,gs,data,True,fast);self.assertEqual(a,b)
     self.assertEqual(len(a[0]),rows*72);self.assertEqual(len(a[1]),rows*128)
 def test_complete_kernel_restore_fallback_and_pipeline_are_unchanged(self):
  a=SelectOld().visit(fn('kernel',BASE));b=SelectOld().visit(fn('kernel'))
  self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
  q=next(n for n in ast.walk(fn('kernel')) if isinstance(n,ast.If) and ast.unparse(n.test)=='cutlass.const_expr(self.ep_decode_opt)' and any(isinstance(x,ast.Call) and ast.unparse(x.func)=='self._ep_q1_pair' for x in ast.walk(n)))
  self.assertEqual(ast.unparse(q.body[0].test),'epi_rows <= Int32(8)')
  self.assertEqual(ast.dump(ast.Module(body=q.body[0].orelse,type_ignores=[]),include_attributes=False),ast.dump(ast.Module(body=q.orelse,type_ignores=[]),include_attributes=False))
  # Even if the new opt branch is selected, the only device-body delta is
  # Q1 helper invocation. The accepted FC1 expansions and FC2 stay identical.
  class RestoreQ1(ast.NodeTransformer):
   def visit_If(self,n):
    t=ast.unparse(n.test)
    if t=='cutlass.const_expr(self.ep_decode_opt)':return n.orelse
    if t=='cutlass.const_expr(self.ep_decode_opt and not self.fast_math)':return n.orelse
    return self.generic_visit(n)
  self.assertEqual(ast.dump(RestoreQ1().visit(fn('kernel')),include_attributes=False),ast.dump(a,include_attributes=False))
  self.assertNotIn('_sf1_load_register_words',NEW);self.assertNotIn('fc1_register_u32',NEW)
  self.assertEqual(ast.dump(fn('_sf_expand_stage'),include_attributes=False),ast.dump(fn('_sf_expand_stage',BASE),include_attributes=False))
 def test_geometry_and_collective_structure_are_fail_closed(self):
  check=runfn(fn('_check_ep_q1_geometry'),{})
  attrs=dict(decode_reform=True,reform_sf_pack=True,tile_m=16,fc1_tile_n=128,sf_vec_size=16,fc1_halves=1,num_mma_warps=4,num_threads_per_warp=32)
  owner=types.SimpleNamespace(**attrs);check(owner);self.assertTrue(owner.ep_q1_pair_geometry_proven)
  for name in attrs:
   bad=attrs.copy();bad[name]=False if isinstance(attrs[name],bool) else attrs[name]+1
   with self.subTest(name=name),self.assertRaises(ValueError):check(types.SimpleNamespace(**bad))
  helper=fn('_ep_q1_pair');shuffles=[]
  for n in helper.body:
   if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='cute.arch.shuffle_sync':shuffles.append(n)
  self.assertEqual(len(shuffles),3)
  self.assertEqual(sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='cute.arch.shuffle_sync' for n in ast.walk(helper)),3)
  self.assertEqual(hashlib.sha256(BASE.encode()).hexdigest(),BASE_SHA)
  self.assertEqual(hashlib.sha256(ORACLE.encode()).hexdigest(),ORACLE_SHA)
 def test_actual_layout_guard_records_physical_mapping_and_rejects_bad_views(self):
  def plain(kind,shape,stride,extent):return types.SimpleNamespace(kind=kind,shape=shape,stride=stride,extent=extent)
  def composed(outer,sw):return types.SimpleNamespace(outer=outer,inner=types.SimpleNamespace(num_bits=sw[0],num_base=sw[1],num_shift=sw[2]),offset=0)
  def args():
   return [composed(plain('sc',(16,128,1),(128,1,0),2048),(0,4,3)),
    composed(plain('a',(16,128,1),(128,1,0),2048),(2,4,3)),
    plain('sf',(128,128,1),('SF16-broadcast',),1024)]
  def idx(crd,l):
   row,col,stage=crd;self.assertEqual(stage,0)
   if l.kind=='symbolic':return object()
   if l.kind=='alias':return 0
   if l.kind=='badscale':return 1
   if l.kind=='sf':return (col//64)*512+(row%32)*16+(row//32)*4+(col//16)%4
   return row*128+col
  cute=types.SimpleNamespace(crd2idx=idx,cosize=lambda l:l.extent)
  guard=runfn(fn('_check_ep_q1_layout'),dict(cute=cute))
  def owner():return types.SimpleNamespace(ep_q1_pair_geometry_proven=True,a_dtype=types.SimpleNamespace(width=4),sf_dtype=types.SimpleNamespace(width=8),buffer_align_bytes=1024,fast_math=True)
  ob=owner();guard(ob,*args());r=ob.ep_q1_pair_layout_receipt
  self.assertTrue(ob.ep_q1_pair_layout_proven);self.assertEqual(r['math_mode'],'fast');self.assertTrue(r['selected'])
  self.assertEqual([x['rows'] for x in r['rows']],list(range(9)))
  for row in r['rows']:
   n=row['rows'];self.assertEqual((row['sc1_bytes'],row['a2_bytes'],row['sfa2_bytes']),(256*n,64*n,8*n))
   for key in ('source_sha256','packed_sha256','scales_sha256','ownership_sha256'):self.assertRegex(row[key],r'^[0-9a-f]{64}$')
  again=owner();guard(again,*args());self.assertEqual(r,again.ep_q1_pair_layout_receipt)
  for kind in ('alias','symbolic','swizzle','scale','range','alignment','dtype'):
   ob=owner();sc,a,sf=args()
   if kind in ('alias','symbolic'):sc.outer.kind=kind
   if kind=='swizzle':a.inner.num_bits=3
   if kind=='scale':sf.kind='badscale'
   if kind=='range':a.outer.extent=1
   if kind=='alignment':ob.buffer_align_bytes=128
   if kind=='dtype':ob.sf_dtype.width=16
   with self.subTest(kind=kind),self.assertRaises(ValueError):guard(ob,sc,a,sf)
if __name__=='__main__':unittest.main(verbosity=2)
