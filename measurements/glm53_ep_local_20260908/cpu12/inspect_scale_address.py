import collections,gzip,hashlib,json,pathlib,re
root=pathlib.Path(__file__).resolve().parents[3]
reg_re=re.compile(r'%[A-Za-z]+\d+')
instruction_re=re.compile(r'^\s*([a-z][a-z0-9_.]*)\s+(%[A-Za-z]+\d+),\s*(.+);\s*$')

def load(revision):
 d=root/f'measurements/glm53_ep_local_20260908/cpu{revision}/local'
 receipt=json.loads((d/'result.json').read_text());a=receipt['artifacts'][0]
 raw=gzip.decompress((d/(a['file']+'.gz')).read_bytes())
 assert len(raw)==a['bytes'] and hashlib.sha256(raw).hexdigest()==a['sha256']
 return receipt,raw.decode().splitlines()

def inspect(revision):
 receipt,lines=load(revision)
 definitions=collections.defaultdict(list)
 for index,line in enumerate(lines):
  match=instruction_re.match(line)
  if match:
   op,dest,tail=match.groups()
   definitions[dest].append((index,op,tail))
 stores=[i for i,l in enumerate(lines) if re.search(r'\bst\.global\.b8\b',l)]
 assert len(stores)==10
 results=[]
 for number,store in enumerate(stores):
  m=instruction_re.match(lines[store-2]);assert m and m.group(1)=='cvt.s64.s32'
  final_offset=m.group(3)
  sf_part='%r36' if number<7 else '%r38'
  selected={};leaves={sf_part:'sf_offset'}
  def trace(register,before):
   if register in leaves:return leaves[register]
   defs=[d for d in definitions[register] if d[0]<before]
   assert defs,register
   index,op,tail=defs[-1]
   if op=='ld.shared.s32':
    leaves[register]='physical_row';return 'physical_row'
   assert op.split('.')[0] in ('shr','shl','and','or','add','sub','mul','setp','selp','cvt'),(index,op)
   selected[index]=dict(line=index+1,op=op,text=lines[index].strip())
   args=[]
   for operand in tail.split(','):
    operand=operand.strip()
    args.append(trace(operand,index) if reg_re.fullmatch(operand) else operand)
   # Compare the opcode/data dependency graph without register identities
   # or instruction ordering. Integer commutative operations are normalized.
   if op.split('.')[0] in ('and','or','add','mul'):
    args.sort(key=repr)
   return (op,*args)
  graph=trace(final_offset,store-2)
  assert list(leaves.values()).count('physical_row')==1
  physical_row=next(k for k,v in leaves.items() if v=='physical_row')
  row_load=[d for d in definitions[physical_row] if d[0]<store][-1][0]
  indexlist=sorted(selected)
  ranges=[]
  for i in indexlist:
   if ranges and ranges[-1][-1]+1==i+1:ranges[-1][-1]=i+1
   else:ranges.append([i+1,i+1])
  arith=[selected[i] for i in indexlist]
  count=41 if revision==11 else 9
  assert len(arith)==count,(number,len(arith))
  results.append(dict(site=number+1,path='equal' if number<7 else 'varied',scale_store_line=store+1,physical_row_register=physical_row,physical_row_load_line=row_load+1,sf_offset_register=sf_part,arithmetic_count=len(arith),arithmetic_ranges=ranges,instructions=arith,dependency_graph=graph,op_counts=dict(collections.Counter(i['op'] for i in arith))))
 assert all(r['dependency_graph']==results[0]['dependency_graph'] for r in results),[r['site'] for r in results]
 return dict(revision=revision,ptx=receipt['artifacts'][0],cubin=receipt['resources'][0],all_ten_sites_same_dependency_graph=True,sites=results)

before=inspect(11);after=inspect(12)
report=dict(scope='Receipt-bound Q0 scale-address dependency graphs only; no GPU execution or throughput measurement',before=before,after=after,summary=dict(sites=10,equal_sites=7,varied_sites=3,per_site_arithmetic_before=41,per_site_arithmetic_after=9,per_site_removed=32,static_arithmetic_before=410,static_arithmetic_after=90,static_removed=320),limitations=['PTX counts are static source instructions, not SASS instructions, executed instruction counts, memory transactions, or speedup.','Ten sites are compiler unroll/tail clones: seven equal-scale stores and three varied-scale stores, not ten mandatory operations per route.','Shared input load, payload store, SF-column setup, pointer widening and final global scale store are excluded from the arithmetic count.','The assembler resource totals remain REG168 STACK112 SHARED1024; generated code size changed but runtime occupancy/performance is unmeasured.'])
p=pathlib.Path('/tmp/glm53-cpu12-scale-address-inspection.json');p.write_text(json.dumps(report,indent=2)+'\n')
rows=[]
for b,a in zip(before['sites'],after['sites']):
 fmt=lambda r:', '.join(str(lo) if lo==hi else f'{lo}–{hi}' for lo,hi in r)
 rows.append(f"| {b['site']} | {b['path']} | {fmt(b['arithmetic_ranges'])} | {fmt(a['arithmetic_ranges'])} | 41 → 9 | {b['scale_store_line']} → {a['scale_store_line']} |")
representatives=[]
for n in (0,7):
 a=after['sites'][n]
 representatives.append(f"CPU12 {a['path']} representative, arithmetic lines {a['arithmetic_ranges']}:\n\n```ptx\n"+'\n'.join(i['text'] for i in a['instructions'])+'\n```')
md=f'''# CPU12 Q0 scale-address PTX inspection

The row-dependent Q0 scale address compiles to **41 → 9 PTX arithmetic instructions per static store site**, a reduction of 32. All **10** compiler unroll/tail sites match the same dependency graph within each build: seven equal-scale and three varied-scale sites. The total across these static copies is **410 → 90**, or **320 removed**.

Both decompressed PTX files match their compilation receipts:

- CPU11: `{before['ptx']['sha256']}` ({before['ptx']['bytes']} bytes).
- CPU12: `{after['ptx']['sha256']}` ({after['ptx']['bytes']} bytes).

The inspection traces each scale store's Int32 byte offset back to exactly one shared physical-row load and its precomputed SF-column offset. It excludes those input operations, payload stores, address widening, and the final byte store. Register names and independent instruction order are normalized when checking all ten dependency graphs; opcodes and literal constants remain exact.

| Site | Path | CPU11 arithmetic lines | CPU12 arithmetic lines | Count | Scale store line |
|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'\n\n'+'\n\n'.join(representatives)+'''

For H4096/M128, CPU12 computes the physical-tile field as `(physical_row << 8) & 0xffff8000`, outer-row field as `(physical_row << 4) & 496`, and inner-row field as `(physical_row >> 3) & 12`. It combines these disjoint fields with the precomputed SF-column offset. The old signed quotient/remainder correction chains disappear from this dependency graph.

**PTX counts are not SASS, executed work, GPU timing, or TTFT results.** Ten sites count compiler copies, not ten mandatory operations for each routed row. The exact runtime saving depends on selected routes and branches. Assembler totals remain **REG168 / STACK112 / SHARED1024**; CuTe PTX shrank 955476 → 942077 bytes and cubin 307264 → 300864 bytes. No runtime speedup or occupancy change is established.
'''
pathlib.Path('/tmp/glm53-cpu12-scale-address-inspection.md').write_text(md)
print(json.dumps(report['summary']))
print(str(p))
