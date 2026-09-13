"""Execute actual Q0 prefix/reservation statements with cooperative CPU lanes."""
import ast
import builtins
from collections import Counter
import copy
import importlib.util
from pathlib import Path
import random
import struct
import symtable
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
BODY=ROOT/'engine/kernels/b12x/_prefill_q0_batch8.py'


class UInt(int):
    def bitcast(self, dtype):
        return struct.unpack('f',struct.pack('I',self))[0]


class Scalar:
    def __init__(self,value): self.value=value
    def to(self,dtype): return dtype(self.value)


class Array(list):
    def __getitem__(self,index): return Scalar(super().__getitem__(index))


def statements():
    return next(n for n in ast.walk(ast.parse(BODY.read_text()))
                if isinstance(n,ast.FunctionDef) and n.name=='initialize_route_q0_and_publish').body


class Cooperative(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        name=ast.unparse(node.func)
        ops={'cute.arch.shuffle_sync':'shuffle','cute.arch.sync_threads':'block','cute.arch.sync_warp':'warp'}
        if name in ops:
            return ast.copy_location(ast.Yield(ast.Tuple(
                [ast.Constant(ops[name]),ast.Constant(node.lineno),*node.args],ast.Load())),node)
        return node


def execute_lanes(nodes, environments):
    fn=ast.FunctionDef(name='run',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
                      body=copy.deepcopy(nodes),decorator_list=[])
    module=ast.fix_missing_locations(Cooperative().visit(ast.Module(body=[fn],type_ignores=[])))
    code=compile(module,'<actual-Q0-statements>','exec')
    coroutines=[]
    for env in environments:
        exec(code,env)
        coroutines.append(env['run']())
    waits={}
    def resume(index,value=None):
        try: waits[index]=coroutines[index].send(value)
        except StopIteration: waits.pop(index,None)
    order=list(range(len(coroutines)));random.Random(87).shuffle(order)
    for i in order: resume(i)
    while waits:
        progress=False
        for lo in range(0,len(coroutines),32):
            ids=list(range(lo,lo+32))
            if not all(i in waits for i in ids): continue
            events=[waits[i] for i in ids]
            if len({e[:2] for e in events})==1 and events[0][0] in ('shuffle','warp'):
                values=([events[e[3]&31][2] for e in events] if events[0][0]=='shuffle' else [None]*32)
                for i,value in zip(ids,values): resume(i,value)
                progress=True
        if len(waits)==len(coroutines) and len({e[:2] for e in waits.values()})==1 and next(iter(waits.values()))[0]=='block':
            for i in list(waits): resume(i)
            progress=True
        if not progress: raise AssertionError('divergent barrier or warp collective')


def environment(shared):
    def store(address,value): shared[address]=value
    return dict(Int32=int,Int64=int,Uint32=UInt,cutlass=NS(Float32=float,range_constexpr=range),
                _ld_shared_i32=shared.__getitem__,ld_shared_i32_relaxed=shared.__getitem__,
                _st_shared_i32=store,st_shared_i32=store)


class Q0Batch8Tests(unittest.TestCase):
    def test_actual_opt_in_guard_excludes_decode_long_context_and_capture(self):
        tree=ast.parse((ROOT/'engine/kernels/b12x/moe_dispatch.py').read_text())
        compiler=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_get_dynamic_kernel')
        guards=[n for n in compiler.body if isinstance(n,ast.If)
                and ast.unparse(n.test).startswith(('type(_prefill_q0_batch8)', '_prefill_q0_batch8 and'))]
        code=compile(ast.Module(body=guards,type_ignores=[]),'<actual-Q0-compiler-guard>','exec')
        valid=dict(_prefill_q0_batch8=True,tp_sf6_q0=True,_prefill_tile64=False,m=2672)
        for rows in (65,2672,8192): exec(code,dict(valid,m=rows))
        for change in ({'m':64},{'m':8193},{'tp_sf6_q0':False},{'_prefill_tile64':True}):
            with self.assertRaises(ValueError): exec(code,dict(valid,**change))
        with self.assertRaises(TypeError): exec(code,dict(valid,_prefill_q0_batch8=1))
        launch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='launch_sm120_dynamic_moe')
        branch=next(n for n in launch.body if isinstance(n,ast.If) and ast.unparse(n.test)=='_prefill_q0_batch8')
        code=compile(ast.Module(body=[branch],type_ignores=[]),'<actual-Q0-launch-guard>','exec')
        env=dict(_prefill_q0_batch8=True,_prefill_tile64=None,_prefill_scale_expansion=None,
                 num_tokens=2672,workspace=NS(tile_m=128),
                 torch=NS(cuda=NS(is_current_stream_capturing=lambda:False)))
        exec(code,env)
        self.assertIs(env['_prefill_tile64'],False)
        self.assertIs(env['_prefill_scale_expansion'],False)
        env['torch']=NS(cuda=NS(is_current_stream_capturing=lambda:True))
        with self.assertRaises(RuntimeError): exec(code,env)
        for node in (compiler,launch):
            defaults=dict(zip((a.arg for a in node.args.kwonlyargs),node.args.kw_defaults))
            self.assertIs(ast.literal_eval(defaults['_prefill_q0_batch8']),False)

    def test_actual_copy_sizes_cover_ragged_requests_and_avoid_route_metadata(self):
        loop=next(n for n in statements() if isinstance(n,ast.While)
                  and ast.unparse(n.test)=='produce_active > Int32(0)')
        branch=next(n for n in loop.body if isinstance(n,ast.If)
                    and ast.unparse(n.test)=='batch_base >= producer_limit')
        stop=next(i for i,n in enumerate(branch.orelse) if isinstance(n,ast.If)
                  and ast.unparse(n.test)=='warp_idx == Int32(self.num_mma_warps)')
        sizes=compile(ast.Module(body=branch.orelse[:stop],type_ignores=[]),'<actual-Q0-copy-sizes>','exec')
        remap=[n for n in statements() if isinstance(n,ast.Assign) and
               isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('route_phys_rows_addr','route_expert_ids_addr')]
        aliases=compile(ast.Module(body=remap,type_ignores=[]),'<actual-Q0-aliases>','exec')
        for tokens in (65,67,2672,2675,8192):
            covered=[]
            for start in range(0,tokens,8):
                env=dict(Int32=int,num_tokens=tokens,batch_base=start,producer_batch_tokens=8,cols=4096)
                exec(sizes,env)
                first,second=env['first_copy_bytes'],env['second_copy_bytes']
                self.assertTrue(0<first<=32768 and 0<=second<=32768)
                self.assertEqual(first+second,min(8,tokens-start)*8192)
                self.assertLessEqual(start*8192+first+second,tokens*8192)
                env.update(self=NS(q0_route_shift=16384),q0_input_stage_base_addr=16384)
                exec(aliases,env)
                self.assertEqual(env['route_phys_rows_addr'],0)
                self.assertEqual(env['route_expert_ids_addr'],1152)
                self.assertLessEqual(env['route_phys_rows_addr']+3*288*4,16384)
                self.assertLessEqual(32768+second,32768+40960)
                covered.extend(range(start,start+(first+second)//8192))
            self.assertEqual(covered,list(range(tokens)))

    def test_static_body_matches_pinned_generator_and_has_no_missing_jit_globals(self):
        path=ROOT/'measurements/st_prefill_oracle_20260913/generate_q0_batch8.py'
        spec=importlib.util.spec_from_file_location('_q0_generator',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(BODY.read_text(),module.build())
        table=symtable.symtable(BODY.read_text(),str(BODY),'exec')
        available=set(table.get_identifiers())|set(dir(builtins))
        pending=[table];missing=[]
        while pending:
            node=pending.pop();pending.extend(node.get_children())
            missing.extend(x.get_name() for x in node.get_symbols()
                           if x.is_global() and x.is_referenced() and x.get_name() not in available)
        self.assertEqual(missing,[])

    def test_actual_nine_warp_prefix_handles_empty_tail_and_many_tiles(self):
        prefix=next(n for n in statements() if isinstance(n,ast.If)
                    and ast.unparse(n.test)=='num_experts == Int32(288) and bidz == Int32(0)')
        for counts in ([0]*288,[1]*288,[(i*173)%4097 for i in range(288)], [0]*287+[65536]):
            shared={};bases=[-1]*289
            envs=[dict(environment(shared),tidx=i,warp_idx=i//32,row_counts=counts,
                       expert_tile_base=bases,num_experts=288,ctrl_base_addr=-64,route_hist_addr=2304)
                  for i in range(288)]
            execute_lanes(prefix.body,envs)
            expected=[0]
            for count in counts: expected.append(expected[-1]+(count+127)//128)
            self.assertEqual(bases,expected)

    def test_actual_parallel_reservations_preserve_all_routes_and_scale_coordinates(self):
        producer=next(n for n in ast.walk(ast.Module(body=statements(),type_ignores=[]))
                      if isinstance(n,ast.If) and ast.unparse(n.test)=='warp_idx < producer_batch_tokens and token_idx < num_tokens')
        end=next(i for i,n in enumerate(producer.body) if isinstance(n,ast.Expr)
                 and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='cute.arch.sync_warp')
        reservation=producer.body[:end+1]
        for equal in (False,True):
            ids=[[((t*13+j)%288 if t%3 else 287) for j in range(8)] for t in range(67)]
            weights=[(-0. if j==0 else (-.125 if j==1 else .25)) for t in ids for j in range(8)]
            counts=Counter(e for row in ids for e in row);bases=[0]
            for e in range(288): bases.append(bases[-1]+(counts[e]+127)//128)
            rows=bases[-1]*128;tokens=[-1]*rows;out_weights=[None]*rows;cursors=[0]*288
            scales=[1. if equal else .5+e/512 for e in range(288)]
            shared={2304+4*e:struct.unpack('I',struct.pack('f',s))[0] for e,s in enumerate(scales)}
            def atomic(address,amount):
                data,index=address;old=data[index];data[index]+=amount;return old
            def store(address,value): data,index=address;data[index]=value
            for token,row in enumerate(ids):
                warp=token%8;slot=warp*32
                env=dict(environment(shared),token_idx=token,warp_idx=warp,num_topk=8,route_slot_base=slot,
                         topk_ids=Array(e for r in ids for e in r),topk_weights=Array(weights),
                         expert_write_rows=cursors,expert_tile_base=bases,token_map=tokens,token_weights=out_weights,
                         route_phys_rows_addr=0,route_expert_ids_addr=1152,route_scales_addr=1152,
                         expert_scales_addr=2304,num_k_tiles=64,
                         get_ptr_as_int64=lambda data,index:(data,index),atomic_add_global_i32=atomic,
                         st_global_i32=store,st_global_f32=store)
                execute_lanes(reservation,[dict(env,lane_id=lane) for lane in range(32)])
                state=shared[1152+(slot+31)*4]
                self.assertEqual(state,8|(int(all(scales[e]==scales[row[0]] for e in row))<<4))
                for j in range(8):
                    phys=shared[(slot+j)*4]
                    self.assertEqual(tokens[phys],token)
                    self.assertEqual(struct.pack('f',out_weights[phys]),struct.pack('f',weights[token*8+j]))
                    expected=(phys//128)*64*512+(phys%32)*16+((phys%128)//32)*4
                    self.assertEqual(shared[(slot+j+8)*4],expected)
            self.assertEqual(cursors,[counts[e] for e in range(288)])
            self.assertEqual(Counter(x for x in tokens if x>=0),Counter({t:8 for t in range(67)}))


if __name__=='__main__': unittest.main()
