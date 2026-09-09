"""CPU source/coordinate oracles; actual CuTe lowering is a separate gate."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy
import struct
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
MICRO = ROOT/'overlay/modules/glm53_moe/moe_micro_kernel.py'
PROOF = ROOT/'measurements/glm53_ep_local_20260908/micro-scatter-ownership'


def function(name):
    node = next(n for n in ast.walk(ast.parse(MICRO.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    if name == 'kernel':
        # Preserve this oracle for the independent direct-scatter path.
        # The shared-FC1-A tests cover its grouped metadata specialization.
        class SelectSeparateFC1(ast.NodeTransformer):
            def visit_If(self, item):
                value = ast.unparse(item.test)
                if value in ('cutlass.const_expr(self.shared_fc1_a)',
                             'cutlass.const_expr(not self.shared_fc1_a)'):
                    statements = item.body if 'not ' in value else item.orelse
                    result = []
                    for statement in statements:
                        chosen = self.visit(statement)
                        result.extend(chosen if isinstance(chosen, list) else [chosen])
                    return result
                return self.generic_visit(item)
        node = SelectSeparateFC1().visit(node)
    return node


def extract(fn, namespace):
    fn = copy.deepcopy(fn); fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom('__future__', [ast.alias('annotations')], 0), fn],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(MICRO), 'exec'), namespace)
    return namespace[fn.name]


def coords(tid):
    # Independent oracle derived from the pinned, actually emitted M32 r2s PTX.
    # test_pair_coordinates_match_original_ptx checks every address, not just
    # this formula's bijectivity. Runtime source uses CuTe identity partitions.
    lane, warp = tid % 32, tid // 32
    values = []
    for pair in range(16):
        row = (lane >> 2) + 16*(warp & 1) + 8*(pair & 1)
        k = pair >> 1
        col = 2*(lane & 3) + 16*(warp >> 1) + 8*(k & 1) + 32*((k >> 1) & 1) + 64*((k >> 2) & 1)
        values.extend([(row, col, 0), (row, col+1, 0)])
    return values


def direct_guard(fn=None):
    return next(n for n in ast.walk(fn or function('kernel')) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'cutlass.const_expr(self.ep_direct_scatter)'
                and any(isinstance(x, ast.For) for x in n.body))


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    return struct.unpack('<f', struct.pack('<I', (rounded & 0xffff) << 16))[0]


class DirectScatterTests(unittest.TestCase):
    def test_constructor_is_default_off_and_rejects_other_variants(self):
        namespace = dict(cutlass=SimpleNamespace(Float32='f32'), DenseGemmKernel=object,
                         is_gated_activation=lambda _: True,
                         utils=SimpleNamespace(get_smem_capacity_in_bytes=lambda _: 101376),
                         pipeline=SimpleNamespace(NamedBarrier=lambda **kw: kw))
        init = extract(function('__init__'), namespace)
        kw = dict(sf_vec_size=16, mma_tiler_mn=(32,128), output_tile_count_n=16,
                  activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                  swiglu_limit=10., skip_zero_weight_expert_id=72, scatter_fp32=True)
        base = SimpleNamespace(); init(base, **kw)
        self.assertFalse(base.ep_direct_scatter)
        candidate = SimpleNamespace(); init(candidate, **kw, ep_direct_scatter=True)
        self.assertTrue(candidate.ep_direct_scatter)
        for field, value in [('sf_vec_size',32), ('mma_tiler_mn',(64,128)),
                             ('output_tile_count_n',8), ('activation','silu'),
                             ('swiglu_alpha',1.702), ('swiglu_beta',1.),
                             ('swiglu_limit',9.), ('skip_zero_weight_expert_id',None),
                             ('scatter_fp32',False), ('single_token',True),
                             ('share_expert_scales',True),
                             ('share_input_across_experts',True)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'direct scatter'):
                init(SimpleNamespace(), **{**kw, field:value}, ep_direct_scatter=True)

    def test_real_shape_admission_precedes_tma_and_layout_work(self):
        fn = function('__call__')
        idx = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'cutlass.const_expr(self.ep_direct_scatter)')
        self.assertEqual(ast.unparse(fn.body[idx+1]), 'self._setup_attributes(hidden_size=hidden_size)')
        code = compile(ast.Module(body=[fn.body[idx]], type_ignores=[]), str(MICRO), 'exec')
        shapes = dict(a_input=(8,4096), topk_ids=(64,), topk_weights=(64,),
                      b_w13=(4096,4096,72), b_down=(4096,2048,72), token_map=(72,64))
        ns = dict(self=SimpleNamespace(ep_direct_scatter=True),
                  cutlass=SimpleNamespace(const_expr=lambda x:x),
                  **{k:SimpleNamespace(shape=v) for k,v in shapes.items()})
        exec(code, ns)
        for key, shape in shapes.items():
            bad_shape = (*shape[:-1], shape[-1]+1)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'exact EP M8'):
                exec(code, {**ns, key:SimpleNamespace(shape=bad_shape)})

    def test_pair_coordinates_match_original_ptx(self):
        identity = json.loads((PROOF/'identity.json').read_text())
        spec = identity['variants']['m32-topk8-fp32']
        raw = gzip.decompress((PROOF/'m32-topk8-fp32.ptx.gz').read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), spec['ptx_sha256'])
        lines = raw.decode().splitlines()
        a,b = spec['store_range']
        registers = [re.search(r'\[(%r\d+)\]', line)[1] for line in lines[a-1:b]]
        evaluate = runpy.run_path(str(PROOF/'verify.py'))['evaluate']
        seen = set()
        for tid in range(128):
            values = evaluate(lines, spec['producer_ranges'],
                              {spec['base_register']:0, spec['tid_register']:tid}, tid)
            points = coords(tid)
            for pair, reg in enumerate(registers):
                row,col,_ = points[2*pair]
                offset = 57344 + 2*(row*64+(col%64)+(col//64)*32*64)
                self.assertEqual(offset ^ ((offset >> 3) & 112), values[reg])
            for point in points:
                self.assertNotIn(point, seen); seen.add(point)
        self.assertEqual(seen, {(r,c,0) for r in range(32) for c in range(128)})

    def test_compile_time_identity_validator_rejects_missing_or_bad_pairs(self):
        class Partition:
            def __init__(self, points, stages=1): self.points,self.stages = points,stages
            def __getitem__(self, key):
                return self if isinstance(key, tuple) else self.points[key]
        class Thread:
            def __init__(self, tid, mutate, stages): self.tid,self.mutate,self.stages=tid,mutate,stages
            def partition_D(self, identity):
                if identity == 'actual-nested-sC-shape':
                    return Partition(coords(self.tid), self.stages)
                return Partition(self.mutate(self.tid, coords(self.tid)))
            def partition_S(self, identity): return (32,1,1,1)
        def run(mutate, stages=1):
            copy_op = SimpleNamespace(get_slice=lambda tid:Thread(tid,mutate,stages))
            cute = SimpleNamespace(
                nvgpu=SimpleNamespace(CopyUniversalOp=lambda:None,
                                      warp=SimpleNamespace(StMatrix8x8x16bOp=lambda *a:None)),
                make_copy_atom=lambda *a:None, make_tiled_copy_C_atom=lambda *a:None,
                make_tiled_copy_S=lambda *a:copy_op, make_identity_tensor=lambda s:s,
                shape=lambda x:x, make_layout=lambda x:x,
                size=lambda x, mode=None: (x.stages if mode == [3]
                                          else len(x.points) if isinstance(x,Partition) else 32))
            validate = extract(function('_validate_ep_direct_scatter_layout'),
                               dict(cute=cute,cutlass=SimpleNamespace(BFloat16='bf16')))
            validate(SimpleNamespace(c_layout=SimpleNamespace(is_m_major_c=lambda:False),
                                     tiled_mma=None,epi_tile=(32,128),num_mma_warps=4,
                                     shared_fc1_a=False,
                                     epi_smem_layout_staged=SimpleNamespace(outer='actual-nested-sC-shape'),
                                     num_threads_per_warp=32))
        run(lambda tid, points:points)
        for stages in (0,2):
            with self.assertRaisesRegex(ValueError, 'one epilogue buffer'):
                run(lambda tid,points:points,stages=stages)
        for mutate in [lambda tid,p:p[:-2] if tid==0 else p,
                       lambda tid,p:[p[1],p[0],*p[2:]] if tid==0 else p,
                       lambda tid,p:[*p[:1],(1,1,0),*p[2:]] if tid==0 else p,
                       lambda tid,p:coords(0) if tid==1 else p,
                       lambda tid,p:[(r,c+128,z) for r,c,z in p] if tid==0 else p]:
            with self.assertRaisesRegex(ValueError, 'direct scatter'):
                run(mutate)

    def test_direct_source_keeps_rounding_and_common_post_barrier(self):
        fn = function('kernel'); guard=direct_guard(fn)
        validator=function('_validate_ep_direct_scatter_layout')
        host_identity=next(n for n in ast.walk(validator) if isinstance(n,ast.Call)
                           and ast.unparse(n.func)=='cute.make_identity_tensor')
        runtime_identity=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign)
                              and ast.unparse(n.targets[0])=='ep_identity')
        self.assertEqual(ast.dump(host_identity.args[0]),
                         ast.dump(runtime_identity.value.args[0]))
        self.assertEqual(ast.unparse(host_identity.args[0]), '(*self.epi_tile, 1)')
        setup=next(n for n in ast.walk(fn) if isinstance(n,ast.If)
                   and runtime_identity in n.body)
        self.assertFalse(any(isinstance(n,ast.Raise) for n in ast.walk(setup)))
        host_guard=next(n for n in ast.walk(validator) if isinstance(n,ast.If)
                        and ast.unparse(n.test)=='cute.size(staged_destination, mode=[3]) != 1')
        self.assertIsInstance(host_guard.body[0],ast.Raise)
        self.assertIn('cute.shape(self.epi_smem_layout_staged.outer)', ast.unparse(validator))
        self.assertIn('staged_destination = thread_copy.partition_D(staged_identity)',
                      ast.unparse(validator))
        parent = next(n for n in ast.walk(fn) if isinstance(n,ast.For) and guard in n.body)
        idx=parent.body.index(guard)
        prefix=[ast.unparse(n) for n in parent.body[idx-4:idx]]
        self.assertEqual(prefix[:3], ['acc_vec = tRS_rD.load()',
                                    'acc_vec = acc_vec.to(cutlass.BFloat16)',
                                    'tRS_rD_out.store(acc_vec)'])
        self.assertEqual(ast.unparse(parent.body[-1]), 'self.epilog_sync_barrier.arrive_and_wait()')
        text=ast.unparse(ast.Module(body=guard.body,type_ignores=[]))
        for forbidden in ['cute.copy(', 'fence_proxy', 'arrive_and_wait', 'shuffle_sync', 'sC[']:
            self.assertNotIn(forbidden,text)
        for expected in ['ep_row < valid_tile_rows', 'cutlass.Float32(tRS_rD_out[2 * ep_pair])',
                         'cutlass.Float32(tRS_rD_out[2 * ep_pair + 1])',
                         'ep_weight * ep_v0', 'ep_weight * ep_v1', 'scatter_add_bf16x2_to_f32(']:
            self.assertIn(expected,text)

    def test_direct_fragment_pair_scatter_preserves_routes_and_poisoned_tail(self):
        code=compile(ast.Module(body=direct_guard().body,type_ignores=[]),str(MICRO),'exec')
        class Slice:
            def __init__(self, points): self.points=points
            def __getitem__(self,key): return self.points
        for count in (0,1,8,31,32,33,48,63,64):
            for col_base in (0,7*128,31*128):
                observed=[]; expected=[]
                for tile in range((count+31)//32):
                    tile_base=tile*32; valid=min(32,count-tile_base)
                    weights=[1.,.5,-2.,0.,-0.,2.,-.5,1.]
                    def tok_load(address):
                        row=(address-1000)//4
                        if not 0<=row<valid: raise AssertionError('poisoned token metadata')
                        return (tile_base+row)%8
                    def weight_load(address):
                        row=(address-2000)//4
                        if not 0<=row<valid: raise AssertionError('poisoned weight metadata')
                        return weights[(tile_base+row)%8]
                    for tid in range(128):
                        points=coords(tid)
                        class Values:
                            def __getitem__(self,index):
                                row,col,_=points[index]
                                if row>=valid: raise AssertionError('poisoned inactive register')
                                # The FC2 prefix has already rounded downalpha*acc.
                                return bf16((tile_base+row+1)*.137+(col+1)*.0031)
                        def scatter(address,v0,v1):
                            observed.extend([(address,struct.pack('<f',bf16(v0))),
                                             (address+1,struct.pack('<f',bf16(v1)))])
                        ns=dict(cute=SimpleNamespace(size=lambda _:32,Tensor=object),
                                cutlass=SimpleNamespace(range_constexpr=range,Float32=float),
                                self=SimpleNamespace(epi_tile=(32,128)), Int32=int,
                                ep_tRS_coords=Slice(points), epi_buffer=0,epi_m=0,
                                valid_tile_rows=valid,tRS_rD_out=Values(),
                                scatter_tok_base_addr=1000,scatter_weight_base_addr=2000,
                                _ld_shared_i32=tok_load,_ld_shared_f32=weight_load,
                                scatter_output=None,scatter_N=4096,tile_n_base_cur=col_base,
                                get_ptr_as_int64=lambda tensor,index:index,
                                scatter_add_bf16x2_to_f32=scatter)
                        exec(code,ns)
                    for row in range(valid):
                        global_row=tile_base+row; token=global_row%8; weight=weights[token]
                        for col in range(128):
                            value=bf16((global_row+1)*.137+(col+1)*.0031)
                            expected.append((token*4096+col_base+col,struct.pack('<f',bf16(weight*value))))
                self.assertEqual(len(observed),count*128)
                self.assertCountEqual(observed,expected)


if __name__ == '__main__':
    unittest.main()
