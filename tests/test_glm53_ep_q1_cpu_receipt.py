"""Register-max CPU admission consumes the host witness, not retired pair/FC1 receipts.

The small generic layout interpreter supplies synthetic consumer layouts only.
Real CuTe layout validation and source-bound artifacts remain the normal CPU gate.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [copy.deepcopy(node) for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == set(names)
    for node in body:
        node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


def size(shape):
    if type(shape) is int:
        return shape
    value = 1
    for part in shape:
        value *= size(part)
    return value


def index(coord, shape, stride):
    if type(shape) is int:
        return coord * stride
    if type(coord) is int:
        parts = []
        for part in shape:
            parts.append(coord % size(part))
            coord //= size(part)
        assert coord == 0
        coord = parts
    return sum(index(c, s, d) for c, s, d in zip(coord, shape, stride))


def extent(shape, stride):
    if type(shape) is int:
        return (shape-1)*stride
    return sum(extent(s, d) for s, d in zip(shape, stride))


def witness(fast_math=True):
    # Synthetic copy partitions exercise transport/guard integration only.
    # CPU7 must independently construct these from actual CuTe partition_D.
    partitions = [[None] * 16 for _ in range(128)]
    for warp in range(4):
        for row in range(8):
            for lane in range(4):
                tid = warp*32 + row*4 + lane
                for block in range(4):
                    for half in range(2):
                        for element in range(2):
                            i = block*4 + half*2 + element
                            partitions[tid][i] = (row+half*8,
                                block*32+(warp%2)*16+(warp//2)*8+lane*2+element, 0)

    class Tensor:
        shape = (4, 1, 4)
        def __init__(self, tid):
            self.tid = tid
        def __getitem__(self, i):
            return self if isinstance(i, tuple) else partitions[self.tid][i]

    class Thread:
        def __init__(self, tid):
            self.tid = tid
        def partition_D(self, identity):
            return Tensor(self.tid)
        def partition_S(self, identity):
            return SimpleNamespace(shape=(4, 1, 4, 1))

    def plain(shape, stride):
        return SimpleNamespace(shape=shape, stride=stride)
    def dense(shape):
        strides = []
        stride = 1
        for dimension in shape:
            strides.append(stride)
            stride *= size(dimension)
        return plain(shape, tuple(strides))
    def swizzled(shape, stride, params):
        return SimpleNamespace(outer=plain(shape, stride), offset=0,
            inner=SimpleNamespace(num_bits=params[0], num_base=params[1], num_shift=params[2]))
    cute = SimpleNamespace(crd2idx=lambda c, l: index(c, l.shape, l.stride),
        cosize=lambda l: extent(l.shape, l.stride)+1,
        make_copy_atom=lambda *args: object(),
        make_tiled_copy_S=lambda *args: SimpleNamespace(get_slice=Thread),
        make_tiled_copy_C_atom=lambda *args: object(),
        make_identity_tensor=lambda shape: SimpleNamespace(shape=shape),
        make_layout=dense, shape=lambda tensor: tensor.shape,
        size=lambda tensor: size(tensor.shape),
        nvgpu=SimpleNamespace(CopyUniversalOp=lambda: object(),
            warp=SimpleNamespace(StMatrix8x8x16bOp=lambda *args: object())))
    namespace = functions(ROOT/'overlay/modules/glm53_moe/moe_static_ep_tiled.py',
        {'_check_ep_q1_register_layout'},
        {'cute': cute, 'cutlass': SimpleNamespace(BFloat16=object())})
    owner = SimpleNamespace(ep_q1_register_geometry_proven=True, ep_decode_opt=True,
        a_dtype=SimpleNamespace(width=4), sf_dtype=SimpleNamespace(width=8),
        buffer_align_bytes=1024, fast_math=fast_math,
        c_layout=SimpleNamespace(is_m_major_c=lambda: False))
    namespace['_check_ep_q1_register_layout'](owner, object(),
        swizzled(((8,2),(64,2),(1,1)), ((64,512),(1,1024),(0,0)), (3,4,3)),
        swizzled((16,128,1), (128,1,0), (2,4,3)),
        plain(((32,4),(16,4,2),1), ((16,4),(0,1,512),0)))
    return owner


class Q1ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hashlib
        cls.probe = functions(ROOT/'probes/glm53_ep_tiled_compile.py',
            {'opt_q1_register_layout', 'validate_q1_register_layout'}, {'hashlib': hashlib, 're': re})

    def test_host_witness_transfer_preserves_actual_math_and_selection(self):
        for fast in (False, True):
            owner = witness(fast)
            # The mode and version tag are bound here; the full
            # cache key/ABI is validated separately by opt_static_specialization.
            key = (None,)*6 + (fast, 'glm53_ep_static_sf6_q1_register_max_v5')
            raw = self.probe['opt_q1_register_layout'](owner, key)
            transferred = json.loads(json.dumps(raw))
            self.assertEqual(self.probe['validate_q1_register_layout'](transferred, fast_math=fast), raw)
            self.assertEqual(raw['math_mode'], 'fast' if fast else 'precise')
            self.assertEqual(raw['rows'][8]['max_store_bytes'], 512)
            self.assertEqual(raw['rows'][8]['max_load_bytes'], 512)
            self.assertEqual(raw['sc1_capacity_bytes'], 4096)
            self.assertEqual(raw['rows'][8]['a2_bytes'], 512)
            with self.assertRaises(AssertionError):
                self.probe['opt_q1_register_layout'](owner, (None,)*6 + (not fast, key[-1]))
            for stale_tag in ('glm53_ep_static_sf6_q1_pair_v4', 'glm53_ep_static_sf6_fc1_register_v2'):
                with self.assertRaises(AssertionError):
                    self.probe['opt_q1_register_layout'](owner, key[:-1] + (stale_tag,))
            for field in ('ep_decode_opt', 'ep_q1_register_layout_proven'):
                changed=copy.copy(owner);setattr(changed, field, False)
                with self.assertRaises(AssertionError):
                    self.probe['opt_q1_register_layout'](changed, key)

    def test_transferred_witness_rejects_old_incomplete_or_wrong_ownership_metadata(self):
        original = witness().ep_q1_register_layout_receipt
        mutations = (
            lambda r: r.update(math_mode='precise'),
            lambda r: r.update(selected=False),
            lambda r: r.update(proven=1),
            lambda r: r.update(subgroup_threads=2),
            lambda r: r.update(partner_warp_xor=1),
            lambda r: r.update(scratch_bytes=256),
            lambda r: r.update(sc1_capacity_bytes=8192),
            lambda r: r.update(source_values=1024),
            lambda r: r.update(max_shuffle_collectives=4),
            lambda r: r.update(source_mapping_sha256='not-a-hash'),
            lambda r: r.update(source_mapping_sha256=hashlib.sha256(b'[]').hexdigest()),
            lambda r: r.update(copy_shapes=[]),
            lambda r: r['copy_shapes'].append(r['copy_shapes'][0]),
            lambda r: r['layout'].update(a2_swizzle='(0, 0, 0)'),
            lambda r: r['layout'].pop('sc1_stride'),
            lambda r: r['rows'].pop(),
            lambda r: r['rows'].reverse(),
            lambda r: r['rows'][8].update(max_store_bytes=256),
            lambda r: r['rows'][8].update(max_load_bytes=256),
            lambda r: r['rows'][8].update(a2_bytes=256),
            lambda r: r['rows'][8].update(source_values=2048),
            lambda r: r['rows'][0].update(rows=False),
            lambda r: r['rows'][8].pop('ownership_sha256'),
            lambda r: r['rows'][8].update(max_loads_sha256='not-a-hash'),
            lambda r: r['rows'][8].update(packed_sha256=r['rows'][7]['packed_sha256']),
            lambda r: r.update(register_layout={'proven': True}),
            lambda r: r.update(q1_pair_layout={'proven': True}),
        )
        for mutate in mutations:
            changed=copy.deepcopy(original);mutate(changed)
            with self.assertRaises(AssertionError):
                self.probe['validate_q1_register_layout'](changed, fast_math=True)
        for stale in ({}, {'proven':True,'threads':128,'raw_stage_bytes':2048,'words_per_thread':[4]}):
            with self.assertRaises(AssertionError):
                self.probe['validate_q1_register_layout'](stale, fast_math=True)
        with self.assertRaises(AssertionError):
            self.probe['validate_q1_register_layout'](original, fast_math=1)

        # Exercise the current generic artifact reader with CPU fixture bytes.
        # Matrix ABI/resource checking is explicitly outside this file-system
        # unit: the stub below checks only declared arm/group connectivity.
        # Retired Q1 ownership assertions remain in the tests above; this gate
        # no longer attaches them to artifacts from decode_opt=False kernels.
        tree = ast.parse((ROOT/'probes/glm53_ep_tiled_compile.py').read_text())
        names = {'BASELINE_STATIC_CASES', 'HYBRID_STATIC_CASES', 'COMPILE_GROUPS'}
        declarations = [copy.deepcopy(node) for node in tree.body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets)]
        self.assertEqual(len(declarations), len(names))
        namespace = {}
        exec(compile(ast.Module(body=declarations, type_ignores=[]),
                     '<actual compile matrix constants>', 'exec'), namespace)
        groups = namespace['COMPILE_GROUPS']
        matrix_calls = []
        def matrix_connectivity_only(result):
            self.assertEqual(set(result), {kind+'_passes' for kind,_,_,_ in groups})
            for kind,_,_,cases in groups:
                self.assertEqual([p['arm'] for p in result[kind+'_passes']],
                    [kind.replace('_','-')+'/'+case[0] for case in cases])
                for passed in result[kind+'_passes']:
                    self.assertFalse({'q1_register_layout','q1_pair_layout',
                                      'register_layout'} & set(passed))
            matrix_calls.append(True)
        outer = functions(ROOT/'probes/run_glm53_ep_tiled_cpu.py', {'validate_artifacts'},
            dict(Path=Path, hashlib=hashlib, COMPILE_GROUPS=groups,
                 validate_compile_matrix=matrix_connectivity_only))
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            resources='CPU unit fixture; not a real compiler result'
            result = {}
            for kind,_,_,cases in groups:
                result[kind+'_passes'] = []
                for case in cases:
                    arm=kind.replace('_','-')+'/'+case[0]
                    folder=directory/arm;folder.mkdir(parents=True)
                    passed=dict(arm=arm)
                    for name,suffix in (('artifacts','.ptx'),('resources','.cubin')):
                        path=folder/('fixture'+suffix)
                        path.write_bytes(b'CPU fixture '+arm.encode()+suffix.encode())
                        item=dict(path=str(path.relative_to(directory)),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        if suffix=='.cubin':
                            item['resources']=resources
                            path.with_suffix('.resources.log').write_text(resources)
                        passed[name]=[item]
                    result[kind+'_passes'].append(passed)
            outer['validate_artifacts'](directory,json.loads(json.dumps(result)))
            self.assertEqual(matrix_calls, [True])
            first = result['baseline_static_passes'][0]
            self.assertTrue(first['arm'].startswith('baseline-static/'))
            def rejected(changed):
                with self.assertRaises((AssertionError, FileNotFoundError)):
                    outer['validate_artifacts'](directory, changed)
            changed=copy.deepcopy(result)
            changed['baseline_static_passes'][0]['artifacts'][0]['sha256']='0'*64
            rejected(changed)
            changed=copy.deepcopy(result)
            changed['baseline_static_passes'][0]['resources'][0]['resources']='changed log'
            rejected(changed)
            for unsafe in ('../outside.ptx', '/tmp/outside.ptx',
                           first['arm']+'/missing.ptx',
                           first['arm']+'/fixture.cubin'):
                changed=copy.deepcopy(result)
                changed['baseline_static_passes'][0]['artifacts'][0]['path']=unsafe
                rejected(changed)
            path=directory/first['artifacts'][0]['path']; original_bytes=path.read_bytes()
            path.write_bytes(b'tampered fixture')
            rejected(result)
            path.write_bytes(original_bytes)
            log=(directory/first['resources'][0]['path']).with_suffix('.resources.log')
            log.write_text('tampered resource log')
            rejected(result)
            log.write_text(resources)
            extra=directory/'unclaimed.ptx';extra.write_bytes(b'extra compiler output')
            rejected(result)
            extra.unlink()
            outer['validate_artifacts'](directory,result)

if __name__ == '__main__':
    unittest.main()
