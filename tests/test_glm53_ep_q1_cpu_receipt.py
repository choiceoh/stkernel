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

        # Exercise the outer artifact-reader connection with stored PTX/cubin
        # bytes. Unrelated ABI/resource validators are stubbed in this unit; the
        # real source-bound normal CPU gate retains every validator unchanged.
        selected = dict(a_ring=True, word_unpack=True, scatter_bf16=True,
            output_dtype='bfloat16', decode_opt=True, storage_bytes=98304)
        outer = functions(ROOT/'probes/run_glm53_ep_tiled_cpu.py', {'validate_artifacts'},
            dict(Path=Path, hashlib=hashlib, STATIC_ROWS=(), GLOBAL_STATIC_CASES=(),
                OPT_STATIC_CASES=(('M6-local',6,'local'),), DYNAMIC_ROWS=(),
                opt_static_specialization=lambda *args: selected,
                opt_shared_capacity=lambda passed: passed['shared_capacity'],
                validate_q1_register_layout=self.probe['validate_q1_register_layout']))
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            folder=directory/'opt-static/M6-local';folder.mkdir(parents=True)
            resources='CPU unit fixture; not a real compiler result'
            passed=dict(arm='opt-static/M6-local', cache_key=[None]*6+[True],
                specialization=selected, shared_capacity={}, q1_register_layout=original)
            for name,suffix in (('artifacts','.ptx'),('resources','.cubin')):
                path=folder/('fixture'+suffix);path.write_bytes(b'CPU fixture '+suffix.encode())
                item=dict(path=str(path.relative_to(directory)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if suffix=='.cubin':
                    item['resources']=resources;path.with_suffix('.resources.log').write_text(resources)
                passed[name]=[item]
            result=dict(static_passes=[],global_static_passes=[],dynamic_passes=[],opt_static_passes=[passed])
            outer['validate_artifacts'](directory,json.loads(json.dumps(result)))
            for key in ('q1_pair_layout','register_layout'):
                changed=copy.deepcopy(result)
                changed['opt_static_passes'][0][key]=changed['opt_static_passes'][0].pop('q1_register_layout')
                with self.assertRaises(AssertionError):
                    outer['validate_artifacts'](directory,changed)
            changed=copy.deepcopy(result)
            changed['opt_static_passes'][0]['q1_register_layout']['rows'][8]['max_load_bytes']=256
            with self.assertRaises(AssertionError):
                outer['validate_artifacts'](directory,changed)


if __name__ == '__main__':
    unittest.main()
