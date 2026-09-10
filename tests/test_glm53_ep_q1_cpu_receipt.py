"""Q1 CPU admission consumes the host layout witness, not the retired FC1 receipt.

The small generic layout interpreter supplies synthetic consumer layouts only.
Real CuTe layout validation and source-bound artifacts remain the normal CPU gate.
"""
import ast
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

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
    # Physical layouts have real padding, broadcast coordinates and swizzles.
    # The Q1 guard under test enumerates its own actual source/output addresses.
    def plain(shape, stride):
        return SimpleNamespace(shape=shape, stride=stride)
    def swizzled(shape, stride, params):
        return SimpleNamespace(outer=plain(shape, stride), offset=0,
            inner=SimpleNamespace(num_bits=params[0], num_base=params[1], num_shift=params[2]))
    cute = SimpleNamespace(crd2idx=lambda c, l: index(c, l.shape, l.stride),
                           cosize=lambda l: extent(l.shape, l.stride)+1)
    namespace = functions(ROOT/'overlay/modules/glm53_moe/moe_static_ep_tiled.py',
                          {'_check_ep_q1_layout'}, {'cute': cute})
    owner = SimpleNamespace(ep_q1_pair_geometry_proven=True, ep_decode_opt=True,
        a_dtype=SimpleNamespace(width=4), sf_dtype=SimpleNamespace(width=8),
        buffer_align_bytes=1024, fast_math=fast_math)
    namespace['_check_ep_q1_layout'](owner,
        swizzled((16,128,1), (128,1,2048), (2,3,3)),
        swizzled((128,128,1), (128,1,16384), (2,4,3)),
        plain(((32,4),(16,4,2),1), ((16,4),(0,1,512),1024)))
    return owner


class Q1ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hashlib
        cls.probe = functions(ROOT/'probes/glm53_ep_tiled_compile.py',
            {'opt_q1_pair_layout', 'validate_q1_pair_layout'}, {'hashlib': hashlib, 're': re})

    def test_host_witness_transfer_preserves_actual_math_and_selection(self):
        for fast in (False, True):
            owner = witness(fast)
            # Only the mode slot is consumed by this witness binder; the full
            # cache key/ABI is validated separately by opt_static_specialization.
            key = (None,)*6 + (fast,)
            raw = self.probe['opt_q1_pair_layout'](owner, key)
            transferred = json.loads(json.dumps(raw))
            self.assertEqual(self.probe['validate_q1_pair_layout'](transferred, fast_math=fast), raw)
            self.assertEqual(raw['math_mode'], 'fast' if fast else 'precise')
            self.assertEqual(raw['rows'][8]['sc1_bytes'], 2048)
            self.assertEqual(raw['rows'][8]['a2_bytes'], 512)
            with self.assertRaises(AssertionError):
                self.probe['opt_q1_pair_layout'](owner, (None,)*6 + (not fast,))
            for field in ('ep_decode_opt', 'ep_q1_pair_layout_proven'):
                changed=copy.copy(owner);setattr(changed, field, False)
                with self.assertRaises(AssertionError):
                    self.probe['opt_q1_pair_layout'](changed, key)

    def test_transferred_witness_rejects_old_incomplete_or_wrong_ownership_metadata(self):
        original = witness().ep_q1_pair_layout_receipt
        mutations = (
            lambda r: r.update(math_mode='precise'),
            lambda r: r.update(selected=False),
            lambda r: r.update(proven=1),
            lambda r: r.update(shuffle_threads=32),
            lambda r: r.update(stage=1),
            lambda r: r['element_bits'].update(a2=8),
            lambda r: r['layouts']['a2'].update(swizzle='(0, 0, 0)'),
            lambda r: r['layouts']['sc1'].pop('stride'),
            lambda r: r['rows'].pop(),
            lambda r: r['rows'].reverse(),
            lambda r: r['rows'][8].update(a2_bytes=256),
            lambda r: r['rows'][8].update(scale_owners=128),
            lambda r: r['rows'][0].update(rows=False),
            lambda r: r['rows'][8].pop('ownership_sha256'),
            lambda r: r['rows'][8].update(source_sha256='not-a-hash'),
            lambda r: r['rows'][8].update(packed_sha256=r['rows'][7]['packed_sha256']),
            lambda r: r.update(register_layout={'proven': True}),
        )
        for mutate in mutations:
            changed=copy.deepcopy(original);mutate(changed)
            with self.assertRaises(AssertionError):
                self.probe['validate_q1_pair_layout'](changed, fast_math=True)
        for stale in ({}, {'proven':True,'threads':128,'raw_stage_bytes':2048,'words_per_thread':[4]}):
            with self.assertRaises(AssertionError):
                self.probe['validate_q1_pair_layout'](stale, fast_math=True)
        with self.assertRaises(AssertionError):
            self.probe['validate_q1_pair_layout'](original, fast_math=1)


if __name__ == '__main__':
    unittest.main()
