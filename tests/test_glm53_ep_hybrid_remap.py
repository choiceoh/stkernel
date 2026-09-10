"""Execute real remap AST and existing baseline contracts, no CUDA imports."""
import ast
import gzip
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'overlay/modules/glm53_moe/glm53_ep_route_remap.py'
spec=importlib.util.spec_from_file_location('existing_remap_contracts',ROOT/'tests/test_glm53_ep_route_remap.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.REMAP=P

class MaskedLanes(base.TritonLanes):
    @staticmethod
    def store(pointer,value,mask=True):
        offsets=pointer.offset.values
        values=value.values if isinstance(value,base.Vector) else [value]*len(offsets)
        masks=mask.values if isinstance(mask,base.Vector) else [mask]*len(offsets)
        for index,item,enabled in zip(offsets,values,masks):
            if enabled:
                if not 0<=index<len(pointer.values):raise AssertionError('unmasked OOB store')
                pointer.values[index]=item

class HybridRemapTests(unittest.TestCase):
    def test_hybrid_is_tiled_only_and_passes_explicit_extent(self):
        harness=base.AdmissionTests()
        ns,_,launch=harness.namespace()
        args=harness.inputs();args.update(num_local_experts=144,local_expert_offset=144)
        self.assertFalse(ns['try_remap_ep_local'](**args))
        self.assertFalse(ns['ep_route_remap_supported'](**args))
        launch.assert_not_called()
        self.assertTrue(ns['try_remap_ep_local'](**args,_tiled_owner=True))
        self.assertEqual(launch.call_args.kwargs['LOCAL_EXPERTS'],144)
        for bad in (143,145,True,144.0):
            self.assertFalse(ns['try_remap_ep_local'](**dict(args,num_local_experts=bad),_tiled_owner=True))

    def test_real_device_ast_preserves_bits_masks_tail_and_narrowing(self):
        node=next(n for n in ast.parse(P.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='_remap_ep_local_kernel')
        node.decorator_list=[]
        lanes=MaskedLanes();ns={'tl':lanes}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(P),'exec'),ns)
        kernel=ns[node.name]
        for e,offset in ((72,0),(72,216),(144,0),(144,144)):
            ids=[-1,-2**40,0,offset,offset+e-1,offset+e,287,288,2**32+offset,offset+1]*3
            mapping=[-1]*288
            mapping[offset:offset+e]=list(range(e))
            mapping[offset:offset+4]=[0,e,e+1,2**32+7]
            for initial_map in (None,[],mapping):
                for dtype,nan,minuszero in (('float32',0x7fc01234,0x80000000),
                                           ('float16',0x7e12,0x8000),
                                           ('bfloat16',0x7fc1,0x8000)):
                    weights=([nan,minuszero,1]*10)
                    oi,ow=[77]*len(ids),[77]*len(ids)
                    saved=(ids[:],weights[:],None if initial_map is None else initial_map[:])
                    for cta in range(2):
                        lanes.row=cta
                        kernel(base.Pointer(ids,'int64'),base.Pointer(weights,dtype),
                               base.Pointer(initial_map or oi,'int64'),
                               base.Pointer(oi,'int32'),base.Pointer(ow,dtype),
                               len(ids),offset,MAP_LEN=len(initial_map or []),
                               HAS_MAP=initial_map is not None,BLOCK=16,LOCAL_EXPERTS=e)
                    want_ids=[];want_weights=[]
                    for expert,w in zip(ids,weights):
                        if initial_map is not None:
                            remote=not 0<=expert<len(initial_map) or initial_map[expert]<0
                            local=e if remote else base._i32(initial_map[expert])
                        else:
                            local=base._i32(base._i32(expert)-offset)
                            remote=expert<0 or not 0<=local<e
                        want_ids.append(e if remote else local)
                        want_weights.append(0 if remote else w)
                    self.assertEqual(oi,want_ids)
                    self.assertEqual(ow,want_weights)
                    self.assertEqual((ids,weights,initial_map),saved)

    def test_short_decode_path_is_unchanged(self):
        original=gzip.decompress((ROOT/'tests/fixtures/glm53_ep_hybrid/ep4_route_remap.py.gz').read_bytes())
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         'b577457ff566add86837c467e2062b715bc9b1f2d1d2bfbb2addc0b9520a9b11')
        before=ast.parse(original)
        after=ast.parse(P.read_text())
        names=('_prepare_ep_short_decode_kernel','_ep_short_decode_metadata',
               'ep_short_decode_prepare_supported','try_prepare_ep_short_decode')
        for name in names:
            old=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
            new=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(old),ast.dump(new))

if __name__=='__main__':
    unittest.main()
