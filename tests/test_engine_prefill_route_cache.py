"""Keep route-cache publication intact while bounding BF16 output zeroing."""
import ast
import copy
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x'


def method(tree, name):
    return copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == name))


class PrefillRouteCacheTests(unittest.TestCase):
    def test_complete_route_producer_changes_only_the_zeroed_word_extent(self):
        original_path = ROOT/'moe_dynamic_gated_sf6_q0.py'
        original = ast.parse(original_path.read_text())
        candidate = ast.parse((ROOT/'moe_dynamic_gated_sf6_prefill.py').read_text())
        expected = next(ast.literal_eval(n.value) for n in candidate.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id == 'Q0_SOURCE_SHA256' for t in n.targets))
        self.assertEqual(hashlib.sha256(original_path.read_bytes()).hexdigest(), expected)
        a = method(original,'initialize_route_q0_and_publish')
        b = method(candidate,'initialize_route_q0_and_publish')
        changed = [n for n in ast.walk(b) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id == 'cols_u32' for t in n.targets)]
        self.assertEqual(len(changed),1)
        self.assertEqual(ast.unparse(changed[0].value),'cols // Int32(2)')
        changed[0].value = ast.Name('cols',ast.Load())
        self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
        cls = next(n for n in candidate.body if isinstance(n,ast.ClassDef))
        self.assertEqual([ast.unparse(n) for n in cls.bases],['MoEGatedDynamicKernelSF6Words'])
        self.assertNotIn('scatter_sC_to_gmem', [getattr(n,'name',None) for n in cls.body])

    def test_zeroing_covers_bf16_output_without_crossing_its_last_byte(self):
        tree = ast.parse((ROOT/'moe_dynamic_gated_sf6_prefill.py').read_text())
        body = method(tree,'initialize_route_q0_and_publish')
        wanted = {'cols_u32','scatter_total_u32','scatter_vecs'}
        nodes = [n for n in body.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets)]
        code = compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), '<zero extent>', 'exec')
        for rows in (8193,9216,16128,32256,32768):
            ns = dict(Int32=int,cols=4096,num_tokens=rows)
            exec(code,ns)
            self.assertEqual(ns['scatter_total_u32']*4,rows*4096*2)
            self.assertEqual(ns['scatter_vecs']*16,rows*4096*2)
            # Each thread owns one arithmetic progression of vector indices.
            total, stride = ns['scatter_vecs'],48*288
            counts = [max(0,(total-1-tid)//stride+1) for tid in range(stride)]
            self.assertEqual(sum(counts),total)
            last = max(tid+(count-1)*stride for tid,count in enumerate(counts) if count)
            self.assertEqual((last+1)*16,rows*4096*2)


if __name__ == '__main__':
    unittest.main()
