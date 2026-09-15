"""Staged MoE scatter: exact column ownership, alignment and handle isolation."""
import ast
import copy
from types import SimpleNamespace as NS
import unittest

from tests.test_engine_moe_scatter_config import namespace
from tests.test_engine_moe_sf6_staging import CLASS, SOURCE, geometry, method
from probes.engine_moe_pair_check import _pair_configs


def staged_scatter():
    """Execute the real staged epilogue with observable global RED arguments."""
    kernel = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
    # The staging branch starts after its existing publication barrier.
    branch = next(n for n in ast.walk(kernel) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'cutlass.const_expr(self.direct_scatter)'
                  and any(isinstance(s, ast.While) for s in n.orelse))
    start = next(i for i, n in enumerate(branch.orelse) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'warp_epi_rows')
    end = next(i for i in range(start, len(branch.orelse))
               if isinstance(branch.orelse[i], ast.While))
    module = ast.Module(body=copy.deepcopy(branch.orelse[start:end+1]), type_ignores=[])
    return compile(ast.fix_missing_locations(module), str(SOURCE), 'exec')


class VectorScatterTests(unittest.TestCase):
    def test_staged_output_preserves_every_contribution_and_halves_red_calls(self):
        body = staged_scatter()
        byte_offset = method('_scatter_smem_byte_offset', {})
        shared_memory = {(row*512+col*2) ^ (((row*512+col*2)//128 % 8)*16):
                         float(row*256+col+1) for row in range(16) for col in range(256)}
        class Shared:
            def layout(self, coord):
                row, col, stage = coord
                self_test.assertEqual(stage, 0)
                return row*256+col
            def __getitem__(self, coord):
                row, col, stage = coord
                self_test.assertEqual(stage, 0)
                reads.append(2)
                return float(row * 256 + col + 1)
        self_test = self
        for valid in (0, 1, 7, 8, 15, 16):
            outputs, counts, load_counts = [], [], []
            for wide, packed in ((False, False), (True, False), (True, True)):
                writes, calls, reads = [], [], []
                def scatter(address, *values):
                    self.assertEqual(address % (4 * len(values)), 0)
                    calls.append(address)
                    writes.extend((address + 4 * i, value) for i, value in enumerate(values))
                def packed_scatter(address, source, weight):
                    self.assertEqual(source % 16, 0)
                    reads.append(16)
                    values = [weight*shared_memory[source+2*j] for j in range(8)]
                    scatter(address, *values[:4])
                    scatter(address+16, *values[4:])
                for tile in (0, 15):
                    for tid in range(128):
                        env = dict(self=NS(route_scatter=False, scatter_fp32=True,
                                           scatter_vec4=wide, scatter_packed_load=packed,
                                           _scatter_smem_byte_offset=lambda i: byte_offset(None, i),
                                           num_threads_per_warp=32),
                                   cutlass=NS(const_expr=bool, Float32=float), Int32=int, Int64=int,
                                   valid_tile_rows=valid, warp_m_base=0, warp_n_base=(tid//32)*64,
                                   lane_id=tid%32, tile_n_base_cur=tile*256, scatter_N=4096,
                                   sC=Shared(), scatter_tok_base_addr=0, scatter_weight_base_addr=0,
                                   scatter_smem_base_addr=0,
                                   ld_shared_i32_relaxed=lambda address: (address//4*5)%8,
                                   _ld_shared_f32=lambda address: (address//4%5-2)*.25,
                                   get_ptr_as_int64=lambda tensor, index: 4*index,
                                   scatter_output=object(), scatter_add_bf16x2_to_f32=scatter,
                                   scatter_add_bf16x4_to_f32=scatter,
                                   scatter_add_bf16x8_from_smem_to_f32=packed_scatter)
                        exec(body, env)
                # Check against a separate row/column enumeration, including
                # repeated destinations, zero and negative route weights.
                expected = sorted((4*((row*5)%8*4096+tile*256+col),
                                   (row%5-2)*.25*(row*256+col+1))
                                  for tile in (0, 15) for row in range(valid) for col in range(256))
                self.assertEqual(sorted(writes), expected)
                outputs.append(sorted(writes)); counts.append(len(calls))
                load_counts.append(len(reads))
                self.assertEqual(sum(reads), valid*256*2*2)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[1], outputs[2])
            self.assertEqual(counts[0], 2*counts[1])
            self.assertEqual(counts[1], counts[2])
            self.assertEqual(load_counts[1], 8*load_counts[2])

    def test_default_and_control_are_isolated_only_for_staged_sf6_tiles(self):
        ns = namespace()
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        for recipe in ('t', 't,r', 't,r,sf6', 't,r,sf6,batch'):
            base = ns['_parse_glm53_static_v2'](recipe)
            for rows in range(129):
                chosen = normalize(base, rows)
                control = normalize(dict(base, scatter_vec4=False), rows)
                scalar_load = normalize(dict(base, scatter_packed_load=False), rows)
                enabled = 'sf6' in recipe and 1 <= rows <= 8
                self.assertEqual(chosen['scatter_vec4'], enabled)
                self.assertEqual(chosen['scatter_packed_load'], enabled)
                self.assertFalse(control['scatter_vec4'])
                self.assertFalse(control['scatter_packed_load'])
                self.assertFalse(scalar_load['scatter_packed_load'])
                self.assertEqual(normalize(chosen, rows), chosen)
                self.assertEqual(normalize(scalar_load, rows), scalar_load)
                self.assertEqual(key(chosen, m=rows) != key(control, m=rows), enabled)
                self.assertEqual(key(chosen, m=rows) != key(scalar_load, m=rows), enabled)
        base = dict(ns['_parse_glm53_static_v2']('t,r,sf6,batch'), c2_direct_scatter=False)
        self.assertTrue(normalize(base, 16)['scatter_vec4'])
        for route, direct in ((True, False), (False, True), (True, True)):
            chosen = normalize(dict(base, probe_route_scatter=route, probe_direct_scatter=direct), 7)
            self.assertFalse(chosen['scatter_vec4'])

    def test_constructor_excludes_non_fp32_and_register_owned_outputs(self):
        self.assertTrue(geometry(scatter_fp32=True).scatter_vec4)
        self.assertTrue(geometry(scatter_fp32=True).scatter_packed_load)
        self.assertFalse(geometry(scatter_fp32=True, scatter_packed_load=False).scatter_packed_load)
        for flags in (dict(scatter_fp32=False), dict(scatter_fp32=True, scatter_vec4=False),
                      dict(scatter_fp32=True, direct_scatter=True),
                      dict(scatter_fp32=True, route_scatter=True),
                      dict(scatter_fp32=True, packed=False), dict(scatter_fp32=True, reform=False)):
            self.assertFalse(geometry(**flags).scatter_vec4)
            self.assertFalse(geometry(**flags).scatter_packed_load)

    def test_pair_probes_change_the_named_axis_at_both_row_counts(self):
        ns = namespace()
        md = NS(_parse_glm53_static_v2=ns['_parse_glm53_static_v2'])
        normalize = ns['_static_v2_decode_config']
        production = md._parse_glm53_static_v2('t,r,sf6,batch')
        for field in ('scatter_vec4', 'scatter_packed_load'):
            configs = _pair_configs(md, **{field+'_only': True})
            for rows in (8, 16):
                control, candidate = [normalize(c, rows) for c in configs]
                changed = {name for name in control.keys() | candidate.keys()
                           if control.get(name) != candidate.get(name)}
                self.assertEqual(changed, {field})
                self.assertFalse(control[field])
                self.assertTrue(candidate[field])
                self.assertFalse(candidate['c2_direct_scatter'])
                if field == 'scatter_packed_load':
                    self.assertEqual(candidate == normalize(production, rows), rows == 8)
        with self.assertRaisesRegex(ValueError, 'one MoE'):
            _pair_configs(md, scatter_vec4_only=True, scatter_packed_load_only=True)


if __name__ == '__main__':
    unittest.main()
