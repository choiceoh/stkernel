"""Private M64 selection and scatter coverage; GPU proof remains separate."""
import ast
from collections import Counter
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class M64ContractTests(unittest.TestCase):
    def test_gpu_gate_checks_prefixes_routes_and_signed_zero_weights(self):
        path = ROOT / 'measurements/st_prefill_phase2_20260913/check_m64_prefill.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'route_samples')
        ns = dict(Counter=Counter, struct=struct)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-m64-route-gate>', 'exec'), ns)
        check = ns['route_samples']
        counts, bases = [8, 8] + [0] * 286, [0, 1] + [2] * 287
        tokens, weights = [-1] * 128, [0.] * 128
        tokens[:8], tokens[64:72] = [0] * 8, [1] * 8
        weights[:8], weights[64:72] = [-0.] * 8, [.125] * 8
        inputs, input_weights = [[0] * 8, [1] * 8], [[-0.] * 8, [.125] * 8]
        self.assertEqual(len(check(counts, bases, tokens, weights, inputs, input_weights)), 16)
        bad = bases.copy(); bad[1] = 0
        with self.assertRaises(AssertionError):
            check(counts, bad, tokens, weights, inputs, input_weights)
        bad_weights = weights.copy(); bad_weights[0] = 0.
        with self.assertRaises(AssertionError):
            check(counts, bases, tokens, bad_weights, inputs, input_weights)
        bad_tokens = tokens.copy(); bad_tokens[64] = 0
        with self.assertRaises(AssertionError):
            check(counts, bases, bad_tokens, weights, inputs, input_weights)

    def test_static_fork_is_reproducible_from_the_pinned_sources(self):
        path = ROOT / 'measurements/st_prefill_phase2_20260913/generate_m64_bodies.py'
        spec = importlib.util.spec_from_file_location('_m64_source_generator', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.build(), (ROOT / 'engine/kernels/b12x/_prefill_m64_bodies.py').read_text())

    def test_actual_q0_address_uses_separate_physical_atoms_for_m64_tiles(self):
        tree = ast.parse((ROOT / 'engine/kernels/b12x/_prefill_m64_bodies.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'route_scale_row_base' for t in n.targets))
        expression = compile(ast.Expression(assignment.value), '<actual-m64-q0-address>', 'eval')
        seen = set()
        for physical in range(193):
            base = eval(expression, dict(Int32=int, Uint32=int, phys_row=physical, num_k_tiles=64))
            atom, row = divmod(physical, 64)
            for block in range(256):
                address = base + (block // 4) * 512 + block % 4
                expected = atom * 32768 + (row % 32) * 16 + (row // 32) * 4 + (block // 4) * 512 + block % 4
                self.assertEqual(address, expected)
                self.assertNotIn(address, seen)
                seen.add(address)

    def test_eight_warps_cover_each_live_output_element_exactly_once(self):
        tree = ast.parse((ROOT / 'engine/kernels/b12x/moe_dynamic_prefill_m64.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                  and n.name == 'scatter_sC_to_gmem'))
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        writes = []
        ns = dict(Int32=int, load_shared_i32_f32_pair=lambda offset: ((offset // 8) * 7 % 64, offset // 8 + .5),
                  get_ptr_as_int64=lambda tensor, offset: offset,
                  get_smem_ptr_as_int32=lambda tensor, offset: offset,
                  scatter_add_weighted_bf16x8_to_f32=lambda *args: writes.append(args))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-m64-scatter>', 'exec'), ns)
        scatter = ns[node.name]
        shared = SimpleNamespace(layout=lambda coordinate: coordinate[0] * 128 + coordinate[1])
        for live in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64):
            writes.clear()
            for tid in range(256):
                scatter(None, tid, 2, live, shared, None, SimpleNamespace(shape=(64, 4096)), 0, 0, .125)
            coordinates = []
            for address, swizzled, weight, alpha in writes:
                # S<3,4,3> is self-inverse. Recover the logical source vector,
                # then independently check its permuted destination metadata.
                offset = swizzled ^ ((swizzled & 0x1C0) >> 3)
                row, col = divmod(offset, 128)
                self.assertEqual(address, (row * 7 % 64) * 4096 + 256 + col)
                self.assertEqual((weight, alpha), (row + .5, .125))
                coordinates.extend((row, col + i) for i in range(8))
            self.assertEqual(len(coordinates), live * 128)
            self.assertEqual(len(set(coordinates)), live * 128)
            self.assertEqual(set(coordinates), {(r, c) for r in range(live) for c in range(128)})

    def test_only_explicit_short_native_m64_geometry_is_eligible(self):
        ns = functions('engine/kernels/b12x/moe_dispatch.py',
                       ['_prefill_scale_expansion_eligible', '_prefill_m64_eligible'])
        select = ns['_prefill_m64_eligible']
        args = dict(m=2672, E=288, k=4096, n=512, num_topk=8, tile_m=64,
                    quant_mode='nvfp4', tiled=True, reform_sf_pack=True,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                    swiglu_limit=10., share_input_across_experts=False)
        for count in (65, 2672, 2675, 8192):
            self.assertTrue(select(**dict(args, m=count)))
        for change in ({'m': 7}, {'m': 28}, {'m': 64}, {'m': 8193}, {'m': True},
                       {'tile_m': 128}, {'E': 72}, {'k': 5120}, {'n': 2048},
                       {'num_topk': 4}, {'reform_sf_pack': False}, {'tiled': False},
                       {'share_input_across_experts': True}, {'swiglu_limit': 0.}):
            self.assertFalse(select(**dict(args, **change)), change)

    def test_compile_override_remains_explicit_and_launch_can_select_eager_candidate(self):
        tree = ast.parse((ROOT / 'engine/kernels/b12x/moe_dispatch.py').read_text())
        for name in ('_get_dynamic_kernel', 'launch_sm120_dynamic_moe'):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            defaults = dict(zip((a.arg for a in node.args.kwonlyargs), node.args.kw_defaults))
            expected = False if name == '_get_dynamic_kernel' else None
            self.assertIs(ast.literal_eval(defaults['_prefill_tile64']), expected)

    def test_actual_automatic_selector_preserves_decode_capture_controls_and_long_prefill(self):
        tree = ast.parse((ROOT / 'engine/kernels/b12x/moe_dispatch.py').read_text())
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('_prefill_scale_expansion_eligible', '_prefill_m64_eligible')]
        launch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_sm120_dynamic_moe')
        branch = next(n for n in launch.body if isinstance(n, ast.If) and ast.unparse(n.test) == '_prefill_tile64 is None')
        code = compile(ast.Module(body=helpers + [branch], type_ignores=[]), '<actual-auto-m64>', 'exec')
        def selected(**changes):
            calls = []
            ns = dict(_prefill_tile64=None, _TP_SF6_Q0_ENABLED=True, _tp_sf6_q0_override=None,
                      _prefill_scale_expansion=None, workspace=SimpleNamespace(tile_m=128),
                      num_tokens=2672, num_experts=288, k=4096, n=512, top_k=8,
                      quant_mode='nvfp4', weights=SimpleNamespace(tiled=True), direct_sf6=True,
                      activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                      swiglu_limit=10., input_gs_is_shared=False,
                      _prefill_m64_workspace=lambda ws, rows: calls.append(rows) or SimpleNamespace(tile_m=64))
            capturing = changes.pop('capturing', False)
            ns['torch'] = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing))
            ns.update(changes)
            exec(code, ns)
            return ns['_prefill_tile64'], calls
        for rows in (65, 2121, 2672, 2675, 4096):
            self.assertEqual(selected(num_tokens=rows), (True, [rows]))
        for change in ({'num_tokens': 7}, {'num_tokens': 28}, {'num_tokens': 64},
                       {'num_tokens': 4097}, {'num_tokens': 8192}, {'num_tokens': 32256},
                       {'direct_sf6': False}, {'capturing': True}, {'input_gs_is_shared': True},
                       {'_tp_sf6_q0_override': True}, {'_tp_sf6_q0_override': False},
                       {'_prefill_scale_expansion': False}, {'_prefill_scale_expansion': True},
                       {'_prefill_tile64': False}):
            self.assertEqual(selected(**change), (False, []), change)


if __name__ == '__main__':
    unittest.main()
