"""Head-log proof and real profile/model wiring; no serving or GPU requests."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
KNOB = 'VLLM_GLM53_EP_TILED'
PREFIX = '[ep-tiled-selftest] PASS '
spec = importlib.util.spec_from_file_location('ep_tiled_proof_cpu', ROOT / 'bench/proof.py')
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)
CASES = (('mixed6', 6), ('balanced12', 12), ('concentrated24', 24), ('zeros32', 32),
         ('remote33', 33), ('balanced2128', 2128), ('balanced4096', 4096),
         ('concentrated6912', 6912), ('balanced8192', 8192))


def receipt():
    result = dict(schema=1, verdict='PASS', phase='complete', caller_preserved=True,
                  actual_weight_owner=True, geometry=dict(E=72, K=4096, I=2048, top8=8), cases=[])
    for name, rows in CASES:
        graph = rows <= 33 or rows == 8192
        labels = ('C1-eager', 'C2-graph-current' if graph else 'C2-eager',
                  'C3-graph-side' if graph else 'C3-side')
        result['cases'].append(dict(case=name, rows=rows, verdict='PASS', phase='complete',
            graph_replay=graph, candidate=[dict(phase=phase + '-' + label, bad_rows=0,
                max_row_relative_l2=0., max_row_relative_abs=0.,
                stock_max_row_relative_l2=0., stock_max_row_relative_abs=0.)
                for phase in ('initial', 'changed') for label in labels]))
    return result


def log(record=None, *, decode=6, prefill=8192):
    return '\n'.join((PREFIX + json.dumps(receipt() if record is None else record),
        f'[ep-tiled] LAUNCHED decode E72/H4096/I2048/top8 T={decode}',
        f'[ep-tiled] LAUNCHED prefill E72/H4096/I2048/top8 T={prefill}'))


class CompositeProofTests(unittest.TestCase):
    def test_complete_actual_weight_canary_and_both_lanes_prove_at_boundaries(self):
        for decode in (1, 6, 32):
            for prefill in (33, 2128, 8192, 16384):
                self.assertIs(proof._startup_proof(KNOB, log(decode=decode, prefill=prefill)), True)

    def test_compile_armed_canary_only_and_either_missing_lane_never_prove(self):
        complete = log().splitlines()
        for text in ('CPU compile PASS\n165 tests PASS', KNOB + '=1\n[ep-tiled] armed',
                     complete[0], '\n'.join(complete[1:]),
                     '\n'.join(complete[:2]), '\n'.join((complete[0], complete[2]))):
            with self.subTest(text=text[:80]):
                self.assertIs(proof._startup_proof(KNOB, text), False)

    def test_fail_before_or_after_pass_is_not_hidden_by_later_success(self):
        failure = '[ep-tiled-selftest] FAIL {"verdict":"FAIL","error":"numerics"}'
        for text in (failure + '\n' + log(), log() + '\n' + failure,
                     log() + '\n' + failure + '\n' + log()):
            self.assertIs(proof._startup_proof(KNOB, text), False)

    def test_wrong_lane_geometry_lengths_or_malformed_actual_marker_fail(self):
        for decode, prefill in ((0, 8192), (33, 8192), (6, 32), (6, 16385),
                                ('6 trailing', 8192), ('6.0', 8192), (6, '-1')):
            self.assertIs(proof._startup_proof(KNOB, log(decode=decode, prefill=prefill)), False)
        for old, new in (('decode E72', 'decode E288'), ('prefill E72', 'prefill E288'),
                         ('I2048/top8', 'I512/top8'), ('LAUNCHED', 'armed')):
            self.assertIs(proof._startup_proof(KNOB, log().replace(old, new)), False)

    def test_partial_case_or_caller_failure_never_proves(self):
        mutations = [lambda r: r.update(verdict='RUNNING'), lambda r: r.update(phase='cleanup'),
                     lambda r: r.update(caller_preserved=False), lambda r: r.update(error=None),
                     lambda r: r.update(cleanup_error='failed'), lambda r: r['cases'].pop(),
                     lambda r: r['cases'].__setitem__(1, copy.deepcopy(r['cases'][0])),
                     lambda r: r['cases'][0].update(verdict='FAIL'),
                     lambda r: r['cases'][0].update(phase='changed-C3-graph-side')]
        for mutate in mutations:
            record = receipt(); mutate(record)
            self.assertIs(proof._startup_proof(KNOB, log(record)), False)

    def test_candidate_shape_phase_or_numerical_failure_cannot_hide_in_pass_envelope(self):
        for candidate in (None, 'abcdef', [None] * 6, [], [{}] * 6):
            record = receipt(); record['cases'][0]['candidate'] = candidate
            with self.subTest(candidate=candidate):
                self.assertIs(proof._startup_proof(KNOB, log(record)), False)
        mutations = [lambda c: c['candidate'][0].update(bad_rows=1),
                     lambda c: c['candidate'][0].update(bad_rows=False),
                     lambda c: c['candidate'][0].update(phase='changed-C1-eager'),
                     lambda c: c['candidate'][0].update(error='failed'),
                     lambda c: c.update(diagnostic_error='copy failed'),
                     lambda c: c.update(first_failure_rows={}),
                     lambda c: c.update(cleanup_error='failed')]
        for mutate in mutations:
            record = receipt(); mutate(record['cases'][0])
            self.assertIs(proof._startup_proof(KNOB, log(record)), False)

    def test_malformed_duplicate_and_nonfinite_json_fail_without_raising(self):
        lanes = '\n'.join(log().splitlines()[1:])
        for value in ('{', '[]', 'null', '{"verdict":"FAIL","verdict":"PASS"}',
                      '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            self.assertIs(proof._startup_proof(KNOB, PREFIX + value + '\n' + lanes), False)
        invalid = log().replace('"bad_rows": 0', '"bad_rows": NaN', 1)
        self.assertIs(proof._startup_proof(KNOB, invalid), False)
        for name in ([], {}):
            record = receipt(); record['cases'][0]['case'] = name
            self.assertIs(proof._startup_proof(KNOB, log(record)), False)

    def test_public_check_cannot_use_the_single_tsv_marker_as_composite_waiver(self):
        table = proof.markers()
        self.assertIn(KNOB, table)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.log'
            path.write_text(table[KNOB][0] + '8192\n')
            result = proof.check([KNOB], str(path), table)
            self.assertEqual(result['proof'], {KNOB: False})
            self.assertEqual(result['proof_ok'], '0/1')
            path.write_text(log())
            result = proof.check([KNOB], str(path), table)
            self.assertEqual(result['proof'], {KNOB: True})
            self.assertEqual(result['proof_ok'], '1/1')


class WiringTests(unittest.TestCase):
    def test_real_profile_loader_keeps_default_off_and_accepts_only_new_candidate_flags(self):
        bash = shutil.which('bash')
        if bash is None: raise RuntimeError('Bash is required for the real profile loader')
        keys = ('ENABLE_EP', KNOB, 'VLLM_GLM53_EP_PREFILL_LOCAL',
                'VLLM_B12X_EP_ZERO_WEIGHT_MICRO', 'VLLM_B12X_EP_WARM_COMPACT')
        script = '''set -euo pipefail
source "$1"
ct_load_profile "$2" ENABLE_EP
shift 2
for key in "$@"; do printf '%s=%s\\n' "$key" "${!key-}"; done
'''
        for overrides in ({}, {'ENABLE_EP': '1', KNOB: '1'}):
            result = subprocess.run([bash, '--noprofile', '--norc', '-c', script, 'ep-tiled-profile',
                str(ROOT / 'launchers/lib/common-tp4.sh'), str(ROOT / 'profiles/glm53.env'), *keys],
                env={'PATH': os.defpath, 'LC_ALL': 'C', **overrides},
                text=True, capture_output=True, check=True, timeout=10)
            actual = dict(line.split('=', 1) for line in result.stdout.splitlines())
            self.assertEqual(actual, {**dict.fromkeys(keys, '0'), **overrides})

    def test_new_flag_alone_retains_the_existing_ep_mhc_single_reduction_guard(self):
        path = ROOT / 'overlay/modules/glm53_model/glm5next_model.py'
        tree = ast.parse(path.read_text())
        setting = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == '_EP_PREFILL_LOCAL' for t in n.targets))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_prefill_sp_layer_reduction_ok')
        module = ast.Module(body=[setting, function], type_ignores=[])
        config = SimpleNamespace(tp_size=1, ep_size=4, dp_size=1, is_sequence_parallel=False,
                                 skip_final_all_reduce=False, moe_backend='flashinfer_b12x')
        runner = SimpleNamespace(moe_config=config, routed_input_transform=None,
            routed_output_transform=None, _fused_output_is_reduced=False, router=object())
        layer = SimpleNamespace(mhc=True, is_mtp_layer=False, is_sequence_parallel=False,
            _mlp_is_moe=True, self_attn=SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True)),
            mlp=SimpleNamespace(experts=runner, shared_experts=SimpleNamespace(
                down_proj=SimpleNamespace(reduce_results=False))))
        for old, new in (('0', '0'), ('1', '0'), ('0', '1')):
            ns = {'os': SimpleNamespace(environ={'VLLM_GLM53_EP_PREFILL_LOCAL': old, KNOB: new})}
            exec(compile(module, str(path), 'exec'), ns)
            gate = ns[function.name]
            self.assertEqual(gate(layer), old == '1' or new == '1')
            if new == '1':
                runner._fused_output_is_reduced = True
                self.assertFalse(gate(layer))
                runner._fused_output_is_reduced = False
                config.dp_size = 2
                self.assertFalse(gate(layer))
                config.dp_size = 1

    def test_owner_decode_canary_and_prefill_sources_are_registered_once(self):
        folder = ROOT / 'overlay/modules/glm53_moe'
        rows = [line.split('\t') for line in (folder / 'manifest.tsv').read_text().splitlines()
                if line and not line.startswith('#')]
        for name in ('glm53_ep_tiled.py', 'moe_static_ep_tiled.py',
                     'glm53_ep_tiled_selftest.py', 'moe_dynamic_ep_local.py', 'glm53_ep_route_remap.py'):
            matched = [row for row in rows if row[0] == name]
            self.assertEqual(len(matched), 1)
            self.assertTrue((folder / name).is_file())
            self.assertEqual(matched[0][1], 'flashinfer/fused_moe/cute_dsl/blackwell_sm12x/' + name)
            self.assertEqual(matched[0][2], 'absent')


if __name__ == '__main__':
    unittest.main()
