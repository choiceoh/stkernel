"""CPU-only API binding and partial-initialization cleanup regressions."""
import importlib.util
import ast
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('moe_overlap_probe',ROOT/'probes/glm53_moe_overlap_check.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
DIAG=importlib.util.spec_from_file_location('moe_overlap_diagnostics',ROOT/'probes/glm53_moe_overlap_diagnostics.py')
d=importlib.util.module_from_spec(DIAG);DIAG.loader.exec_module(d)

class ProbeTests(unittest.TestCase):
    def test_diagnostic_keeps_per_row_limits_and_records_control_failures(self):
        result=d.error_summary([.02,.006,.031],[0,.003,.01],[.04,.065,.03],[0,.001,.02])
        self.assertEqual(result['bad_row_indices'],[1,2])
        self.assertEqual(result['first_bad_rows'][0]['peak_limit'],.04)
        self.assertEqual(result['first_bad_rows'][1]['l2_limit'],.03)
        # More noise in another row must never increase this row's limit.
        result=d.error_summary([0,.006],[1,0],[0,.065],[1,0])
        self.assertEqual(result['bad_row_indices'],[1])
        # Use limits computed in device precision, including an exact boundary.
        result=d.error_summary([.02],[0],[.04],[0],limits=([.019999999],[.04]))
        self.assertEqual(result['bad_row_indices'],[0])
        invalid=d.error_summary([float('nan')],[0],[0],[0])
        self.assertFalse(invalid['finite'])
        self.assertEqual(invalid['bad_rows'],1)
        json.dumps(invalid,allow_nan=False)

    def test_real_count_tile_budget_exposes_extra_stripe_padding(self):
        full=d.routed_tile_budget([129,127]+[0]*286,128)
        left=d.routed_tile_budget([65,64]+[0]*286,128)
        right=d.routed_tile_budget([64,63]+[0]*286,128)
        self.assertEqual(full['physical_tiles'],3)
        self.assertEqual(left['physical_tiles']+right['physical_tiles'],4)
        self.assertEqual(full['routed_rows'],left['routed_rows']+right['routed_rows'])
        with self.assertRaises(ValueError):d.routed_tile_budget([1]*287,128)
        with self.assertRaises(ValueError):d.routed_tile_budget([1]*288,0)

    def test_three_arm_timings_balance_order_and_position(self):
        self.assertEqual(set(d.timing_orders()),set(itertools.permutations(range(3))))
        for pos in range(3):
            self.assertEqual([sum(o[pos]==arm for o in d.timing_orders()) for arm in range(3)],[2]*3)

    def test_diagnostic_cannot_emit_serving_gate_pass(self):
        record=d.diagnostic_record(transport='fp8-v3',rows=4143,skew=True,phase='whole_path',ranks=[{}]*4)
        self.assertIs(record['serving_gate'],False)
        self.assertEqual(record['activation_seed'],13354)
        with self.assertRaises(ValueError):d.diagnostic_record(transport='bf16',rows=4143,skew=True,phase='whole_path',ranks=[{}])
        tree=ast.parse((ROOT/'probes/glm53_moe_overlap_check.py').read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.diagnose')
        self.assertIsInstance(branch.body[-1],ast.Return)
        self.assertNotIn('MOE_OVERLAP_GPU_PASS',(ROOT/'probes/glm53_moe_overlap_diagnostics.py').read_text())
        shell=(ROOT/'probes/run_glm53_moe_overlap_tp4_check.sh').read_text()
        self.assertIn('if [[ ${#probe_args[@]} == 0 ]]; then\n  echo MOE_OVERLAP_ALL_GATES_PASS\nelse\n  echo MOE_OVERLAP_DIAGNOSTIC_COMPLETE',shell)

    def test_actual_frozen_distributed_signatures_accept_probe_calls(self):
        self.assertEqual(len(m.validate_distributed_api()),5)

    def test_original_keyword_typo_is_rejected_without_importing_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);probe=root/'probes/glm53_moe_overlap_check.py'
            api=root/'overlay/modules/glm53_runtime/parallel_state.py'
            probe.parent.mkdir(parents=True);api.parent.mkdir(parents=True)
            probe.write_text((ROOT/'probes/glm53_moe_overlap_check.py').read_text().replace(
                'pipeline_model_parallel_size=1','pipeline_parallel_size=1'))
            api.write_bytes((ROOT/'overlay/modules/glm53_runtime/parallel_state.py').read_bytes())
            with patch.object(m,'__file__',str(probe)),self.assertRaisesRegex(TypeError,'pipeline_parallel_size'):
                m.validate_distributed_api()

    def test_world_cleanup_runs_after_model_init_body_and_teardown_failures(self):
        for failure in (None,'world','model','body','destroy_model'):
            calls=[]
            def call(name):
                def run(*args,**kw):
                    calls.append(name)
                    if name==failure:raise RuntimeError(name)
                return run
            ps=SimpleNamespace(set_custom_all_reduce=call('custom'),
                init_distributed_environment=call('world'),initialize_model_parallel=call('model'),
                destroy_model_parallel=call('destroy_model'),destroy_distributed_environment=call('destroy_world'))
            def run():
                with m.distributed_probe(ps,rank=0,local_rank=0):
                    calls.append('body')
                    if failure=='body':raise RuntimeError('body')
            if failure:
                with self.subTest(failure=failure),self.assertRaisesRegex(RuntimeError,failure):run()
            else:run()
            self.assertEqual(calls[-2:],['destroy_model','destroy_world'])
            if failure in ('world','model'):self.assertNotIn('body',calls)

if __name__=='__main__':unittest.main()
