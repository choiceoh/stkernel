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
SPEC=importlib.util.spec_from_file_location('moe_m64_probe',ROOT/'probes/glm53_moe_m64_check.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

class ProbeTests(unittest.TestCase):
    def test_actual_frozen_distributed_signatures_accept_probe_calls(self):
        self.assertEqual(len(m.validate_distributed_api()),5)

    def test_original_keyword_typo_is_rejected_without_importing_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);probe=root/'probes/glm53_moe_m64_check.py'
            api=root/'overlay/modules/glm53_runtime/parallel_state.py'
            probe.parent.mkdir(parents=True);api.parent.mkdir(parents=True)
            probe.write_text((ROOT/'probes/glm53_moe_m64_check.py').read_text().replace(
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
