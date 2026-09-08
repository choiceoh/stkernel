"""CPU checks for PDL eligibility and the read/publication boundaries."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OSAR = ROOT / 'overlay/modules/tp_oneshot_ar'
MK = ROOT / 'overlay/modules/glm53_megakernel'


def shim():
    spec = importlib.util.spec_from_file_location('ar_consumer_test_shim', OSAR / 'dsv4_oneshot_shim.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConsumerTests(unittest.TestCase):
    def test_exact_environment_gate(self):
        for flag, pdl, expected in [('1', '1', True), ('0', '1', False),
                                    ('1', '0', False), ('true', '1', False)]:
            with patch.dict('os.environ', {'VLLM_GLM53_AR_CONSUMER_PDL': flag,
                                           'VLLM_GLM53_MK_PDL': pdl}):
                self.assertEqual(shim()._CONSUMER_PDL, expected)

    def test_eligibility_and_capture_dispatch(self):
        m = shim()
        m._disabled = False
        m._connected = m._selftest_ok = m._CONSUMER_PDL = True
        m._SHADOW = False
        m._eligible = lambda t: t.eligible
        m._ext = SimpleNamespace(healthy=lambda: True,
                                oneshot_ar=lambda t: 'base',
                                oneshot_ar_consumer=lambda t: 'early')
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: True))
        with patch.dict(sys.modules, {'torch': fake_torch}):
            for elements, expected in [(4096, 'early'), (24576, 'early'),
                                       (32768, 'early'), (65536, 'base')]:
                t = SimpleNamespace(eligible=True, numel=lambda: elements)
                self.assertEqual(m.maybe_all_reduce(None, t, None), expected)
            self.assertTrue(m._CONSUMER_CAPTURED)
            t.eligible = False
            self.assertIsNone(m.maybe_all_reduce(None, t, None))
            t.eligible = True
            m._SHADOW = True
            self.assertIsNone(m.maybe_all_reduce(None, t, None))

    def test_runtime_failure_does_not_split_collectives(self):
        m = shim()
        m._disabled = False
        m._connected = m._selftest_ok = m._CONSUMER_PDL = True
        m._SHADOW = False
        m._eligible = lambda t: True
        m._ext = SimpleNamespace(healthy=lambda: False)
        with self.assertRaises(m.OneShotFatal):
            m.maybe_all_reduce(None, object(), None)

    def test_protocol_wait_and_publication_order(self):
        source = (OSAR / 'dsv4_oneshot_ar.cu').read_text()
        kernel = source.split('__global__ void k_oneshot(', 1)[1].split('// ----------------', 1)[0]
        wait = kernel.index('griddepcontrol.wait;')
        self.assertLess(wait, kernel.index('c->tx_seq'))
        release = kernel.index('griddepcontrol.launch_dependents;')
        self.assertLess(kernel.index('c->tx_seq = nxt;'), release)
        self.assertLess(release, kernel.index('while (left)'))
        self.assertIn('done_ctr, 1ULL) %\n               ARGRID == ARGRID - 1', kernel)
        self.assertIn('cfg.gridDim = dim3(ARGRID);', source)

    def test_mhc_weight_only_before_wait(self):
        source = (MK / 'glm53_megakernel.cu').read_text()
        body = source.split('__device__ void mk_mhc_p1_impl(', 1)[1].split('MK_MHC_TS(1);', 1)[0]
        self.assertIn('if constexpr (!AR_CONSUMER)\n      if (g < a.num_tokens) load_tok', body)
        wait = body.index('griddepcontrol.wait;')
        self.assertLess(body.index('fnr[m][j] = a.fn['), wait)
        self.assertLess(wait, body.index('int pend = -1'))
        self.assertIn('if (bid >= NCHUNK * groups)', body)
        self.assertIn('static int ar_grids[2]', source)
        self.assertIn('ar_consumer && mk_pdl_enabled() && a.num_tokens <= 8', source)

    def test_default_off_and_cache_identity(self):
        self.assertIn('\nVLLM_GLM53_AR_CONSUMER_PDL=0\n', (ROOT / 'profiles/glm53.env').read_text())
        dense = (ROOT / 'overlay/modules/glm53_model/glm53_fp8_dense.py').read_text()
        self.assertIn('_register_compile_factor(\n    "VLLM_GLM53_AR_CONSUMER_PDL",', dense)

    def test_probe_capture_is_not_serving_evidence(self):
        import ast
        tree = ast.parse((MK / 'glm53_megakernel.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_mhc_call')
        guarded = [n for n in function.body if isinstance(n, ast.If)
                   and 'AR consumer MHC CAPTURED' in ast.unparse(n)]
        self.assertEqual(len(guarded), 1)
        condition = ast.unparse(guarded[0].test)
        self.assertIn('_ar_consumer is None', condition)
        self.assertIn('is_current_stream_capturing()', condition)


if __name__ == '__main__':
    unittest.main()
