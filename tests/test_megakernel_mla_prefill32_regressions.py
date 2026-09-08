"""Execute the real dispatch function against shape-only tensors, without CUDA."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch


class BaselineSelected(Exception):
    pass


class Tensor:
    def __init__(self, shape, dtype, *, contiguous=True, element_size=1):
        self.shape, self.dtype, self.device = shape, dtype, "cuda"
        self.contiguous, self.size = contiguous, element_size

    def is_contiguous(self):
        return self.contiguous

    def element_size(self):
        return self.size


class DispatchTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "overlay/modules/glm53_megakernel/glm53_megakernel.py"
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "mla_decode")
        self.torch = types.SimpleNamespace(
            bfloat16="bf16", int32="i32", cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: False))
        def baseline(*args):
            raise BaselineSelected()
        self.ns = dict(MLA_H=16, MLA_D=512, ENABLE_MLA_PREFILL32=True,
                       ENABLE_MLA_PREFILL_PAIR=False, _ensure_workspace=baseline,
                       logger=types.SimpleNamespace(warning=lambda *a: None),
                       _mla_prefill32=lambda *a: "prefill32", _mla_prefill_pair=lambda *a: "pair")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), self.ns)

    def route(self, T=4096, W=2048, **kw):
        q = Tensor((T, 16, 512), kw.get("q_dtype", "bf16"))
        cache = Tensor((8192, 512), "u8", contiguous=kw.get("cache_contiguous", True),
                       element_size=kw.get("cache_element_size", 1))
        slots = Tensor((T, W), "i32")
        lens = Tensor((T,), "i32", contiguous=kw.get("lens_contiguous", True))
        with patch.dict(sys.modules, torch=self.torch):
            try:
                return self.ns["mla_decode"](q, cache, slots, lens, .044, .7)
            except BaselineSelected:
                return "baseline"

    def test_actual_prefill_chunks_and_bounds(self):
        for T in (4096, 4143, 6912, 8192):
            for W in (1, 17, 32, 33, 2048, 2176):
                with self.subTest(T=T, W=W):
                    self.assertEqual(self.route(T, W), "prefill32")

    def test_short_requests_and_outside_contract_stay_baseline(self):
        for T, W in ((1, 2048), (8, 2048), (32, 2048), (127, 2048),
                     (128, 2048), (2048, 2048), (2593, 2048), (4095, 2048),
                     (8193, 2048), (4096, 0), (4096, 2177)):
            with self.subTest(T=T, W=W):
                self.assertEqual(self.route(T, W), "baseline")

    def test_storage_guards(self):
        for kw in (dict(q_dtype="fp16"), dict(cache_contiguous=False),
                   dict(cache_element_size=2), dict(lens_contiguous=False)):
            with self.subTest(**kw):
                self.assertEqual(self.route(**kw), "baseline")

    def test_default_off(self):
        self.ns["ENABLE_MLA_PREFILL32"] = False
        self.assertEqual(self.route(), "baseline")

    def test_captured_decode_keeps_existing_path(self):
        self.torch.cuda.is_current_stream_capturing = lambda: True
        self.assertEqual(self.route(), "baseline")

    def test_pair_experiment_takes_precedence_without_combining(self):
        self.ns["ENABLE_MLA_PREFILL_PAIR"] = True
        self.assertEqual(self.route(), "pair")

    def test_serving_marker_follows_successful_candidate_call(self):
        events = []
        def launched(*args):
            events.append('launch')
            return 'prefill32'
        self.ns['_mla_prefill32'] = launched
        self.ns['logger'].warning = lambda *a: events.append('proof')
        self.assertEqual(self.route(), 'prefill32')
        self.assertEqual(self.route(), 'prefill32')
        self.assertEqual(events, ['launch', 'proof', 'launch'])
        def failed(*args):
            raise RuntimeError('candidate launch failed')
        self.ns['_mla_prefill32'] = failed
        events.clear()
        with self.assertRaises(RuntimeError):
            self.route()
        self.assertFalse(getattr(failed, '_announced', False))
        self.assertEqual(events, [])

    def test_arming_or_engaged_marker_is_not_launch_proof(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('mla32_proof_test', root/'bench/proof.py')
        proof = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proof)
        knob = 'VLLM_GLM53_MK_MLA_PREFILL32'
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'boot.log'
            log.write_text('[megakernel] mla prefill32 ENGAGED T=6912\nselftest mla prefill32=True -> ARM\n')
            self.assertIs(proof.check([knob], str(log))['proof'][knob], False)
            log.write_text('[megakernel] mla prefill32 LAUNCHED T=6912 W=2048\n')
            self.assertIs(proof.check([knob], str(log))['proof'][knob], True)


if __name__ == "__main__":
    unittest.main()
