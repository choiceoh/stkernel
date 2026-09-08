"""CPU checks for PDL eligibility and the read/publication boundaries."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import shutil
import subprocess
import tempfile
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


def driver():
    spec = importlib.util.spec_from_file_location('ar_consumer_test_driver', MK / 'glm53_megakernel.py')
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
        kernel = source.split('__device__ __forceinline__ void k_oneshot_impl(', 1)[1].split('// ----------------', 1)[0]
        wait = kernel.index('griddepcontrol.wait;')
        self.assertLess(wait, kernel.index('c->tx_seq'))
        release = kernel.index('griddepcontrol.launch_dependents;')
        self.assertLess(kernel.index('c->tx_seq = nxt;'), release)
        self.assertLess(release, kernel.index('while (left)'))
        self.assertIn('done_ctr, 1ULL) %\n               ARGRID == ARGRID - 1', kernel)
        self.assertIn('cfg.gridDim = dim3(ARGRID);', source)

    def test_collective_ownership_covers_actual_vector_and_tail_accesses(self):
        source = (OSAR / 'dsv4_oneshot_ar.cu').read_text()
        start = source.index('template <bool VECTOR_EXACT>')
        end = source.index('\n}', start) + 2
        helper = source[start:end].replace('__host__ __device__ ', '')
        self.assertIn('osar_block_owns<CONSUMER_PDL>(blockIdx.x, blockDim.x, n)', source)
        compiler = shutil.which('c++')
        self.assertIsNotNone(compiler, 'host C++ compiler required for the ownership oracle')
        # Derive owners by visiting the actual grid-stride vector positions,
        # not by copying the launch mask. Grow the payload one BF16 at a time
        # through MAXEL, retaining the visited vectors and enumerating tails.
        oracle = r'''
#include <cstdio>
int main() {
  constexpr int blocks = 48, threads = 256, limit = 131072;
  bool vector_owners[blocks] = {};
  for (int n = 0; n <= limit; ++n) {
    if (n && n % 8 == 0) {
      int vector = n / 8 - 1;
      int lane = vector % (blocks * threads);
      vector_owners[lane / threads] = true;
    }
    bool accessed[blocks];
    for (int b = 0; b < blocks; ++b) accessed[b] = vector_owners[b];
    for (int i = n / 8 * 8; i < n; ++i) {
      int lane = (i - n / 8 * 8) % (blocks * threads);
      accessed[lane / threads] = true;
    }
    for (int b = 0; b < blocks; ++b) {
      bool expected = accessed[b] || b == 0;
      bool candidate = osar_block_owns<true>(b, threads, n);
      bool ordinary = osar_block_owns<false>(b, threads, n);
      if (candidate != expected || (expected && !ordinary)) {
        std::fprintf(stderr, "ownership mismatch n=%d block=%d\n", n, b);
        return 1;
      }
    }
  }
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ownership.cc'
            binary = Path(tmp) / 'ownership'
            path.write_text(helper + '\n' + oracle)
            subprocess.run([compiler, '-std=c++17', '-O2', str(path), '-o', str(binary)],
                           check=True, capture_output=True, text=True, timeout=60)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)

    def test_bf16_layout_preserves_every_bit_pattern(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires Torch CPU tensors in the serving image')
        m = driver()
        # Include all BF16 encodings, signed zeros and NaN payloads. Packing
        # is a byte-preserving permutation, independent of the finite gate.
        bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).repeat(6)
        original = bits.view(torch.bfloat16).view(24, 16384)
        packed = m._mhc_bf16_vec4(original)
        self.assertEqual(tuple(packed.shape), (24, 4096, 4))
        self.assertTrue(packed.is_contiguous())
        flat = packed.view(torch.int16).flatten()
        # The CUDA uint2 at m * HIDDEN + h must contain streams 0,1,2,3.
        for output in (0, 7, 23):
            for h in (0, 31, 32, 255, 256, 4095):
                address = (output * 4096 + h) * 4
                expected = [int(bits[output * 16384 + j * 4096 + h]) for j in range(4)]
                self.assertEqual(flat[address:address + 4].tolist(), expected)
        restored = packed.transpose(1, 2).contiguous().view(torch.int16).flatten()
        self.assertTrue(torch.equal(restored, bits))

    def test_layout_cache_preserves_graph_storage_and_versions(self):
        m = driver()
        m.ENABLE_AR_CONSUMER = False
        capture = [False]
        class Pack:
            def float(self): return self
            def view(self, *_): return self
        fn = SimpleNamespace(dtype='fp32', is_cuda=True, shape=(24, 16384),
                             is_contiguous=lambda: True, _version=0, device='cuda:0',
                             data_ptr=lambda: 1234, to=lambda _: Pack(), view=lambda _: None)
        fake_torch = SimpleNamespace(float32='fp32', bfloat16='bf16', int32='int32',
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capture[0]),
            isfinite=lambda _: SimpleNamespace(all=lambda: True), equal=lambda *_: True)
        m._mhc_bf16_vec4 = lambda _: Pack()
        with patch.dict(sys.modules, {'torch': fake_torch}):
            scalar = m._mhc_bf16_weight(fn)
            capture[0] = True
            self.assertIsNone(m._mhc_bf16_weight(fn, ar_consumer=True))
            self.assertIs(m._mhc_bf16_weight(fn), scalar)
            capture[0] = False
            vector = m._mhc_bf16_weight(fn, ar_consumer=True)
            self.assertIsNot(vector, scalar)
            self.assertEqual(len(m._MHC_BF16_CACHE), 1)
            capture[0] = True
            self.assertIs(m._mhc_bf16_weight(fn, ar_consumer=True), vector)
            self.assertIs(m._mhc_bf16_weight(fn), scalar)
            capture[0] = False
            fn._version += 1
            changed = m._mhc_bf16_weight(fn, ar_consumer=True)
            self.assertIsNot(changed, vector)
            retained = list(m._MHC_BF16_CACHE.values())[0]
            self.assertIs(retained[1], scalar)
            self.assertIs(retained[2], vector)
            m._MHC_BF16_CACHE_LIMIT = 2
            fn._version += 1
            self.assertIsNone(m._mhc_bf16_weight(fn, ar_consumer=True))

    def test_large_warmup_prepares_first_small_capture_without_repacking(self):
        # Reproduce serving: T=12 visits the weight eagerly, then the graph
        # ladder captures T=12 and T=6 without an eager T=6 weight visit.
        for enabled, exact in ((False, True), (True, True), (True, False)):
            with self.subTest(enabled=enabled, exact=exact):
                m = driver()
                m.ENABLE_AR_CONSUMER = m._AR_CONSUMER_OK = enabled
                m.ENABLE_MHC_BF16 = m._MHC_BF16_OK = True
                m._ARMED['mhc'] = True
                capture = [False]
                packed_layouts = []
                launches = []

                class Tensor:
                    dtype, device, is_cuda, _version = 'fp32', 'cuda:0', True, 0
                    def __init__(self, *shape): self.shape = shape
                    def data_ptr(self): return id(self)
                    def is_contiguous(self): return True
                    def float(self): return self
                    def view(self, *_): return self
                    def to(self, _):
                        self_test.assertFalse(capture[0], 'packing during capture')
                        return Tensor(*self.shape)

                self_test = self
                def vector_pack(_):
                    self.assertFalse(capture[0], 'vector packing during capture')
                    pack = Tensor(24, 4096, 4)
                    packed_layouts.append(pack)
                    return pack
                m._mhc_bf16_vec4 = vector_pack
                m._ar_note = lambda _: None
                workspace = {key: Tensor() for key in ('yp', 'rp', 'sq', 'pmix', 'ol_stash', 'barrier_mhc')}
                m._ensure_workspace = lambda _: workspace
                m._EXT = SimpleNamespace(run_mhc=lambda ptrs, scalars, ints, bf16, early:
                    launches.append((ints[0], ptrs[4], bf16, early)))
                fake_torch = SimpleNamespace(float32='fp32', bfloat16='bf16', int32='int32',
                    cuda=SimpleNamespace(is_current_stream_capturing=lambda: capture[0]),
                    empty=lambda *shape, **_: Tensor(*shape), empty_like=lambda t: Tensor(*t.shape),
                    isfinite=lambda _: SimpleNamespace(all=lambda: True), equal=lambda *_: exact)
                fn = Tensor(24, 16384)
                def call(t):
                    return m._mhc_call(Tensor(t, 4096), Tensor(t, 4, 4096), Tensor(t, 4),
                        Tensor(t, 16), fn, Tensor(3), Tensor(24), Tensor(4096),
                        t, 1e-6, 1e-6, 1e-6, 1., 1e-6, 20)

                with patch.dict(sys.modules, {'torch': fake_torch}):
                    call(12)
                    entry = next(iter(m._MHC_BF16_CACHE.values()))
                    self.assertEqual(len(packed_layouts), int(enabled and exact))
                    capture[0] = True
                    call(12)
                    call(6)
                    self.assertEqual(launches[0], launches[1])
                    expected = entry[2] if enabled else entry[1]
                    self.assertEqual(launches[2], (6, (expected if exact else fn).data_ptr(), exact, enabled))
                    self.assertEqual(len(packed_layouts), int(enabled and exact))
                    self.assertEqual(len(m._MHC_BF16_CACHE), 1)

    def test_standalone_pre_packs_the_real_three_dimensional_weight_view(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires Torch CPU tensors in the serving image')
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                m = driver()
                m.ENABLE_MHC_BF16 = m._MHC_BF16_OK = True
                m.ENABLE_AR_CONSUMER = m._AR_CONSUMER_OK = enabled
                m._ARMED['mhc_pre'] = True
                m._ARMED['mhc'] = False
                captured = [False]
                seen, launches = [], []
                m._ar_note = seen.append
                workspace = {key: torch.empty(1) for key in ('yp', 'rp', 'sq', 'pmix', 'ol_stash', 'barrier_mhc')}
                m._ensure_workspace = lambda _: workspace
                m._EXT = SimpleNamespace(run_mhc=lambda ptrs, scalars, ints, bf16, early:
                    launches.append((ints[0], ptrs[4], bf16, early)))
                flat = torch.randn(24, 16384).bfloat16().float()
                view = flat.view(24, 4, 4096)
                scale, base = torch.ones(3), torch.zeros(24)
                norm = torch.ones(4096, dtype=torch.bfloat16)
                def call(t):
                    return m.mhc_pre_only(torch.zeros(t, 4, 4096, dtype=torch.bfloat16),
                        view, scale, base, 1e-6, 1e-6, 1e-6, 1., 20, norm, 1e-6)
                # Keep real CPU storage/view/version behavior; replace only
                # device eligibility and the native launch for this CPU test.
                with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)), \
                     patch.object(torch.cuda, 'is_current_stream_capturing', lambda: captured[0]):
                    call(12)
                    self.assertEqual(seen[-1].dtype, torch.bfloat16)
                    self.assertEqual(tuple(seen[-1].shape), (24, 16384))
                    captured[0] = True
                    call(6)
                    selected = seen[-1]
                    self.assertEqual(selected.dtype, torch.bfloat16)
                    self.assertEqual(tuple(selected.shape), (24, 4096, 4) if enabled else (24, 16384))
                    self.assertEqual(launches[-1], (6, selected.data_ptr(), True, enabled))
                    self.assertEqual(len(m._MHC_BF16_CACHE), 1)
                    self.assertEqual(view.data_ptr(), flat.data_ptr())
                    self.assertTrue(torch.equal(view.reshape_as(flat), flat))

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
        for marker in ('AR consumer MHC CAPTURED', 'mhc-bf16 CAPTURED'):
            guarded = [n for n in function.body if isinstance(n, ast.If)
                       and marker in ast.unparse(n)]
            self.assertEqual(len(guarded), 1)
            condition = ast.unparse(guarded[0].test)
            self.assertIn('_ar_consumer is None', condition)
            self.assertIn('is_current_stream_capturing()', condition)


if __name__ == '__main__':
    unittest.main()
