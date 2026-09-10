"""CPU instance/lifecycle tests with pinned-function-shaped portable fixtures.

The independent diff probe executes the full official source AST. This fixture
isolates install/rollback, genuine buffer release, publication and reset rules.
"""
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay/modules/dsv41_model"))
import torch
import dsv41_reference_adapter as old
import dsv41_packed_reference_adapter as adapter
import dsv41_packed_index as packed

spec = importlib.util.spec_from_file_location("compact_fixture", ROOT / "tests/test_dsv41_reference_adapter.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
KERNEL_FIXTURE = '''import torch

def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor:
    """FP4 with E8M0 scales for the indexer or E4M3 scales for compressed KV.
    inplace=True writes the dequantized values back to x."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    tl_dtype = FP8 if scale_dtype == torch.float8_e4m3fn else FE8M0
    kernel = fp4_quant_kernel(N, block_size, scale_dtype=tl_dtype, inplace=inplace)
    kernel(z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s
'''


class PackedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ref = ModuleType("packed_reference_fixture")
        self.kernel = ModuleType("packed_kernel_fixture")
        for module, filename, source in ((self.ref, "model.py", fixture.REFERENCE_FIXTURE),
                                         (self.kernel, "kernel.py", KERNEL_FIXTURE)):
            module.__file__ = str(Path(self.temp.name) / filename)
            Path(module.__file__).write_text(source)
            exec(compile(source, module.__file__, "exec"), vars(module))
        self.events, self.reductions = [], []
        self.kernel.FP8, self.kernel.FE8M0 = "fp8", "e8m0"
        self.kernel.fp4_quant_kernel = self.quant_kernel
        self.ref.fp4_act_quant = self.kernel.fp4_act_quant
        self.ref.fp4_block_size = 32
        self.ref.world_size = 1
        self.ref.shared_attn = SimpleNamespace(index_k=None, candidates=None, topk_idxs=None)
        self.ref.dist = SimpleNamespace(all_reduce=lambda s: self.reductions.append(tuple(s.shape)))
        self.ref.apply_rotary_emb = lambda x, f: self.events.append(("rope", tuple(x.shape)))
        self.model = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(indexer=None)) for _ in range(40)])
        for layer in adapter.LAYERS:
            obj = self.ref.Indexer()
            attrs = dict(compress_ratio=2 if layer < 20 else 1, dim=5120, n_heads=32,
                n_local_heads=32, index_head_dim=128, rope_head_dim=64, index_topk=512,
                q_lora_rank=1280, candidate_topk_blocks=2048, candidate_block_size=8,
                owns_k=layer in adapter.OWNERS, is_candidate_source=layer == 20,
                uses_candidates=layer > 20, softmax_scale=128**-.5)
            for k, v in attrs.items():
                setattr(obj, k, v)
            obj.freqs_cis = torch.ones(64, 32)
            obj.wq_b = lambda qr, o=obj: torch.ones((*qr.shape[:2], o.n_local_heads * 128), dtype=torch.bfloat16)
            obj.weights_proj = lambda x, o=obj: torch.ones((*x.shape[:2], o.n_local_heads), dtype=torch.bfloat16)
            obj.wk, obj.k_norm = lambda z: z[..., :128], lambda z: z
            if layer in adapter.OWNERS:
                obj.register_buffer("k_cache", torch.zeros(2, 64 // obj.compress_ratio, 128,
                                                          dtype=torch.bfloat16), persistent=False)
            self.model.layers[layer].attn.indexer = obj
        self.pins = [patch.object(old, "REFERENCE_SHA256", hashlib.sha256(Path(self.ref.__file__).read_bytes()).hexdigest()),
                     patch.object(adapter, "KERNEL_SHA256", hashlib.sha256(Path(self.kernel.__file__).read_bytes()).hexdigest())]
        for pin in self.pins:
            pin.start()
            self.addCleanup(pin.stop)
        self.addCleanup(lambda: old._ACTIVE.pop(self.ref, None))
        self.handle = None

    def quant_kernel(self, n, block, *, scale_dtype, inplace):
        self.events.append(("quant", inplace, scale_dtype))
        def run(z, y, sf):
            # Test values are exactly in E2M1 at scale1; actual rounding is
            # separately checked by the official-AST arithmetic probe.
            lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
            codes = (z.float().abs().unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)
            codes |= z.signbit().to(torch.uint8) * 8
            sf.view(torch.uint8).fill_(127)
            if inplace:
                y.copy_(lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.))
            else:
                y.view(torch.uint8).copy_(codes[:, ::2] | (codes[:, 1::2] << 4))
        return run

    def obj(self, layer):
        return self.model.layers[layer].attn.indexer

    def install(self, **kwargs):
        self.handle = adapter.install_reference_packed_indexer(self.model, self.ref, enabled=True, **kwargs)
        return self.handle

    def args(self, layer, start, queries=1, batch=1):
        ratio = self.obj(layer).compress_ratio
        latent = None
        if layer in adapter.OWNERS and (start == 0 or (start + 1) % ratio == 0):
            rows = queries // ratio if start == 0 else 1
            latent = torch.full((batch, rows, 128), float(1 + adapter.OWNERS.index(layer)), dtype=torch.bfloat16)
        return (torch.ones(batch, queries, 5120, dtype=torch.bfloat16),
                torch.ones(batch, queries, 1280, dtype=torch.bfloat16), latent, start, 8)

    def step(self, start, queries=1, batch=1):
        layers = adapter.LAYERS if start + queries >= 2 else (20, *adapter.CONSUMERS)
        return {layer: self.obj(layer)(*self.args(layer, start, queries, batch)) for layer in layers}

    def test_default_off_and_both_source_pins_fail_closed(self):
        self.assertFalse(adapter.install_reference_packed_indexer(None, None).active)
        with self.assertRaises(ValueError):
            adapter.install_reference_packed_indexer(None, None, enabled=1)
        with self.assertRaises(ValueError):
            self.install(cache_boundary="live_history")
        path = Path(self.kernel.__file__)
        raw = path.read_text()
        path.write_text(raw + "# drift\n")
        with self.assertRaises(ValueError):
            self.install()
        path.write_text(raw)
        defaults = self.ref.fp4_act_quant.__defaults__
        self.ref.fp4_act_quant.__defaults__ = (32, False, torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "differs from pinned source"):
            self.install()
        self.ref.fp4_act_quant.__defaults__ = defaults
        self.ref.fp4_act_quant = lambda *a: None
        with self.assertRaises((ValueError, KeyError)):
            self.install()
        self.assertTrue(all(self.obj(k).k_cache is not None for k in adapter.OWNERS))

    def test_actual_bf16_allocations_released_and_buffers_registered(self):
        self.ref.shared_attn.index_k = self.obj(20).k_cache
        weak = [weakref.ref(self.obj(k).k_cache) for k in adapter.OWNERS]
        h = self.install()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in weak))
        for layer in adapter.OWNERS:
            obj, cache = self.obj(layer), h._caches[layer]
            self.assertIsNone(obj.k_cache)
            self.assertEqual(cache.storage_bytes, cache.batch_size * cache.capacity * 68)
            self.assertIs(obj._buffers[adapter._PACKED], cache.packed)
            self.assertNotIn(adapter._PACKED, obj.state_dict())
        self.assertIsNone(self.ref.shared_attn.index_k)

    def test_install_allocation_and_registration_failure_are_atomic(self):
        original = {k: self.obj(k).k_cache for k in adapter.OWNERS}
        allocate = packed.allocate_packed_index_cache
        calls = []
        def oom(*a, **kw):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("OOM")
            return allocate(*a, **kw)
        with patch.object(packed, "allocate_packed_index_cache", side_effect=oom):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                self.install()
        with patch.object(self.obj(8), "register_buffer", side_effect=RuntimeError("register failure")):
            with self.assertRaisesRegex(RuntimeError, "register failure"):
                self.install()
        for k, tensor in original.items():
            self.assertIs(self.obj(k).k_cache, tensor)
            self.assertNotIn(adapter._PACKED, self.obj(k)._buffers)
        self.assertTrue(all("forward" not in self.obj(k).__dict__ for k in adapter.LAYERS))
        self.assertNotIn(self.ref, old._ACTIVE)

    def test_all_eight_dense_prefill_and_ratio_two_no_publication(self):
        baseline = [self.step(0, 4), self.step(4), self.step(5)]
        h = self.install()
        actual0 = self.step(0, 4)
        cache20 = self.ref.shared_attn.index_k
        # Odd group completion: owners2/8/14 read the prior L20 slot without
        # republishing their own buffers, exactly as the original does.
        for layer in (2, 8, 14):
            value = self.obj(layer)(*self.args(layer, 4))
            self.assertIs(self.ref.shared_attn.index_k, cache20)
            self.assertTrue(torch.equal(value, baseline[1][layer]))
        actual1 = {layer: self.obj(layer)(*self.args(layer, 4)) for layer in (20, *adapter.CONSUMERS)}
        actual2 = self.step(5)
        for actual, expected in ((actual0, baseline[0]), (actual1, baseline[1]), (actual2, baseline[2])):
            for layer, value in actual.items():
                self.assertTrue(torch.equal(value, expected[layer]), (layer, value, expected[layer]))
        self.assertEqual(h.counters["packed_writes"], 9)
        self.assertEqual(h.counters["dense_calls"], 24)
        self.assertEqual(sum(event[0] == "quant" and event[1] is False for event in self.events), 9)

    def test_first_start_zero_and_contiguous_history_required(self):
        self.install()
        with self.assertRaisesRegex(RuntimeError, "fresh prefill"):
            self.obj(2)(*self.args(2, 8))
        self.assertEqual(self.reductions, [])
        self.assertTrue(self.handle._failed)

    def test_short_batch_two_and_tp_four_collective_shapes(self):
        self.ref.world_size = 4
        for layer in adapter.LAYERS:
            self.obj(layer).n_local_heads = 8
        baseline = self.step(0, 4, 2)
        self.reductions.clear()
        self.install()
        actual = self.step(0, 4, 2)
        for layer in adapter.LAYERS:
            self.assertTrue(torch.equal(actual[layer], baseline[layer]))
        self.assertEqual(self.reductions, [(2, 4, 2)] * 3 + [(2, 4, 4)] * 5)

    def test_restore_oom_keeps_packed_then_reset_guard_requires_prefill(self):
        h = self.install()
        self.step(0, 4)
        bindings = [self.obj(k).__dict__["forward"] for k in adapter.LAYERS]
        planes = [h._caches[k].packed for k in adapter.OWNERS]
        with self.assertRaises(ValueError):
            h.restore()
        zero = torch.zeros
        count = []
        def oom(*a, **kw):
            count.append(1)
            if len(count) == 3:
                raise RuntimeError("OOM")
            return zero(*a, **kw)
        with patch.object(torch, "zeros", side_effect=oom):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                h.restore(reset=True)
        self.assertTrue(h.active)
        self.assertIs(old._ACTIVE[self.ref], h)
        self.assertEqual(bindings, [self.obj(k).__dict__["forward"] for k in adapter.LAYERS])
        self.assertTrue(all(h._caches[k].packed is p for k, p in zip(adapter.OWNERS, planes)))
        h.restore(reset=True)
        self.assertFalse(h.active)
        self.assertTrue(h.reset_pending)
        self.assertEqual(h._caches, {})
        with self.assertRaisesRegex(RuntimeError, "fresh prefill"):
            self.obj(2)(*self.args(2, 4))
        with self.assertRaises(RuntimeError):
            old.install_reference_indexer(self.model, self.ref, enabled=True)
        self.step(0, 4)
        self.assertFalse(h.reset_pending)
        self.assertNotIn(self.ref, old._ACTIVE)
        self.assertTrue(all("forward" not in self.obj(k).__dict__ for k in adapter.LAYERS))

    def test_mutual_exclusion_runtime_drift_and_external_mutation_fail_closed(self):
        h = self.install()
        with self.assertRaises(RuntimeError):
            old.install_reference_indexer(self.model, self.ref, enabled=True)
        with self.assertRaises(RuntimeError):
            self.install()
        h._caches[2].packed[0, 0, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "outside the official"):
            self.step(0, 4)
        self.assertEqual(self.reductions, [])

    def test_capture_is_rejected_before_install_or_restore_allocation(self):
        with patch.object(adapter, "_no_capture_device", side_effect=RuntimeError("capture")):
            with patch.object(packed, "allocate_packed_index_cache") as allocate:
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    self.install()
                allocate.assert_not_called()
        h = self.install()
        with patch.object(adapter, "_no_capture_device", side_effect=RuntimeError("capture")):
            with patch.object(torch, "zeros") as allocate:
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    h.restore(reset=True)
                allocate.assert_not_called()
        self.assertTrue(h.active)
        fake = SimpleNamespace(cuda=MagicMock())
        fake.cuda.is_initialized.return_value = False
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            adapter._no_capture_device(fake, torch.device("cuda:7"))
        fake.cuda.device.assert_not_called()
        fake.cuda.is_initialized.return_value = True
        fake.cuda.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "capture"):
            adapter._no_capture_device(fake, torch.device("cuda:7"))
        fake.cuda.device.assert_called_once_with(torch.device("cuda:7"))

    def test_tp_state_mismatch_throws_before_collective(self):
        self.ref.world_size = 4
        for layer in adapter.LAYERS:
            self.obj(layer).n_local_heads = 8
        h = self.install()
        self.step(0, 4)
        h._eligible = lambda x, start: start > 0  # Exercise compact state with a small CPU fixture.
        for layer in (2, 8, 14, 20):
            self.obj(layer)(*self.args(layer, 4))
        self.ref.shared_attn.candidates = self.ref.shared_attn.candidates.clone()
        before = len(self.reductions)
        with self.assertRaisesRegex(RuntimeError, "state mismatch before collective"):
            self.obj(24)(*self.args(24, 4))
        self.assertEqual(len(self.reductions), before)

    def test_compact_same_step_selection_and_restore_external_override_guard(self):
        h = self.install()
        self.step(0, 4)
        h._eligible = lambda x, start: start > 0
        self.step(4)
        self.assertEqual(h.counters["source_compact_steps"], 1)
        self.assertEqual(h.counters["consumer_compact_calls"], 4)
        self.assertIsNone(h._state)
        self.obj(24).forward = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "replaced externally"):
            h.restore(reset=True)
        self.assertTrue(h.active)


if __name__ == "__main__":
    unittest.main()
