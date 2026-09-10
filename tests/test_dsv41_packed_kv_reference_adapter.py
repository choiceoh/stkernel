"""Packed-KV lifecycle and pinned Attention skeleton tests, CPU only.

The original Python quantizer wrapper executes a scale=1 CPU fixture here;
actual E4M3 rounding/TileLang layout are independent probe/GPU obligations.
"""
import ast
import copy
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay/modules/dsv41_model"))
import torch
import dsv41_packed_kv as core
import dsv41_packed_kv_reference_adapter as adapter
import dsv41_dual_sparse_reference_adapter as dual
import dsv41_packed_reference_adapter as packed_index


def load_test(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


attention_fixture = load_test("test_dsv41_dual_sparse_reference_adapter")
quant_fixture = load_test("test_dsv41_packed_reference_adapter")


class PackedKVAdapterTests(unittest.TestCase):
    original_kernel = attention_fixture.DualAdapterTests.original_kernel
    gather = attention_fixture.DualAdapterTests.gather
    dual_kernel = attention_fixture.DualAdapterTests.dual_kernel
    step = attention_fixture.DualAdapterTests.step

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        attention_fixture.DualAdapterTests.setUp(self)
        kernel_source = attention_fixture.KERNEL_FIXTURE + "\n" + quant_fixture.KERNEL_FIXTURE
        Path(self.kernel.__file__).write_text(kernel_source)
        exec(compile(kernel_source, self.kernel.__file__, "exec"), vars(self.kernel))
        self.kernel.FP8, self.kernel.FE8M0 = "e4m3", "e8m0"
        self.kernel.fp4_quant_kernel = self.quant_factory
        self.ref.sparse_attn, self.ref.fp4_act_quant = self.kernel.sparse_attn, self.kernel.fp4_act_quant
        pin = hashlib.sha256(kernel_source.encode()).hexdigest()
        patch.object(dual, "KERNEL_SHA256", pin).start()
        patch.object(packed_index, "KERNEL_SHA256", pin).start()
        patch.object(core, "packed_sparse_attn", new=self.packed_kernel).start()
        self.addCleanup(lambda: dual._ACTIVE.pop(self.ref, None))
        # Match the actual compressor's None on incomplete initial groups.
        for layer in dual.OWNERS:
            inst = self.model.layers[layer].attn
            previous = inst.compressor
            inst.compressor = lambda x, pos, old=previous, ratio=inst.compress_ratio: (
                None if pos == 0 and x.shape[1] < ratio else old(x, pos))

    def quant_factory(self, n, block_size, *, scale_dtype, inplace):
        self.assertEqual((n, block_size, scale_dtype), (512, 16, "e4m3"))
        self.events.append(("official_fp4", block_size, inplace))
        def run(x, y, sf):
            lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
            code = (x.float().abs().unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)
            code |= x.signbit().to(torch.uint8) << 3
            sf.view(torch.uint8).fill_(0x38)
            if inplace:
                y.copy_(lut[(code & 7).long()] * torch.where(code & 8 != 0, -1., 1.))
            else:
                y.view(torch.uint8).copy_(code[:, ::2] | (code[:, 1::2] << 4))
        return run

    def packed_kernel(self, q, window, cache, sink, ids, scale, *, width, backend, query_chunk_size):
        self.assertEqual((backend, query_chunk_size), ("torch", 1))
        # Small test-only materialization for gather-byte comparison. Production
        # packed_sparse_attn reconstructs bounded selected tiles, never a prefix.
        compressed = core.unpack_kv_tile(cache.packed[:q.shape[0], :width], cache.scales[:q.shape[0], :width])
        self.records.append(self.gather(q, window, compressed, ids))
        return torch.zeros_like(q)

    def install(self):
        self.handle = adapter.install_reference_packed_kv_attention(self.model, self.ref, enabled=True)
        return self.handle

    def test_default_off_pins_quantizer_defaults_and_cache_boundary(self):
        self.assertFalse(adapter.install_reference_packed_kv_attention(None, None).active)
        with self.assertRaises(ValueError):
            adapter.install_reference_packed_kv_attention(None, None, enabled=1)
        with self.assertRaises(ValueError):
            adapter.install_reference_packed_kv_attention(self.model, self.ref, enabled=True, cache_boundary="live")
        self.ref.fp4_act_quant.__defaults__ = (16, True, torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "differs from pinned"):
            self.install()
        self.assertNotIn(self.ref, dual._ACTIVE)
        self.assertTrue(all(self.model.layers[i].attn.compress_kv_cache is not None for i in dual.OWNERS))

    def test_install_releases_all_old_bf16_owners_without_history_copy(self):
        self.ref.shared_attn.compress_kv = self.model.layers[20].attn.compress_kv_cache
        references = [weakref.ref(self.model.layers[i].attn.compress_kv_cache) for i in dual.OWNERS]
        h = self.install()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        for i, cache in h._caches.items():
            inst = self.model.layers[i].attn
            self.assertIsNone(inst.compress_kv_cache)
            self.assertIs(inst._buffers[adapter._PACKED], cache.packed)
            self.assertEqual(cache.storage_bytes, cache.batch_size * cache.capacity * 288)
            self.assertNotIn(adapter._PACKED, inst.state_dict())
        self.assertIsNone(self.ref.shared_attn.compress_kv)

    def test_all_38_original_gathers_quant_order_and_empty_publication(self):
        sequence = [(0, 1, 2), (1, 1, 2), (2, 1, 2)]
        for start, q, batch in sequence:
            self.step(start, q, batch)
        baseline, events = self.records[:], self.events[:]
        self.records.clear(); self.events.clear()
        h = self.install()
        cat = torch.cat
        def no_full_cat(tensors, *args, **kwargs):
            if kwargs.get("dim", args[0] if args else 0) == 1 and tensors[0].dtype == torch.bfloat16:
                raise AssertionError("full KV concat reached packed candidate")
            return cat(tensors, *args, **kwargs)
        with patch.object(torch, "cat", side_effect=no_full_cat):
            for start, q, batch in sequence:
                self.step(start, q, batch)
        self.assertEqual(h.counters["packed_calls"], 114)
        self.assertEqual(h.counters["owner_publications"], 12)
        self.assertEqual(h.counters["packed_writes"], 6)  # 1 + 4 + 1, odd ratio2 skips write only.
        for a, b in zip(baseline, self.records):
            self.assertTrue(torch.equal(a[0], b[0]))
            self.assertTrue(torch.equal(a[1], b[1]))
        normalize = lambda xs: [item[:2] if item[0] == "official_fp4" else item for item in xs]
        self.assertEqual(normalize(events), normalize(self.events))
        self.assertTrue(all(item[2] is False for item in self.events if item[0] == "official_fp4"))
        self.assertTrue(all("forward" not in self.model.layers[i].attn.__dict__ for i in (0, 1)))

    def test_install_allocation_and_registration_failures_rollback(self):
        original = {i: self.model.layers[i].attn.compress_kv_cache for i in dual.OWNERS}
        klass, calls = core.PackedKVCache, []
        def allocation(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("OOM")
            return klass(*args, **kwargs)
        with patch.object(core, "PackedKVCache", side_effect=allocation):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                self.install()
        with patch.object(self.model.layers[8].attn, "register_buffer", side_effect=RuntimeError("register failed")):
            with self.assertRaisesRegex(RuntimeError, "register failed"):
                self.install()
        for i, value in original.items():
            self.assertIs(self.model.layers[i].attn.compress_kv_cache, value)
            self.assertNotIn(adapter._PACKED, self.model.layers[i].attn._buffers)
        self.assertTrue(all("forward" not in self.model.layers[i].attn.__dict__ and
                            "_compress_kv" not in self.model.layers[i].attn.__dict__ for i in range(2, 40)))
        self.assertNotIn(self.ref, dual._ACTIVE)

    def test_restore_oom_is_atomic_then_packed_planes_release_and_fresh_prefill(self):
        h = self.install()
        self.step(0, 4)
        before = {i: c.packed.clone() for i, c in h._caches.items()}
        methods = [self.model.layers[i].attn.forward for i in range(2, 40)]
        zeros, calls = torch.zeros, []
        def oom(*a, **kw):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("OOM")
            return zeros(*a, **kw)
        with patch.object(torch, "zeros", side_effect=oom):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                h.restore(reset=True)
        self.assertTrue(h.active)
        self.assertIs(dual._ACTIVE[self.ref], h)
        self.assertEqual(methods, [self.model.layers[i].attn.forward for i in range(2, 40)])
        for i, data in before.items():
            self.assertTrue(torch.equal(h._caches[i].packed, data))
        references = [weakref.ref(c.packed) for c in h._caches.values()]
        with self.assertRaises(ValueError):
            h.restore()
        h.restore(reset=True)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertFalse(h.active)
        self.assertTrue(h.reset_pending)
        with self.assertRaisesRegex(RuntimeError, "fresh prefill"):
            self.step(4, layers=(2,))
        self.assertIs(dual._ACTIVE[self.ref], h)
        self.step(0, 4)
        self.assertFalse(h.reset_pending)
        self.assertNotIn(self.ref, dual._ACTIVE)
        self.assertTrue(all("forward" not in self.model.layers[i].attn.__dict__ and
                            "_compress_kv" not in self.model.layers[i].attn.__dict__ for i in range(2, 40)))

    def test_ordered_start_layer_and_contiguous_decode_fail_closed(self):
        h = self.install()
        with self.assertRaisesRegex(RuntimeError, "fresh prefill"):
            self.step(4, layers=(2,))
        self.assertTrue(h._failed)
        with self.assertRaisesRegex(RuntimeError, "previous packed KV forward failed"):
            self.step(0, 4, layers=(2,))
        h.restore(reset=True)
        self.step(0, 4)
        h = self.install()
        self.step(0, 4)
        with self.assertRaisesRegex(RuntimeError, "contiguous decode"):
            self.step(8, layers=(2,))

    def test_external_packed_mutation_and_method_drift_rejected(self):
        h = self.install()
        h._caches[2].packed[0, 0, 0] = 7
        with self.assertRaisesRegex(RuntimeError, "outside the official-byte"):
            self.step(0, 4, layers=(2,))
        self.assertTrue(h._failed)
        self.assertEqual(h.counters["packed_calls"], 0)
        self.model.layers[8].attn._compress_kv = lambda *a: None
        with self.assertRaises(RuntimeError):
            h.restore(reset=True)
        self.assertTrue(h.active)

    def test_capture_and_registry_coexistence_contracts(self):
        with patch.object(dual, "_no_capture", side_effect=RuntimeError("capture")):
            with patch.object(core, "PackedKVCache") as allocate:
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    self.install()
                allocate.assert_not_called()
        import dsv41_reference_adapter as old_index
        with patch.dict(old_index._ACTIVE, {self.ref: object()}):
            h = self.install()
            self.step(0, 2)
            self.assertIn(self.ref, old_index._ACTIVE)
            with self.assertRaises(RuntimeError):
                dual.install_reference_dual_sparse_attention(self.model, self.ref, enabled=True)
            with patch.object(dual, "_no_capture", side_effect=RuntimeError("capture")):
                with patch.object(torch, "zeros") as allocate:
                    with self.assertRaisesRegex(RuntimeError, "capture"):
                        h.restore(reset=True)
                    allocate.assert_not_called()
            self.assertTrue(h.active)

    def test_compress_ast_only_changes_publication_storage_quant_mode_and_return(self):
        filename = str(Path(self.ref.__file__).resolve())
        tree = ast.parse(Path(filename).read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compress_kv")
        captured = []
        with patch.object(dual, "_compile_method", side_effect=lambda node, *_: captured.append(node)):
            adapter._adapt_compress(self.ref, {}, filename)
        class Undo(ast.NodeTransformer):
            def visit_Expr(self, node):
                text = ast.unparse(node)
                if text == "_packed_publish(self)":
                    return ast.parse("shared_attn.compress_kv = self.compress_kv_cache").body[0]
                if text.startswith("_packed_write("):
                    return ast.parse("self.compress_kv_cache[:bsz, start_pos // ratio:start_pos // ratio + latent.size(1)] = latent").body[0]
                return self.generic_visit(node)
            def visit_Assign(self, node):
                if ast.unparse(node).startswith("_packed_y, _packed_sf ="):
                    return ast.parse("fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)").body[0]
                return self.generic_visit(node)
            def visit_Return(self, node):
                if ast.unparse(node).startswith("return (_packed_prefix("):
                    return ast.parse("return shared_attn.compress_kv[:bsz, :compress_len], idxs").body[0]
                return self.generic_visit(node)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(Undo().visit(captured[0]), include_attributes=False))


if __name__ == "__main__":
    unittest.main()
