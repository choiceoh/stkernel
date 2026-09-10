"""CPU lifecycle and original Attention skeleton integration; no GPU claims.

Fixtures retain the pinned source methods verbatim. Projection/quantizer/kernel
stubs isolate routing and lifetime; the independent probe owns math comparison.
"""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay/modules/dsv41_model"))
import torch
import dsv41_dual_sparse_reference_adapter as adapter

MODEL_FIXTURE = r'''import torch
from functools import lru_cache

@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Which sliding-window cache slots each query attends to; -1 marks a slot holding nothing.

    The cache is a ring of `window_size` slots. Prefill needs one row per query, each seeing its own
    causal window. A decode step has a single query that sees the whole ring, listed oldest first.
    Order within a row does not matter to `sparse_attn`, which handles every slot independently.
    """
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)  # ring still filling
    # sparse_attn needs real [b, m, topk] int32 memory, hence the materializing expand
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()

class Attention(torch.nn.Module):
    def _window_kv(self, x, freqs_cis, start_pos):
        """This layer's sliding-window K and the window positions every query may attend to. The K
        stays fp8, quantized over the whole post-RoPE vector, RoPE tail included."""
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)
        if start_pos == 0:  # prefill: attend over this chunk, seeding the ring buffer for decode
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            window_kv = kv
        else:  # decode: one token into the ring buffer, attend over the whole window
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, start_pos)

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len):
        """Which compressed positions each query attends to. Index sources run their own indexer;
        the layers in between reuse the result their source published."""
        if not self.is_index_source:
            return shared_attn.topk_idxs

        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            idxs = self.indexer(x, qr, latent, start_pos, offset)
        shared_attn.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset):
        """The shared compressed KV and the compressed positions every query may attend to. This
        layer compresses its own KV only when it is a source; otherwise it just reads the cache."""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            shared_attn.compress_kv = self.compress_kv_cache
        # the indexer needs the latent before RoPE, so it runs before the cache is written
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)
            # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
            fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
        # read after the write, so this does not depend on the slice aliasing the cache
        return shared_attn.compress_kv[:bsz, :compress_len], idxs

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # wo_a is block-diagonal over groups (each projects only its own heads), hence einsum not
        # Linear. convert.py dequantizes it to bf16; an fp8 grouped GEMM would halve the memory.
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x
'''

KERNEL_FIXTURE = r'''import torch

def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    b, s, h, d = q.size()
    # Pad heads to 16 for kernel efficiency (stripped after)
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
    o = torch.empty_like(q)
    kernel = sparse_attn_kernel(q.size(2), d, softmax_scale)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o
'''


class FakeIndexer(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio, self.freqs_cis = ratio, None

    def forward(self, x, qr, latent, start_pos, offset):
        width = (start_pos + x.shape[1]) // self.ratio
        return (torch.arange(min(512, width), dtype=torch.int32).view(1, 1, -1)
                .expand(x.shape[0], x.shape[1], -1).contiguous() + offset)


class DualAdapterTests(unittest.TestCase):
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
        self.ref, self.kernel = ModuleType("dual_ref_fixture"), ModuleType("dual_kernel_fixture")
        for module, name, source in ((self.ref, "model.py", MODEL_FIXTURE), (self.kernel, "kernel.py", KERNEL_FIXTURE)):
            module.__file__ = str(Path(self.temp.name) / name)
            Path(module.__file__).write_text(source)
            exec(compile(source, module.__file__, "exec"), vars(module))
        self.events, self.records = [], []
        self.ref.world_size = 4
        self.ref.shared_attn = SimpleNamespace(compress_kv=None, topk_idxs=None, index_k=None, candidates=None)
        self.ref.sparse_attn = self.kernel.sparse_attn
        self.ref.fp8_block_size, self.ref.scale_fmt, self.ref.scale_dtype = 32, "ue8m0", torch.float8_e8m0fnu
        self.ref.apply_rotary_emb = lambda x, f, inverse=False: self.events.append(("rope", inverse, tuple(x.shape)))
        self.ref.act_quant = lambda x, *a: self.events.append(("fp8", tuple(x.shape)))
        self.ref.fp4_act_quant = lambda x, *a, **kw: self.events.append(("fp4", tuple(x.shape)))
        self.kernel.sparse_attn_kernel = self.original_kernel
        self.model = SimpleNamespace(layers=[SimpleNamespace(attn=None) for _ in range(40)])
        for layer in range(40):
            inst = self.ref.Attention()
            attrs = dict(layer_id=layer, dim=5120, n_heads=64, n_local_heads=16,
                q_lora_rank=1280, o_lora_rank=1024, head_dim=512, rope_head_dim=64,
                nope_head_dim=448, n_groups=8, n_local_groups=2, window_size=128,
                compress_ratio=0 if layer < 2 else 2 if layer < 20 else 1,
                eps=1e-20, softmax_scale=512**-.5,
                is_kv_source=layer in adapter.OWNERS, is_index_source=layer in adapter.INDEXERS)
            for name, value in attrs.items():
                setattr(inst, name, value)
            inst.register_buffer("window_kv_cache", torch.zeros(2, 128, 512, dtype=torch.bfloat16), persistent=False)
            inst.register_buffer("freqs_cis", torch.ones(258, 32, dtype=torch.complex64), persistent=False)
            inst.attn_sink = torch.nn.Parameter(torch.zeros(16))
            inst.wq_a, inst.q_norm = lambda x: x[..., :1280], lambda x: x
            inst.wq_b = lambda qr: torch.zeros((*qr.shape[:2], 16*512), dtype=torch.bfloat16)
            inst.wkv, inst.kv_norm = lambda x: x[..., :512].clone(), lambda x: x
            inst.wo_a = SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16).expand(2048, 4096))
            inst.wo_b = lambda x: torch.zeros((*x.shape[:2], 5120), dtype=torch.bfloat16)
            inst.indexer = FakeIndexer(inst.compress_ratio) if inst.is_index_source else None
            inst.compressor = lambda x, pos, ratio=inst.compress_ratio: (
                x[:, :x.shape[1] // ratio, :512].clone() if pos == 0 else
                x[..., :512].clone() if (pos + 1) % ratio == 0 else None)
            if inst.is_kv_source:
                inst.register_buffer("compress_kv_cache", torch.zeros(2, 258//inst.compress_ratio, 512,
                                                                      dtype=torch.bfloat16), persistent=False)
            self.model.layers[layer].attn = inst
        self.core = SimpleNamespace(dual_sparse_attn=self.dual_kernel)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"dsv41_dual_sparse": self.core}).start()
        patch.object(adapter, "MODEL_SHA256", hashlib.sha256(Path(self.ref.__file__).read_bytes()).hexdigest()).start()
        patch.object(adapter, "KERNEL_SHA256", hashlib.sha256(Path(self.kernel.__file__).read_bytes()).hexdigest()).start()
        self.addCleanup(lambda: adapter._ACTIVE.pop(self.ref, None))
        # This suite tests the exact forward skeleton, not an 8M-element output
        # projection per layer. The independent probe owns arithmetic oracles.
        einsum = torch.einsum
        def projection(equation, *tensors):
            if equation == "bsgd,grd->bsgr":
                b, q, g, _ = tensors[0].shape
                return torch.zeros(b, q, g, 1024, dtype=torch.bfloat16)
            return einsum(equation, *tensors)
        patch.object(torch, "einsum", side_effect=projection).start()
        self.handle = None

    def original_kernel(self, heads, dim, scale):
        def run(q, kv, output, sink, indices):
            self.records.append(self.gather(q, kv, None, indices))
            output.zero_()
        return run

    def gather(self, q, window, compressed, indices):
        out = torch.zeros((*indices.shape, 512), dtype=torch.bfloat16)
        split = window.shape[1]
        for batch in range(q.shape[0]):
            idx = indices[batch].long()
            mask = (idx >= 0) & (idx < split)
            if mask.any():
                out[batch][mask] = window[batch, idx[mask]]
            if compressed is not None:
                mask = (idx >= split) & (idx < split + compressed.shape[1])
                if mask.any():
                    out[batch][mask] = compressed[batch, idx[mask] - split]
        return indices.clone(), out

    def dual_kernel(self, q, window, compressed, sink, indices, scale, **kwargs):
        self.records.append(self.gather(q, window, compressed, indices))
        self.assertEqual(kwargs, dict(backend="torch", query_chunk_size=1))
        self.assertEqual(compressed.stride(0), self.ref.shared_attn.compress_kv.stride(0))
        return torch.zeros_like(q)

    def install(self):
        self.handle = adapter.install_reference_dual_sparse_attention(self.model, self.ref, enabled=True)
        return self.handle

    def step(self, start, queries=1, batch=1, layers=range(2, 40)):
        x = torch.arange(batch*queries*5120, dtype=torch.float32).reshape(batch, queries, 5120).remainder(5).bfloat16()
        for layer in layers:
            self.model.layers[layer].attn(x, start)

    def test_default_off_source_and_loaded_methods_are_strict(self):
        self.assertFalse(adapter.install_reference_dual_sparse_attention(None, None).active)
        with self.assertRaises(ValueError):
            adapter.install_reference_dual_sparse_attention(None, None, enabled=1)
        path = Path(self.kernel.__file__)
        raw = path.read_text()
        path.write_text(raw + "# drift\n")
        with self.assertRaisesRegex(ValueError, "kernel source SHA"):
            self.install()
        path.write_text(raw)
        self.ref.Attention._window_kv = lambda *args: None
        with self.assertRaisesRegex(ValueError, "differs from pinned"):
            self.install()
        self.assertNotIn(self.ref, adapter._ACTIVE)

    def test_loaded_default_drift_and_geometry_rejected(self):
        self.ref.sparse_attn.__defaults__ = (1.0,)
        with self.assertRaisesRegex(ValueError, "differs from pinned"):
            self.install()
        self.ref.sparse_attn.__defaults__ = None
        self.model.layers[20].attn.head_dim = 256
        with self.assertRaisesRegex(ValueError, "layer 20"):
            self.install()
        self.assertTrue(all("forward" not in self.model.layers[i].attn.__dict__ for i in range(40)))

    def test_ast_rewrite_removes_only_kv_cat_and_changes_one_call(self):
        node, _, filename, _ = adapter._source_contract(self.ref)
        saved = copy.deepcopy(node)
        captured = []
        with patch.object(adapter, "_compile_method", side_effect=lambda n, *_: captured.append(n)):
            adapter._adapt_forward(node, {}, filename)
        changed = captured[0]
        class Undo(ast.NodeTransformer):
            def visit_If(self, n):
                n = self.generic_visit(n)
                if ast.unparse(n.test) == "self.compress_ratio":
                    n.body.insert(1, ast.parse("kv = torch.cat([kv, compress_kv], dim=1)").body[0])
                return n
            def visit_Call(self, n):
                if isinstance(n.func, ast.Name) and n.func.id == "_dual_sparse_attn":
                    return ast.parse("sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)", mode="eval").body
                return self.generic_visit(n)
        self.assertEqual(ast.dump(saved, include_attributes=False), ast.dump(Undo().visit(changed), include_attributes=False))

    def test_all_38_prefill_decode_empty_prefix_and_ring_wrap_equal_gathers(self):
        sequence = [(0, 1, 2), (1, 1, 2), (2, 1, 2), (0, 129, 1), (129, 1, 1)]
        for start, q, b in sequence:
            self.step(start, q, b)
        baseline = self.records[:]
        events = self.events[:]
        self.records.clear(); self.events.clear()
        handle = self.install()
        cat = torch.cat
        def forbid_full_cat(tensors, *args, **kwargs):
            dim = kwargs.get("dim", args[0] if args else 0)
            if dim == 1 and tensors[0].dtype == torch.bfloat16:
                raise AssertionError("full KV concat reached candidate")
            return cat(tensors, *args, **kwargs)
        with patch.object(torch, "cat", side_effect=forbid_full_cat):
            for start, q, b in sequence:
                self.step(start, q, b)
        self.assertEqual(events, self.events)
        self.assertEqual(handle.counters, {"dual_calls": 38*len(sequence)})
        for a, b in zip(baseline, self.records):
            self.assertTrue(torch.equal(a[0], b[0]))
            self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(all("forward" not in self.model.layers[i].attn.__dict__ for i in (0, 1)))

    def test_restore_preserves_cache_history_and_shared_slots(self):
        h = self.install()
        self.step(0, 3)
        owners = {i: self.model.layers[i].attn.compress_kv_cache for i in adapter.OWNERS}
        contents = {i: t.clone() for i, t in owners.items()}
        shared = self.ref.shared_attn.compress_kv
        idxs = self.ref.shared_attn.topk_idxs
        h.restore(); h.restore()
        self.assertFalse(h.active)
        self.assertNotIn(self.ref, adapter._ACTIVE)
        self.assertIs(self.ref.shared_attn.compress_kv, shared)
        self.assertIs(self.ref.shared_attn.topk_idxs, idxs)
        for i, t in owners.items():
            self.assertIs(self.model.layers[i].attn.compress_kv_cache, t)
            self.assertTrue(torch.equal(t, contents[i]))
        self.step(3)  # Ordinary original decode resumes existing history.

    def test_replaced_instance_helpers_storage_or_world_are_rejected(self):
        h = self.install()
        obj = self.model.layers[2].attn
        original = obj.window_kv_cache
        obj.window_kv_cache = original.clone()
        with self.assertRaisesRegex(RuntimeError, "tensor device/shape/stride/storage"):
            self.step(0, 1, layers=(2,))
        obj.window_kv_cache = original
        self.ref.world_size = 1
        with self.assertRaisesRegex(RuntimeError, "runtime, TP"):
            self.step(0, 1, layers=(2,))
        self.ref.world_size = 4
        self.model.layers[2].attn = self.model.layers[3].attn
        with self.assertRaisesRegex(RuntimeError, "instance or forward"):
            self.step(0, 1, layers=(2,))
        self.model.layers[2].attn = obj
        obj._compress_kv = lambda *a: None
        with self.assertRaisesRegex(RuntimeError, "helper changed"):
            h.restore()
        self.assertTrue(h.active)

    def test_install_rollback_and_capture_guards_are_pre_mutation(self):
        with patch.object(adapter, "_no_capture", side_effect=RuntimeError("capture")):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.install()
        self.assertTrue(all("forward" not in self.model.layers[i].attn.__dict__ for i in range(40)))
        setattr_original = self.ref.Attention.__setattr__
        def fail(inst, name, value):
            if name == "forward" and inst.layer_id == 20:
                raise RuntimeError("install mutation failed")
            return setattr_original(inst, name, value)
        with patch.object(self.ref.Attention, "__setattr__", fail):
            with self.assertRaisesRegex(RuntimeError, "install mutation failed"):
                self.install()
        self.assertNotIn(self.ref, adapter._ACTIVE)
        self.assertTrue(all("forward" not in self.model.layers[i].attn.__dict__ for i in range(40)))
        h = self.install()
        with patch.object(adapter, "_no_capture", side_effect=RuntimeError("capture")):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                h.restore()
        self.assertTrue(h.active)
        fake = SimpleNamespace(cuda=MagicMock())
        fake.cuda.is_initialized.return_value = False
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            adapter._no_capture(fake, torch.device("cuda:7"))
        fake.cuda.device.assert_not_called()

    def test_indexer_adapter_registry_is_independent(self):
        import dsv41_reference_adapter as compact
        marker = SimpleNamespace(active=True)
        with patch.dict(compact._ACTIVE, {self.ref: marker}):
            h = self.install()
            self.step(0, 3)
            self.assertIs(compact._ACTIVE[self.ref], marker)
            h.restore()
            self.assertIs(compact._ACTIVE[self.ref], marker)


if __name__ == "__main__":
    unittest.main()
