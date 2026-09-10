"""Opt-in packed compressed-KV storage for the official V4.1 reference.

Retains the dual-pool no-concat Attention path, but owns four E2M1/E4M3 caches.
The original indexer runs before latent RoPE and quantization. Install only at
an explicit fresh-prefill boundary after device placement. Restore discards
history and requires a new prefill; neither old BF16 caches nor full-cache
unpacked copies are retained. Ordered, single-model, non-captured execution
remains mandatory. Source pins and CPU tests do not prove GPU quantizer layout
or numerics; there is no per-step tensor-content scan or host synchronization.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

try:
    from . import dsv41_dual_sparse_reference_adapter as dual
    from . import dsv41_packed_reference_adapter as index_adapter
except ImportError:
    import dsv41_dual_sparse_reference_adapter as dual
    import dsv41_packed_reference_adapter as index_adapter

_PACKED = "_dsv41_compressed_kv_packed"
_SCALES = "_dsv41_compressed_kv_scales"
_ABSENT = object()


@dataclass(frozen=True)
class PackedKVPrefix:
    cache: object
    batch: int
    width: int
    generation: int


def _adapt_compress(reference, namespace, filename):
    tree = ast.parse(Path(filename).read_bytes(), filename=filename)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compress_kv"))
    changed = dict(publish=0, quant=0, write=0, result=0)
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, n):
            if ast.unparse(n) == "shared_attn.compress_kv = self.compress_kv_cache":
                changed["publish"] += 1
                return ast.copy_location(ast.parse("_packed_publish(self)").body[0], n)
            if ast.unparse(n) == "self.compress_kv_cache[:bsz, start_pos // ratio:start_pos // ratio + latent.size(1)] = latent":
                changed["write"] += 1
                return ast.copy_location(ast.parse(
                    "_packed_write(self, _packed_y, _packed_sf, bsz, start_pos // ratio, latent.size(1))").body[0], n)
            return self.generic_visit(n)

        def visit_Expr(self, n):
            if ast.unparse(n) == "fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)":
                changed["quant"] += 1
                return ast.copy_location(ast.parse(
                    "_packed_y, _packed_sf = fp4_act_quant(latent, 16, False, scale_dtype=torch.float8_e4m3fn)").body[0], n)
            return self.generic_visit(n)

        def visit_Return(self, n):
            if ast.unparse(n) == "return (shared_attn.compress_kv[:bsz, :compress_len], idxs)":
                changed["result"] += 1
                return ast.copy_location(ast.parse("return _packed_prefix(self, bsz, compress_len), idxs").body[0], n)
            return self.generic_visit(n)
    node = Rewrite().visit(node)
    if changed != dict(publish=1, quant=1, write=1, result=1):
        raise ValueError("pinned compressed KV replacement boundaries changed")
    return dual._compile_method(node, namespace, filename)


class PackedKVAttentionHandle(dual.DualSparseAttentionHandle):
    def __init__(self):
        super().__init__()
        self.reset_pending = False
        self._caches = {}
        self._cache_metadata = {}
        self._valid = {}
        self._step = None
        self._last_layer = 39
        self._failed = False
        self._issued_prefix = None
        self._counts = dict(packed_writes=0, owner_publications=0, packed_calls=0)

    def _runtime(self):
        super()._runtime()
        if (self._reference.fp4_act_quant is not self._quantizer
                or dual._signature(self._quantizer) != self._quantizer_signature
                or dual._ACTIVE.get(self._reference) is not self):
            raise RuntimeError("packed KV quantizer or installation ownership changed")

    def _instance(self, layer):
        instance = self._instances[layer]
        if (self._layers[layer].attn is not instance or type(instance) is not self._reference.Attention
                or instance.__dict__.get("forward") is not self._installed[layer]
                or instance.__dict__.get("_compress_kv") is not self._compress_installed[layer]):
            raise RuntimeError("installed packed Attention instance or method changed")
        if any(type(getattr(instance, name, None)) is not type(value) or getattr(instance, name) != value
               for name, value in self._geometry[layer].items()):
            raise RuntimeError("installed packed Attention geometry changed")
        for name in ("_window_kv", "_compress_topk_idxs"):
            if getattr(getattr(instance, name), "__func__", None) is not self._originals[name]:
                raise RuntimeError("installed packed Attention helper changed")
        for name, metadata in self._metadata[layer].items():
            if dual._tensor_metadata(getattr(instance, name)) != metadata:
                raise RuntimeError("installed Attention tensor metadata changed")
        if layer in self._caches:
            cache = self._caches[layer]
            cache.validate()
            if (instance._buffers.get("compress_kv_cache") is not None
                    or instance._buffers.get(_PACKED) is not cache.packed
                    or instance._buffers.get(_SCALES) is not cache.scales):
                raise RuntimeError("installed packed KV buffers changed")
        return instance

    def _clear_shared(self):
        self._shared.compress_kv = None
        self._shared.topk_idxs = None
        self._issued_prefix = None
        # index_k and candidates belong to the independently installed indexer.

    def _begin(self, layer, instance, x, start_pos):
        self._runtime()
        self._instance(layer)
        dual._no_capture(self._reference.torch, x.device)
        if self._failed:
            raise RuntimeError("previous packed KV forward failed; restore/reset required")
        if (type(start_pos) is not int or start_pos < 0 or x.ndim != 3 or min(x.shape[:2]) < 1
                or x.shape[2] != 5120 or x.dtype != self._reference.torch.bfloat16
                or x.device != instance.window_kv_cache.device or x.shape[0] > instance.window_kv_cache.shape[0]
                or start_pos + x.shape[1] > instance.freqs_cis.shape[0]
                or (start_pos > 0 and x.shape[1] != 1)):
            raise ValueError("unsupported packed Attention input geometry/device")
        step = (start_pos, x.shape[0], x.shape[1])
        if self._step is None or step != self._step or self._last_layer == 39:
            if layer != 2 or self._last_layer != 39:
                raise RuntimeError("packed Attention requires ordered layers2..39")
            if start_pos == 0:
                self._valid = {owner: 0 for owner in dual.OWNERS}
                self._clear_shared()
            elif (self._step is None or start_pos != self._step[0] + self._step[2]
                  or x.shape[0] != self._step[1]):
                raise RuntimeError("packed Attention requires fresh prefill or contiguous decode")
            self._step, self._last_layer = step, 1
        if layer != self._last_layer + 1:
            raise RuntimeError("packed Attention layer order changed")
        self._last_layer = layer

    def _forward(self, layer, instance, x, start_pos):
        try:
            self._begin(layer, instance, x, start_pos)
            return self._forward_impl(instance, x, start_pos)
        except BaseException:
            self._failed = True
            self._issued_prefix = None
            raise

    def _publish(self, instance):
        # AST keeps publication before indexer, including latent=None steps.
        cache = self._caches[instance.layer_id]
        cache.validate()
        self._shared.compress_kv = cache
        self._counts["owner_publications"] += 1

    def _write(self, instance, y, scales, batch, start, rows):
        layer = instance.layer_id
        cache = self._caches[layer]
        if (self._shared.compress_kv is not cache or start != self._valid[layer]
                or y.shape[0] != batch or rows != y.shape[1]):
            raise RuntimeError("packed KV write does not extend the published owner")
        self._core.write_packed_kv(cache, y, scales, start_slot=start, rows=rows)
        self._valid[layer] = start + rows
        self._counts["packed_writes"] += 1

    def _prefix(self, instance, batch, width):
        cache = self._shared.compress_kv
        owner = max(i for i in dual.OWNERS if i <= instance.layer_id)
        if (cache is not self._caches[owner] or width > self._valid[owner]
                or batch != self._step[1] or width != (self._step[0] + self._step[2]) // instance.compress_ratio):
            raise RuntimeError("packed KV prefix publication/order/width mismatch")
        self._instance(owner)
        prefix = PackedKVPrefix(cache, batch, width, cache.generation)
        self._issued_prefix = prefix
        return prefix

    def _dual(self, q, window, prefix, sink, ids, scale):
        if (not isinstance(prefix, PackedKVPrefix) or prefix is not self._issued_prefix
                or prefix.cache is not self._shared.compress_kv
                or prefix.cache.generation != prefix.generation or q.shape[0] != prefix.batch):
            raise RuntimeError("packed KV prefix is stale or not issued by this forward")
        try:
            output = self._core.packed_sparse_attn(q, window, prefix.cache, sink, ids, scale,
                            width=prefix.width, backend=self._backend, query_chunk_size=1)
        finally:
            self._issued_prefix = None
        self._counts["packed_calls"] += 1
        return output

    def restore(self, *, reset=False):
        """Allocate fresh BF16 owners before changing methods; discard all history."""
        if not self.active:
            return
        if reset is not True:
            raise ValueError("restore requires reset=True and a subsequent fresh prefill")
        self._runtime()
        for layer in self._instances:
            instance = self._instance(layer)
            dual._no_capture(self._reference.torch, instance.window_kv_cache.device)
        torch = self._reference.torch
        fresh = {layer: torch.zeros(shape, dtype=dtype, device=device)
                 for layer, (shape, dtype, device) in self._cache_metadata.items()}
        # OOM above leaves all active packed data, hooks and registry unchanged.
        for layer, tensor in fresh.items():
            instance = self._instances[layer]
            instance._buffers["compress_kv_cache"] = tensor
            for name in (_PACKED, _SCALES):
                instance._buffers.pop(name)
                instance._non_persistent_buffers_set.discard(name)
        for instance, _, _, previous, _ in self._bindings:
            if previous is _ABSENT:
                instance.__dict__.pop("_compress_kv", None)
            else:
                instance._compress_kv = previous
        self._clear_shared()
        self._caches.clear()
        self.active, self.reset_pending = False, True
        for instance, _, _, _, _ in self._bindings:
            def guard(inst, x, start_pos):
                self._runtime()
                dual._no_capture(self._reference.torch, x.device)
                if type(start_pos) is not int or start_pos != 0 or inst.layer_id != 2:
                    raise RuntimeError("restored KV caches require fresh prefill from layer2 at start_pos=0")
                self._remove_guards()
                return self._originals["forward"](inst, x, start_pos)
            instance.forward = MethodType(guard, instance)
        self._reset_guards = [(i, i.__dict__["forward"]) for i, _, _, _, _ in self._bindings]

    def _remove_guards(self):
        if any(i.__dict__.get("forward") is not guard for i, guard in self._reset_guards):
            raise RuntimeError("restored packed KV guard was replaced externally")
        for instance, previous, _, _, _ in self._bindings:
            if previous is _ABSENT:
                instance.__dict__.pop("forward", None)
            else:
                instance.forward = previous
        self.reset_pending = False
        dual._ACTIVE.pop(self._reference, None)


def install_reference_packed_kv_attention(model, reference_module, *, enabled=False,
                                          cache_boundary="fresh_prefill", backend="torch"):
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit bool")
    handle = PackedKVAttentionHandle()
    if not enabled:
        return handle
    if cache_boundary != "fresh_prefill" or backend not in ("torch", "triton"):
        raise ValueError("unsupported cache boundary or backend")
    if reference_module in dual._ACTIVE:
        raise RuntimeError("reference runtime already has an attention adapter or reset guard")
    forward_node, originals, model_path, kernel_path = dual._source_contract(reference_module)
    if index_adapter._kernel_contract(reference_module) != kernel_path:
        raise ValueError("quantizer and attention wrapper must use the same pinned kernel module")
    try:
        from . import dsv41_packed_kv as core
    except ImportError:
        import dsv41_packed_kv as core
    torch, world = reference_module.torch, reference_module.world_size
    if type(world) is not int or world < 1 or 8 % world or len(model.layers) != 40:
        raise ValueError("unsupported official Attention TP or layer count")
    handle._reference, handle._shared, handle._world = reference_module, reference_module.shared_attn, world
    handle._model, handle._layers = model, model.layers
    handle._originals, handle._core, handle._backend = originals, core, backend
    handle._method_signatures = {name: dual._signature(fn) for name, fn in originals.items()}
    handle._sparse, handle._sparse_signature = reference_module.sparse_attn, dual._signature(reference_module.sparse_attn)
    handle._quantizer, handle._quantizer_signature = reference_module.fp4_act_quant, dual._signature(reference_module.fp4_act_quant)
    handle._instances, handle._geometry, handle._metadata, handle._installed = {}, {}, {}, {}
    handle._compress_installed = {}
    originals_cache = {}
    for layer in range(2, 40):
        inst = model.layers[layer].attn
        expected = dict(layer_id=layer, dim=5120, n_heads=64, n_local_heads=64 // world,
            q_lora_rank=1280, o_lora_rank=1024, head_dim=512, rope_head_dim=64, nope_head_dim=448,
            n_groups=8, n_local_groups=8 // world, window_size=128, compress_ratio=2 if layer < 20 else 1,
            eps=1e-20, softmax_scale=512**-.5, is_kv_source=layer in dual.OWNERS,
            is_index_source=layer in dual.INDEXERS)
        if (type(inst) is not reference_module.Attention
                or any(type(getattr(inst, k, None)) is not type(v) or getattr(inst, k) != v for k, v in expected.items())
                or any(getattr(getattr(inst, name), "__func__", None) is not originals[name] for name in dual.METHODS)
                or hasattr(inst, _PACKED) or hasattr(inst, _SCALES)):
            raise ValueError(f"unsupported official Attention at layer {layer}")
        window = inst.window_kv_cache
        if (window.ndim != 3 or tuple(window.shape[1:]) != (128, 512) or window.dtype != torch.bfloat16
                or window.shape[0] < 1 or not window.is_contiguous() or inst.attn_sink.shape != (64 // world,)
                or inst.attn_sink.dtype != torch.float32 or inst.attn_sink.device != window.device
                or inst.freqs_cis.device != window.device or inst.freqs_cis.ndim != 2 or inst.freqs_cis.shape[1] != 32):
            raise ValueError("unsupported window/sink/frequency geometry")
        dual._no_capture(torch, window.device)
        if layer in dual.OWNERS:
            cache = inst._buffers.get("compress_kv_cache")
            if (cache is None or "compress_kv_cache" not in inst._non_persistent_buffers_set
                    or cache.dtype != torch.bfloat16 or not cache.is_contiguous() or cache.device != window.device
                    or tuple(cache.shape) != (window.shape[0], inst.freqs_cis.shape[0] // inst.compress_ratio, 512)):
                raise ValueError("unsupported BF16 compressed owner cache")
            # Transaction-local tensors only. The handle stores scalar metadata.
            originals_cache[layer] = cache
            handle._cache_metadata[layer] = (tuple(cache.shape), cache.dtype, cache.device)
        handle._instances[layer], handle._geometry[layer] = inst, expected
        handle._metadata[layer] = {name: dual._tensor_metadata(getattr(inst, name))
                                  for name in ("window_kv_cache", "freqs_cis", "attn_sink")}
    if len({id(i) for i in handle._instances.values()}) != 38 or len({id(c) for c in originals_cache.values()}) != 4:
        raise ValueError("distinct Attention instances and compressed owners required")
    if len({(i.window_kv_cache.shape[0], i.freqs_cis.shape[0], i.window_kv_cache.device)
            for i in handle._instances.values()}) != 1:
        raise ValueError("Attention caches must share batch/capacity/device geometry")
    namespace = dict(vars(reference_module))
    namespace.update(_dual_sparse_attn=handle._dual, _packed_publish=handle._publish,
                     _packed_write=handle._write, _packed_prefix=handle._prefix)
    handle._forward_impl = dual._adapt_forward(forward_node, namespace, model_path)
    handle._compress_impl = _adapt_compress(reference_module, namespace, model_path)
    for layer, (shape, _, device) in handle._cache_metadata.items():
        handle._caches[layer] = core.PackedKVCache(shape[0], shape[1], device, layer, handle._instances[layer].compress_ratio)
        handle._valid[layer] = 0
    try:
        for layer, cache in handle._caches.items():
            inst = handle._instances[layer]
            inst.register_buffer(_PACKED, cache.packed, persistent=False)
            inst.register_buffer(_SCALES, cache.scales, persistent=False)
            inst._buffers["compress_kv_cache"] = None
        for layer, inst in handle._instances.items():
            def wrapped(instance, x, start_pos, _layer=layer):
                return handle._forward(_layer, instance, x, start_pos)
            forward = MethodType(wrapped, inst)
            compress = MethodType(handle._compress_impl, inst)
            handle._bindings.append((inst, inst.__dict__.get("forward", _ABSENT), forward,
                                     inst.__dict__.get("_compress_kv", _ABSENT), compress))
            handle._installed[layer], handle._compress_installed[layer] = forward, compress
            inst._compress_kv, inst.forward = compress, forward
    except BaseException:
        for layer, cache in originals_cache.items():
            inst = handle._instances[layer]
            inst._buffers["compress_kv_cache"] = cache
            for name in (_PACKED, _SCALES):
                inst._buffers.pop(name, None)
                inst._non_persistent_buffers_set.discard(name)
        for inst, previous, _, previous_compress, _ in handle._bindings:
            for name, value in (("forward", previous), ("_compress_kv", previous_compress)):
                if value is _ABSENT:
                    inst.__dict__.pop(name, None)
                else:
                    setattr(inst, name, value)
        raise
    handle._clear_shared()
    handle.runtime_sources = {model_path: dual.MODEL_SHA256, kernel_path: dual.KERNEL_SHA256}
    dual._ACTIVE[reference_module] = handle
    handle.active = True
    return handle
