"""Default-off dual-pool attention for the pinned official V4.1 reference.

Only 38 compressed Attention instances are changed. Original cache writes,
indexers, projections, RoPE and merged index order remain intact. The source's
shared global runtime still requires ordered single-model execution; graph
capture and concurrent forwards are not supported. Install after device/dtype
placement. Restore changes methods only and preserves live cache history.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import CodeType, FunctionType, MethodType
import weakref

MODEL_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
KERNEL_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
METHODS = ("forward", "_window_kv", "_compress_kv", "_compress_topk_idxs")
OWNERS = (2, 8, 14, 20)
INDEXERS = (2, 8, 14, 20, 24, 28, 32, 36)
_ACTIVE = weakref.WeakKeyDictionary()  # Independent of the index-cache adapters.
_ABSENT = object()


def _code_child(code, name):
    children = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == name]
    if len(children) != 1:
        raise ValueError(f"ambiguous pinned code: {name}")
    return children[0]


def _compile_method(node, namespace, filename):
    cls = ast.ClassDef(name="Attention", bases=[], keywords=[], body=[node], decorator_list=[])
    ast.copy_location(cls, node)
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    local = {}
    exec(compile(tree, filename, "exec", dont_inherit=True), namespace, local)
    return getattr(local["Attention"], node.name)


def _signature(fn):
    return fn.__code__, fn.__defaults__, fn.__kwdefaults__


def _verify_loaded(fn, rebuilt, full_code, namespace, path):
    if (not isinstance(fn, FunctionType) or fn.__globals__ is not namespace
            or Path(fn.__code__.co_filename).resolve() != path
            or fn.__code__ not in (rebuilt.__code__, full_code)
            or fn.__defaults__ != rebuilt.__defaults__
            or fn.__kwdefaults__ != rebuilt.__kwdefaults__):
        raise ValueError("loaded reference function differs from pinned source/defaults")


def _source_contract(reference):
    path = Path(reference.__file__).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise ValueError("unsupported official model source SHA256")
    tree = ast.parse(raw, filename=str(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
    full_class = _code_child(compile(raw, str(path), "exec", dont_inherit=True), "Attention")
    nodes, originals = {}, {}
    for name in METHODS:
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        fn = getattr(reference.Attention, name)
        rebuilt = _compile_method(copy.deepcopy(node), dict(vars(reference)), str(path))
        _verify_loaded(fn, rebuilt, _code_child(full_class, name), vars(reference), path)
        nodes[name], originals[name] = node, fn
    sparse = reference.sparse_attn
    if not isinstance(sparse, FunctionType):
        raise ValueError("official sparse_attn wrapper required")
    kernel_path = Path(sparse.__globals__["__file__"]).resolve()
    kernel_raw = kernel_path.read_bytes()
    if hashlib.sha256(kernel_raw).hexdigest() != KERNEL_SHA256:
        raise ValueError("unsupported official kernel source SHA256")
    kernel_tree = ast.parse(kernel_raw, filename=str(kernel_path))
    node = next(n for n in kernel_tree.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn")
    namespace = dict(sparse.__globals__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(kernel_path), "exec", dont_inherit=True), namespace)
    full = compile(kernel_raw, str(kernel_path), "exec", dont_inherit=True)
    _verify_loaded(sparse, namespace["sparse_attn"], _code_child(full, "sparse_attn"),
                   sparse.__globals__, kernel_path)
    return nodes["forward"], originals, str(path), str(kernel_path)


def _adapt_forward(node, namespace, filename):
    node = copy.deepcopy(node)
    changed = {"cat": 0, "call": 0}
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, statement):
            if ast.unparse(statement) == "kv = torch.cat([kv, compress_kv], dim=1)":
                changed["cat"] += 1
                return None
            return self.generic_visit(statement)

        def visit_Call(self, call):
            if ast.unparse(call) == "sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)":
                changed["call"] += 1
                new = ast.parse("_dual_sparse_attn(q, kv, compress_kv, self.attn_sink, topk_idxs, self.softmax_scale)", mode="eval").body
                return ast.copy_location(new, call)
            return self.generic_visit(call)
    node = Rewrite().visit(node)
    if changed != {"cat": 1, "call": 1}:
        raise ValueError("pinned attention replacement boundaries changed")
    return _compile_method(node, namespace, filename)


def _tensor_metadata(value):
    # Cache writes are expected; content/version counters are deliberately not sealed.
    return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _no_capture(torch, device):
    if device.type == "cuda":
        if not torch.cuda.is_initialized():
            raise RuntimeError("existing CUDA tensors require an initialized device")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("dual-pool reference adapter does not support CUDA graph capture")


class DualSparseAttentionHandle:
    def __init__(self):
        self.active = False
        self._bindings = []
        self._counts = {"dual_calls": 0}

    @property
    def counters(self):
        return dict(self._counts)

    def _runtime(self):
        if (self._reference.shared_attn is not self._shared or type(self._reference.world_size) is not int
                or self._reference.world_size != self._world or self._model.layers is not self._layers):
            raise RuntimeError("installed reference runtime, TP size, or layer list changed")
        if any(self._layers[layer].attn is not instance for layer, instance in self._instances.items()):
            raise RuntimeError("installed Attention instance or forward changed")
        if (self._reference.sparse_attn is not self._sparse
                or _signature(self._sparse) != self._sparse_signature):
            raise RuntimeError("original sparse wrapper changed")
        for name, fn in self._originals.items():
            if getattr(self._reference.Attention, name) is not fn or _signature(fn) != self._method_signatures[name]:
                raise RuntimeError("original Attention methods changed")

    def _instance(self, layer):
        instance = self._instances[layer]
        if (self._layers[layer].attn is not instance or type(instance) is not self._reference.Attention
                or instance.__dict__.get("forward") is not self._installed[layer]):
            raise RuntimeError("installed Attention instance or forward changed")
        if any(type(getattr(instance, name, None)) is not type(value) or getattr(instance, name) != value
               for name, value in self._geometry[layer].items()):
            raise RuntimeError("installed Attention geometry changed")
        for name in METHODS[1:]:
            if getattr(getattr(instance, name), "__func__", None) is not self._originals[name]:
                raise RuntimeError("installed Attention helper changed")
        for name, metadata in self._metadata[layer].items():
            value = getattr(instance, name)
            if _tensor_metadata(value) != metadata:
                raise RuntimeError("installed Attention tensor device/shape/stride/storage changed")
        return instance

    def _forward(self, layer, instance, x, start_pos):
        self._runtime()
        self._instance(layer)
        torch = self._reference.torch
        _no_capture(torch, x.device)
        if (type(start_pos) is not int or start_pos < 0 or x.ndim != 3 or x.shape[0] < 1
                or x.shape[1] < 1 or x.shape[2] != 5120 or x.dtype != torch.bfloat16
                or x.device != instance.window_kv_cache.device
                or x.shape[0] > instance.window_kv_cache.shape[0]
                or start_pos + x.shape[1] > instance.freqs_cis.shape[0]
                or (start_pos > 0 and x.shape[1] != 1)):
            raise ValueError("unsupported official Attention call geometry/device")
        # Same-stream ordered execution is the reference contract. Forward can
        # mutate the shared slot legitimately; validate its actual owner at use.
        return self._forward_impl(instance, x, start_pos)

    def _dual(self, q, window, compressed, sink, indices, scale):
        published = self._shared.compress_kv
        source = self._source_by_id.get(id(published))
        if source is None:
            raise RuntimeError("shared compressed cache is not an admitted owner")
        owner = self._instance(source)
        if published is not owner.compress_kv_cache:
            raise RuntimeError("shared compressed cache owner changed")
        # The view's real strides must reach the kernel unchanged, including
        # capacity-based batch strides on short active prefixes.
        if (compressed.untyped_storage().data_ptr() != published.untyped_storage().data_ptr()
                or compressed.storage_offset() != published.storage_offset()
                or compressed.stride() != published.stride()
                or compressed.shape[0] != q.shape[0]
                or compressed.shape[1] > published.shape[1]):
            raise RuntimeError("compressed prefix is not the original owner view")
        output = self._core.dual_sparse_attn(q, window, compressed, sink, indices, scale,
                                           backend=self._backend, query_chunk_size=1)
        self._counts["dual_calls"] += 1
        return output

    def restore(self):
        if not self.active:
            return
        self._runtime()
        for layer in self._instances:
            instance = self._instance(layer)
            _no_capture(self._reference.torch, instance.window_kv_cache.device)
        for instance, previous, _ in self._bindings:
            if previous is _ABSENT:
                instance.__dict__.pop("forward", None)
            else:
                instance.forward = previous
        self.active = False
        _ACTIVE.pop(self._reference, None)


def install_reference_dual_sparse_attention(model, reference_module, *, enabled=False, backend="torch"):
    """Install on layer2..39 only; independent of the index-cache adapter registry.

    Source-file and loaded-wrapper validation do not constitute validation of
    TileLang's generated arithmetic/layout or Triton GPU numerics.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit bool")
    handle = DualSparseAttentionHandle()
    if not enabled:
        return handle
    if backend not in ("torch", "triton"):
        raise ValueError("unsupported dual-pool backend")
    if reference_module in _ACTIVE:
        raise RuntimeError("reference runtime already has an active dual-pool adapter")
    forward_node, originals, model_path, kernel_path = _source_contract(reference_module)
    try:
        from . import dsv41_dual_sparse as core
    except ImportError:
        import dsv41_dual_sparse as core
    torch, world = reference_module.torch, reference_module.world_size
    if type(world) is not int or world < 1 or 8 % world or len(model.layers) != 40:
        raise ValueError("unsupported official Attention TP or layer count")
    handle._reference, handle._shared, handle._world = reference_module, reference_module.shared_attn, world
    handle._model, handle._layers = model, model.layers
    handle._originals, handle._core, handle._backend = originals, core, backend
    handle._method_signatures = {name: _signature(fn) for name, fn in originals.items()}
    handle._sparse, handle._sparse_signature = reference_module.sparse_attn, _signature(reference_module.sparse_attn)
    handle._instances, handle._geometry, handle._metadata, handle._installed = {}, {}, {}, {}
    handle._source_by_id = {}
    for layer in range(2, 40):
        instance = model.layers[layer].attn
        expected = dict(layer_id=layer, dim=5120, n_heads=64, n_local_heads=64 // world,
            q_lora_rank=1280, o_lora_rank=1024, head_dim=512, rope_head_dim=64,
            nope_head_dim=448, n_groups=8, n_local_groups=8 // world, window_size=128,
            compress_ratio=2 if layer < 20 else 1, eps=1e-20, softmax_scale=512**-.5,
            is_kv_source=layer in OWNERS, is_index_source=layer in INDEXERS)
        if (type(instance) is not reference_module.Attention
                or any(type(getattr(instance, name, None)) is not type(value) or getattr(instance, name) != value
                       for name, value in expected.items())
                or any(getattr(getattr(instance, name), "__func__", None) is not originals[name] for name in METHODS)):
            raise ValueError(f"unsupported official Attention at layer {layer}")
        window = instance.window_kv_cache
        if (window.ndim != 3 or tuple(window.shape[1:]) != (128, 512) or window.dtype != torch.bfloat16
                or window.shape[0] < 1 or not window.is_contiguous()
                or instance.attn_sink.shape != (64 // world,) or instance.attn_sink.dtype != torch.float32
                or instance.attn_sink.device != window.device or instance.freqs_cis.device != window.device
                or instance.freqs_cis.ndim != 2 or instance.freqs_cis.shape[1] != 32):
            raise ValueError("unsupported official Attention cache/sink/frequency tensors")
        _no_capture(torch, window.device)
        tensor_names = ["window_kv_cache", "attn_sink", "freqs_cis"]
        if layer in OWNERS:
            cache = instance.compress_kv_cache
            if (cache.dtype != torch.bfloat16 or not cache.is_contiguous() or cache.device != window.device
                    or tuple(cache.shape) != (window.shape[0], instance.freqs_cis.shape[0] // instance.compress_ratio, 512)):
                raise ValueError("unsupported compressed owner cache")
            tensor_names.append("compress_kv_cache")
            handle._source_by_id[id(cache)] = layer
        handle._instances[layer], handle._geometry[layer] = instance, expected
        handle._metadata[layer] = {name: _tensor_metadata(getattr(instance, name)) for name in tensor_names}
    if len({id(i) for i in handle._instances.values()}) != 38 or len(handle._source_by_id) != 4:
        raise ValueError("distinct Attention instances and compressed owners required")
    if len({(i.window_kv_cache.shape[0], i.freqs_cis.shape[0], i.window_kv_cache.device)
            for i in handle._instances.values()}) != 1:
        raise ValueError("Attention caches must share batch/capacity/device geometry")
    namespace = dict(vars(reference_module))
    namespace["_dual_sparse_attn"] = handle._dual
    handle._forward_impl = _adapt_forward(forward_node, namespace, model_path)
    try:
        for layer, instance in handle._instances.items():
            def wrapped(inst, x, start_pos, _layer=layer):
                return handle._forward(_layer, inst, x, start_pos)
            installed = MethodType(wrapped, instance)
            handle._bindings.append((instance, instance.__dict__.get("forward", _ABSENT), installed))
            handle._installed[layer] = installed
            instance.forward = installed
    except BaseException:
        for instance, previous, _ in handle._bindings:
            if previous is _ABSENT:
                instance.__dict__.pop("forward", None)
            else:
                instance.forward = previous
        raise
    handle.runtime_sources = {model_path: MODEL_SHA256, kernel_path: KERNEL_SHA256}
    _ACTIVE[reference_module] = handle
    handle.active = True
    return handle
