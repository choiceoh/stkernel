"""Instance-only packed index cache for the pinned official HF reference.

Default off; this is not a vLLM model registration. Install after device placement
at an explicit fresh-prefill boundary. Restore discards history and requires a
new prefill; it never retains the displaced BF16 caches. The reference's global
shared_attn and ordered, single-model forwards remain the concurrency contract.
CUDA graph capture is unsupported. Numerical validation covers finite inputs;
non-finite producer behavior is not established, and no per-step host tensor
inspection/synchronization is added.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import CodeType, FunctionType, MethodType

try:
    from . import dsv41_reference_adapter as reference_adapter
except ImportError:
    import dsv41_reference_adapter as reference_adapter

KERNEL_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)
OWNERS = (2, 8, 14, 20)
CONSUMERS = (24, 28, 32, 36)
_PACKED = "_dsv41_index_packed"
_SCALES = "_dsv41_index_scales"
_ABSENT = object()


def _kernel_contract(reference):
    fn = reference.fp4_act_quant
    if not isinstance(fn, FunctionType):
        raise ValueError("official fp4_act_quant function required")
    path = Path(fn.__globals__["__file__"]).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != KERNEL_SHA256:
        raise ValueError("unsupported HF quantizer source SHA256")
    if Path(fn.__code__.co_filename).resolve() != path:
        raise ValueError("loaded quantizer source path mismatch")
    tree = ast.parse(raw, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fp4_act_quant"]
    if len(nodes) != 1:
        raise ValueError("ambiguous official quantizer")
    namespace = dict(fn.__globals__)
    exec(compile(ast.Module(body=[nodes[0]], type_ignores=[]), str(path), "exec",
                 dont_inherit=True), namespace)
    full = compile(raw, str(path), "exec", dont_inherit=True)
    full_code = next(c for c in full.co_consts if isinstance(c, CodeType) and c.co_name == fn.__name__)
    rebuilt = namespace[fn.__name__]
    if (fn.__code__ not in (rebuilt.__code__, full_code)
            or fn.__defaults__ != rebuilt.__defaults__ or fn.__kwdefaults__ != rebuilt.__kwdefaults__):
        raise ValueError("loaded fp4_act_quant differs from pinned source")
    return str(path)


def _compiled_paths(reference, handle, filename):
    """Change only the pinned producer/store and score boundaries, not math order."""
    tree = ast.parse(Path(filename).read_bytes(), filename=filename)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    score_at = next(i for i, n in enumerate(method.body)
                    if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "index_score")
    reduce_at = score_at + 2
    if ast.unparse(method.body[reduce_at].test) != "world_size > 1":
        raise ValueError("official score collective boundary changed")
    tail = ast.parse("def _packed_tail(self, index_score, x, start_pos, offset, end_pos, seqlen, ratio):\n pass").body[0]
    tail.body = copy.deepcopy(method.body[reduce_at + 1:])
    prefix = method.body[:score_at]
    producer = next(n for n in prefix if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "self.owns_k and latent is not None")
    quant_at = next(i for i, n in enumerate(producer.body)
                    if isinstance(n, ast.Expr) and ast.unparse(n.value) == "fp4_act_quant(k, fp4_block_size, True)")
    if [ast.unparse(n) for n in producer.body[quant_at + 1:]] != [
            "self.k_cache[:bsz, start_pos // ratio:start_pos // ratio + k.size(1)] = k",
            "shared_attn.index_k = self.k_cache"]:
        raise ValueError("official cache publication boundary changed")
    producer.body = producer.body[:quant_at] + ast.parse(
        "_packed_y, _packed_sf = fp4_act_quant(k, fp4_block_size, False, scale_dtype=torch.float8_e8m0fnu)\n"
        "_packed_publish(self, _packed_y, _packed_sf, bsz, start_pos // ratio, k.size(1))").body
    prefix = [n for n in prefix if not (isinstance(n, ast.Assign)
              and ast.unparse(n.targets[0]) == "index_k")]
    method.body = prefix + ast.parse(
        "return _packed_finish(self, q, weights, x, start_pos, offset, bsz, seqlen, ratio, end_pos)").body
    namespace = dict(vars(reference))
    namespace.update(_packed_publish=handle._publish,
                     _packed_finish=handle._finish, select_candidate_blocks=handle._select)
    prefix_fn = reference_adapter._compile_method(method, namespace, filename)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tail], type_ignores=[])), filename,
                 "exec", dont_inherit=True), namespace)
    return prefix_fn, namespace["_packed_tail"]


def _storage_identity(tensor):
    # Writes change versions; allocation/shape/device changes are never accepted.
    return reference_adapter._tensor_identity(tensor)[:-1]


def _no_capture_device(torch, device):
    if device.type == "cuda":
        # A real CUDA cache is already materialized. Never create a context just
        # to inspect installation eligibility or select an unrelated device.
        if not torch.cuda.is_initialized():
            raise RuntimeError("CUDA cache installation requires an initialized device")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("packed reference adapter does not support CUDA graph capture")


class PackedIndexerHandle:
    def __init__(self):
        self.active = False
        self.reset_pending = False
        self._bindings = []
        self._caches = {}
        self._metadata = {}
        self._valid = {}
        self._state = None
        self._step = None
        self._last_layer = -1
        self._failed = False
        self._counts = dict(packed_writes=0, dense_calls=0,
                            source_compact_steps=0, consumer_compact_calls=0)

    @property
    def counters(self):
        return dict(self._counts)

    def _runtime(self):
        if (self._reference.shared_attn is not self._shared
                or type(self._reference.world_size) is not int
                or self._reference.world_size != self._world):
            raise RuntimeError("installed shared runtime or TP size changed")

    def _no_capture(self, x):
        _no_capture_device(self._reference.torch, x.device)

    def _check_buffers(self):
        for layer, cache in self._caches.items():
            cache.validate()
            instance = self._instances[layer]
            if (instance._buffers.get("k_cache") is not None
                    or instance._buffers.get(_PACKED) is not cache.packed
                    or instance._buffers.get(_SCALES) is not cache.scales
                    or (_storage_identity(cache.packed), _storage_identity(cache.scales)) != self._identities[layer]):
                raise RuntimeError("installed packed buffer allocation changed")

    def _clear_shared(self):
        self._shared.index_k = None
        self._shared.candidates = None
        if hasattr(self._shared, "topk_idxs"):
            self._shared.topk_idxs = None
        self._state = None

    def _begin(self, layer, x, start_pos, offset):
        self._runtime()
        self._no_capture(x)
        self._check_buffers()
        if self._failed:
            raise RuntimeError("packed forward failed; restore/reset and fresh prefill required")
        if (type(start_pos) is not int or start_pos < 0 or type(offset) is not int or offset < 0
                or x.ndim != 3 or x.shape[2] != 5120 or x.shape[0] < 1 or x.shape[1] < 1
                or x.dtype != self._reference.torch.bfloat16):
            raise ValueError("unsupported reference indexer input")
        step = (start_pos, x.shape[0], x.shape[1], offset)
        if step != self._step or layer <= self._last_layer:
            if start_pos == 0:
                if self._step is not None and self._last_layer != 36:
                    raise RuntimeError("previous Indexer step is incomplete")
                self._valid = {owner: 0 for owner in OWNERS}
                self._clear_shared()
            elif (self._step is None or self._last_layer != 36
                  or start_pos != self._step[0] + self._step[2]
                  or x.shape[0] != self._step[1] or x.shape[1] != 1):
                raise RuntimeError("fresh prefill or contiguous ordered decode required")
            self._step = step
            self._last_layer = -1
        active_layers = LAYERS if start_pos + x.shape[1] >= 2 else (20, *CONSUMERS)
        expected_layer = active_layers[0] if self._last_layer == -1 else (
            active_layers[active_layers.index(self._last_layer) + 1] if self._last_layer != 36 else None)
        if layer != expected_layer:
            raise RuntimeError("Indexer layer order changed")
        for owner, cache in self._caches.items():
            if (x.device != cache.packed.device or x.shape[0] > cache.batch_size
                    or (start_pos + x.shape[1]) // cache.ratio > cache.capacity):
                raise ValueError("input exceeds packed cache device/capacity")
        self._last_layer = layer
        if layer == 20:
            self._state = None

    def _publish(self, instance, y, sf, batch, start, rows):
        layer = self._layer_ids[id(instance)]
        cache = self._caches[layer]
        if start != self._valid[layer] or y.shape[0] != batch:
            raise RuntimeError("non-contiguous packed cache write")
        self._packed.write_packed_index(cache, y, sf, start_slot=start, rows=rows)
        self._valid[layer] = start + rows
        # Crucially, this remains inside owns_k AND latent-is-not-None.
        self._shared.index_k = cache
        self._counts["packed_writes"] += 1

    def _eligible(self, x, start_pos):
        return (self._compact and start_pos > 0 and x.shape[0] == x.shape[1] == 1
                and start_pos + 1 > 2048 * 8)

    def _cache_identity(self, cache):
        return (id(cache), cache.generation, reference_adapter._tensor_identity(cache.packed),
                reference_adapter._tensor_identity(cache.scales))

    def _select(self, scores, compress_lens, topk_blocks, block_size):
        if self._select_compact:
            ids, mask = self._core.select_candidate_ids(scores, compress_lens, topk_blocks,
                                                       block_size, return_mask=True)
            self._state = dict(step=self._step, ids=ids, mask=mask,
                               ids_identity=reference_adapter._tensor_identity(ids),
                               mask_identity=reference_adapter._tensor_identity(mask),
                               cache=self._cache_identity(self._shared.index_k), next_consumer=0)
            self._counts["source_compact_steps"] += 1
            return mask
        return self._reference.select_candidate_blocks(scores, compress_lens, topk_blocks, block_size)

    def _finish(self, instance, q, weights, x, start_pos, offset, batch, seqlen, ratio, end_pos):
        cache = self._shared.index_k
        if (not any(cache is item for item in self._caches.values())
                or end_pos // ratio > self._valid[cache.owner_layer]):
            raise RuntimeError("shared packed index slot is absent, stale, or too short")
        width = end_pos // ratio
        reduce_fn = self._reference.dist.all_reduce if self._world > 1 else None
        layer = self._layer_ids[id(instance)]
        compact = instance.uses_candidates and self._eligible(x, start_pos)
        state = self._state
        if compact:
            valid = (state is not None and state["step"] == self._step
                     and state["next_consumer"] < 4 and layer == CONSUMERS[state["next_consumer"]]
                     and state["cache"] == self._cache_identity(cache)
                     and self._shared.candidates is state["mask"]
                     and state["ids_identity"] == reference_adapter._tensor_identity(state["ids"])
                     and state["mask_identity"] == reference_adapter._tensor_identity(state["mask"]))
            if not valid:
                # Never switch rank-local C to S collective width on a TP mismatch.
                raise RuntimeError("compact packed state mismatch before collective")
        scores = self._packed.packed_index_scores(q, cache, weights, width=width,
                    ids=state["ids"] if compact else None, reduce_fn=reduce_fn,
                    query_chunk_size=1, backend=self._backend)
        if compact:
            result = self._core.compact_topk(scores, state["ids"], end_pos // ratio, offset,
                        instance.index_topk, full_width=width, query_chunk_size=1)
            state["next_consumer"] += 1
            self._counts["consumer_compact_calls"] += 1
            if state["next_consumer"] == 4:
                self._state = None
            return result
        self._counts["dense_calls"] += 1
        self._select_compact = instance.is_candidate_source and self._eligible(x, start_pos)
        try:
            return self._tail(instance, scores, x, start_pos, offset, end_pos, seqlen, ratio)
        finally:
            self._select_compact = False

    def _forward(self, layer, instance, x, qr, latent, start_pos, offset):
        try:
            self._begin(layer, x, start_pos, offset)
            return self._prefix(instance, x, qr, latent, start_pos, offset)
        except BaseException:
            self._state = None
            self._failed = True
            raise

    def restore(self, *, reset=False):
        """Discard packed history, allocate fresh BF16 caches, then require start=0.

        Allocation failure leaves the packed installation active and unchanged.
        No full-cache dequantization or retained old BF16 allocation is involved.
        """
        if not self.active:
            return
        if reset is not True:
            raise ValueError("restore requires reset=True and a subsequent fresh prefill")
        self._runtime()
        if any(instance.__dict__.get("forward") is not installed
               for instance, _, installed in self._bindings):
            raise RuntimeError("installed Indexer.forward was replaced externally")
        self._check_buffers()
        torch = self._reference.torch
        for _, _, device in self._metadata.values():
            _no_capture_device(torch, device)
        # Complete potentially failing allocations before any runtime mutation.
        fresh = {layer: torch.zeros(shape, dtype=dtype, device=device)
                 for layer, (shape, dtype, device) in self._metadata.items()}
        for layer, tensor in fresh.items():
            instance = self._instances[layer]
            instance._buffers["k_cache"] = tensor
            instance._buffers.pop(_PACKED)
            instance._buffers.pop(_SCALES)
            instance._non_persistent_buffers_set.discard(_PACKED)
            instance._non_persistent_buffers_set.discard(_SCALES)
        self._clear_shared()
        self._caches.clear()
        self._identities.clear()
        self.active = False
        self.reset_pending = True
        # Retain registry ownership until a caller actually begins fresh history.
        for instance, _, _ in self._bindings:
            def guard(inst, x, qr, latent, start_pos, offset):
                self._runtime()
                self._no_capture(x)
                if start_pos != 0:
                    raise RuntimeError("restored caches require a fresh prefill at start_pos=0")
                self._remove_guards()
                return self._original(inst, x, qr, latent, start_pos, offset)
            instance.forward = MethodType(guard, instance)
        self._reset_guards = [(i, i.__dict__["forward"]) for i, _, _ in self._bindings]

    def _remove_guards(self):
        if any(i.__dict__.get("forward") is not guard for i, guard in self._reset_guards):
            raise RuntimeError("restored forward guard was replaced externally")
        for instance, previous, _ in self._bindings:
            if previous is _ABSENT:
                instance.__dict__.pop("forward", None)
            else:
                instance.forward = previous
        self.reset_pending = False
        reference_adapter._ACTIVE.pop(self._reference, None)


def install_reference_packed_indexer(model, reference_module, *, enabled=False,
        cache_boundary="fresh_prefill", compact_decode=True, backend="torch"):
    """Install on all eight pinned reference Indexer instances, explicitly opt-in."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit bool")
    handle = PackedIndexerHandle()
    if not enabled:
        return handle
    if cache_boundary != "fresh_prefill" or type(compact_decode) is not bool or backend not in ("torch", "triton"):
        raise ValueError("unsupported cache boundary, compact selection, or backend")
    if reference_module in reference_adapter._ACTIVE:
        raise RuntimeError("reference runtime already has an active adapter or reset guard")
    original, _, filename = reference_adapter._reference_contract(reference_module)
    kernel_path = _kernel_contract(reference_module)
    try:
        from . import dsv41_indexer as core, dsv41_packed_index as packed
    except ImportError:
        import dsv41_indexer as core
        import dsv41_packed_index as packed
    torch = reference_module.torch
    world = reference_module.world_size
    if type(world) is not int or world < 1 or 32 % world or len(model.layers) != 40 or reference_module.fp4_block_size != 32:
        raise ValueError("unsupported reference geometry or TP size")
    instances = {}
    for layer in LAYERS:
        inst = model.layers[layer].attn.indexer
        expected = dict(compress_ratio=2 if layer < 20 else 1, dim=5120, n_heads=32,
            n_local_heads=32 // world, index_head_dim=128, rope_head_dim=64,
            index_topk=512, q_lora_rank=1280, candidate_topk_blocks=2048,
            candidate_block_size=8, owns_k=layer in OWNERS,
            is_candidate_source=layer == 20, uses_candidates=layer > 20)
        if (type(inst) is not reference_module.Indexer
                or getattr(inst.forward, "__func__", None) is not original
                or any(type(getattr(inst, key, None)) is not type(value) or getattr(inst, key) != value
                       for key, value in expected.items())
                or hasattr(inst, _PACKED) or hasattr(inst, _SCALES)):
            raise ValueError(f"unsupported Indexer at layer {layer}")
        instances[layer] = inst
    if len({id(i) for i in instances.values()}) != 8:
        raise ValueError("Indexer instances must be distinct")
    handle._reference, handle._shared, handle._world = reference_module, reference_module.shared_attn, world
    handle._instances, handle._layer_ids = instances, {id(i): layer for layer, i in instances.items()}
    handle._original, handle._core, handle._packed = original, core, packed
    handle._backend, handle._compact = backend, compact_decode
    handle.runtime_sources = {filename: reference_adapter.REFERENCE_SHA256, kernel_path: KERNEL_SHA256}
    handle._identities = {}
    originals = {}
    for layer in OWNERS:
        inst = instances[layer]
        cache = inst._buffers.get("k_cache")
        if (cache is None or cache.ndim != 3 or cache.shape[-1] != 128
                or cache.dtype != torch.bfloat16 or not cache.is_contiguous()
                or "k_cache" not in inst._non_persistent_buffers_set):
            raise ValueError("expected the official nonpersistent BF16 index cache")
        handle._metadata[layer] = (tuple(cache.shape), cache.dtype, cache.device)
        originals[layer] = cache  # Transaction only; never attached to handle/closure.
        _no_capture_device(torch, cache.device)
    full_shape, _, device = handle._metadata[20]
    if any(m[0][0] != full_shape[0] or m[0][1] != full_shape[1] // instances[layer].compress_ratio
           or m[2] != device for layer, m in handle._metadata.items()):
        raise ValueError("owner cache capacities/devices must describe the same model")
    handle._prefix, handle._tail = _compiled_paths(reference_module, handle, filename)
    for layer, (shape, _, device) in handle._metadata.items():
        handle._caches[layer] = packed.allocate_packed_index_cache(shape[0], shape[1],
                               device=device, owner_layer=layer, ratio=instances[layer].compress_ratio)
        handle._valid[layer] = 0
    try:
        for layer in OWNERS:
            inst, cache = instances[layer], handle._caches[layer]
            inst.register_buffer(_PACKED, cache.packed, persistent=False)
            inst.register_buffer(_SCALES, cache.scales, persistent=False)
            inst._buffers["k_cache"] = None
            handle._identities[layer] = (_storage_identity(cache.packed), _storage_identity(cache.scales))
        for layer, instance in instances.items():
            def wrapped(inst, x, qr, latent, start_pos, offset, _layer=layer):
                return handle._forward(_layer, inst, x, qr, latent, start_pos, offset)
            installed = MethodType(wrapped, instance)
            handle._bindings.append((instance, instance.__dict__.get("forward", _ABSENT), installed))
            instance.forward = installed
    except BaseException:
        for layer, cache in originals.items():
            inst = instances[layer]
            inst._buffers["k_cache"] = cache
            for name in (_PACKED, _SCALES):
                inst._buffers.pop(name, None)
                inst._non_persistent_buffers_set.discard(name)
        for inst, previous, _ in handle._bindings:
            if previous is _ABSENT:
                inst.__dict__.pop("forward", None)
            else:
                inst.forward = previous
        raise
    handle._clear_shared()
    reference_adapter._ACTIVE[reference_module] = handle
    handle.active = True
    return handle
