"""Opt-in compact CED indexer for the pinned official HF reference model.

This patches five *instances*, not a class, model registry, or vLLM backend.
Only long-context, one-token decode is compact. The reference still owns its
global ``shared_attn`` and ordered TP collectives; independent models and
concurrent forwards sharing that global are not supported. CUDA graph capture
is rejected because replay would bypass this Python state machine.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import CodeType, FunctionType, MethodType
import weakref

REFERENCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
SOURCE_LAYER = 20
CONSUMER_LAYERS = (24, 28, 32, 36)
_ACTIVE = weakref.WeakKeyDictionary()
_ABSENT = object()


def _compile_method(node, namespace, filename):
    # Preserve line numbers/qualname for an exact loaded-code/source check.
    cls = ast.ClassDef(name="Indexer", bases=[], keywords=[], body=[node], decorator_list=[])
    ast.copy_location(cls, node)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    local = {}
    exec(compile(module, filename, "exec", dont_inherit=True), namespace, local)
    return local["Indexer"].forward


def _reference_contract(reference):
    path = Path(reference.__file__).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != REFERENCE_SHA256:
        raise ValueError("unsupported HF reference source SHA256")
    tree = ast.parse(source, filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer"]
    if len(classes) != 1:
        raise ValueError("reference Indexer source is ambiguous")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    if len(methods) != 1:
        raise ValueError("reference Indexer.forward source is ambiguous")
    original = reference.Indexer.forward
    if (not isinstance(original, FunctionType) or original.__globals__ is not vars(reference)
            or Path(original.__code__.co_filename).resolve() != path):
        raise ValueError("loaded Indexer.forward is not from the pinned reference module")
    rebuilt = _compile_method(methods[0], dict(vars(reference)), str(path))
    full_module = compile(source, str(path), "exec", dont_inherit=True)
    class_code = next(c for c in full_module.co_consts if isinstance(c, CodeType) and c.co_name == "Indexer")
    full_forward = next(c for c in class_code.co_consts if isinstance(c, CodeType) and c.co_name == "forward")
    # marshal bytes include string-interning/reference-table details that can
    # differ for equivalent code compiled as part of a larger module.
    # CPython can specialize LOAD_ATTR differently when it sees the module's
    # import statements. Permit only the two exact pinned compilations: normal
    # full-module loading, or the source-extracted CPU differential reference.
    if original.__code__ not in (full_forward, rebuilt.__code__):
        raise ValueError("loaded Indexer.forward differs from pinned source")
    # Retain all original key-cache, RoPE, quantization, and weight projection
    # instructions. Only the score/all-reduce/topk suffix is replaced.
    method = methods[0]
    positions = [i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "index_score" for t in n.targets)]
    if not positions or ast.unparse(method.body[positions[0]].value) != \
            "torch.einsum('bshd,btd->bsht', q, index_k)":
        raise ValueError("pinned dense score boundary is missing")
    finish = ast.parse("return _compact_finish(self, q, index_k, weights, x, start_pos, offset)").body[0]
    ast.copy_location(finish, method.body[positions[0]])
    method.body = method.body[:positions[0]] + [finish]
    return original, method, str(path)


def _tensor_identity(value):
    try:
        version = value._version
    except RuntimeError:  # Inference tensors: same-storage mutation is not observable here.
        version = None
    return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, version)


class ReferenceIndexerAdapter:
    """Installation handle. Keep it for counters and explicit ``restore()``."""

    def __init__(self):
        self.active = False
        self._state = None
        self._bindings = []
        self._counts = {"source_compact_steps": 0, "consumer_compact_calls": 0,
                        "dense_fallback_calls": 0}

    @property
    def counters(self):
        return dict(self._counts)

    def restore(self):
        if not self.active:
            return
        if any(instance.__dict__.get("forward") is not installed
               for instance, _, installed in self._bindings):
            raise RuntimeError("an installed Indexer.forward was replaced externally")
        for instance, previous, _ in self._bindings:
            if previous is _ABSENT:
                del instance.__dict__["forward"]
            else:
                instance.forward = previous
        self._state = None
        self.active = False
        _ACTIVE.pop(self._reference, None)

    def _geometry(self, x, start_pos, offset):
        return (type(start_pos) is int and start_pos > 0 and type(offset) is int
                and offset >= 0 and x.ndim == 3 and x.shape[1] == 1
                and x.shape[2] == 5120 and x.shape[0] == 1
                and start_pos + 1 > 2048 * 8)

    def _runtime_matches(self):
        return (self._reference.shared_attn is self._shared
                and self._reference.world_size == self._world)

    def _require_runtime(self):
        if not self._runtime_matches():
            self._state = None
            raise RuntimeError("the installed reference shared runtime or TP size changed")

    def _no_capture(self, x):
        if x.is_cuda:
            with self._reference.torch.cuda.device(x.device):
                if self._reference.torch.cuda.is_current_stream_capturing():
                    self._state = None
                    raise RuntimeError("reference compact indexer does not support CUDA graph capture")

    def _dense(self, original, x, qr, latent, start_pos, offset):
        self._counts["dense_fallback_calls"] += 1
        return original(x, qr, latent, start_pos, offset)

    def _source(self, instance, original, x, qr, latent, start_pos, offset):
        self._no_capture(x)
        self._require_runtime()
        self._state = None  # A short step, new prefill, or failed source invalidates old IDs.
        if not self._geometry(x, start_pos, offset):
            return self._dense(original, x, qr, latent, start_pos, offset)
        step = (start_pos, x.shape[0], start_pos + 1, offset)

        def select(scores, compress_lens, topk_blocks, block_size):
            ids, mask = self._core.select_candidate_ids(
                scores, compress_lens, topk_blocks, block_size, return_mask=True)
            self._state = {"step": step, "ids": ids, "mask": mask,
                           "cache": _tensor_identity(self._shared.index_k),
                           "ids_identity": _tensor_identity(ids),
                           "mask_identity": _tensor_identity(mask), "next_consumer": 0}
            return mask

        namespace = dict(self._original.__globals__)
        namespace["select_candidate_blocks"] = select
        source = FunctionType(self._original.__code__, namespace,
                              self._original.__name__, self._original.__defaults__)
        try:
            output = source(instance, x, qr, latent, start_pos, offset)
            if self._state is None or self._shared.candidates is not self._state["mask"]:
                raise RuntimeError("source did not publish the selected compact candidates")
        except BaseException:
            self._state = None
            raise
        self._counts["source_compact_steps"] += 1
        return output

    def _consumer(self, layer, instance, original, x, qr, latent, start_pos, offset):
        self._no_capture(x)
        self._require_runtime()
        eligible = self._geometry(x, start_pos, offset)
        state = self._state
        valid = (eligible
                 and state is not None and state["step"] == (start_pos, x.shape[0], start_pos + 1, offset)
                 and state["next_consumer"] < len(CONSUMER_LAYERS)
                 and layer == CONSUMER_LAYERS[state["next_consumer"]]
                 and self._shared.candidates is state["mask"]
                 and state["cache"] == _tensor_identity(self._shared.index_k)
                 and state["ids_identity"] == _tensor_identity(state["ids"])
                 and state["mask_identity"] == _tensor_identity(state["mask"]))
        if not valid:
            self._state = None
            if eligible and self._world > 1:
                # A rank-local dense fallback would change its collective
                # width S while peers still reduce compact width C.
                raise RuntimeError("compact state mismatch before the TP collective")
            return self._dense(original, x, qr, latent, start_pos, offset)
        try:
            output = self._prefix(instance, x, qr, latent, start_pos, offset)
        except BaseException:
            self._state = None
            raise
        state["next_consumer"] += 1
        self._counts["consumer_compact_calls"] += 1
        if state["next_consumer"] == len(CONSUMER_LAYERS):
            self._state = None
        return output

    def _finish(self, instance, q, index_k, weights, x, start_pos, offset):
        state = self._state
        reduce_fn = self._reference.dist.all_reduce if self._world > 1 else None
        scores = self._core.compact_index_scores(
            q, index_k, weights, state["ids"], reduce_fn,
            query_chunk_size=1, backend=self._backend)
        # Preserve the original dense-width topk/tie domain, including -inf
        # masked positions. Compact-space topk would change tied selections.
        return self._core.compact_topk(
            scores, state["ids"], start_pos + 1, offset, instance.index_topk,
            full_width=index_k.shape[1], query_chunk_size=1)


def install_reference_indexer(model, reference_module, *, enabled=False, backend="torch"):
    """Install only on the official 40-layer reference architecture, default off.

    This is not a vLLM registration. Source layer 20 still computes dense scores
    once and publishes the exact dense mask for original-path fallbacks. Only
    layers 24/28/32/36 score gathered positions. One shared runtime has one active
    adapter; the reference's existing single-model/ordered-forward restriction
    remains in force. No CUDA context is initialized by installation.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be an explicit bool")
    handle = ReferenceIndexerAdapter()
    if not enabled:
        return handle
    if backend not in ("torch", "triton"):
        raise ValueError("unsupported compact backend")
    if reference_module in _ACTIVE:
        raise RuntimeError("this reference runtime already has an active adapter")
    original, prefix_node, filename = _reference_contract(reference_module)
    try:
        from . import dsv41_indexer as core
    except ImportError:
        import dsv41_indexer as core

    world = reference_module.world_size
    if type(world) is not int or world < 1 or 32 % world or len(model.layers) != 40:
        raise ValueError("unsupported reference TP or layer count")
    instances = []
    for layer in (SOURCE_LAYER, *CONSUMER_LAYERS):
        instance = model.layers[layer].attn.indexer
        expected = {"compress_ratio": 1, "dim": 5120, "n_heads": 32,
                    "n_local_heads": 32 // world, "index_head_dim": 128,
                    "rope_head_dim": 64, "index_topk": 512, "q_lora_rank": 1280,
                    "candidate_topk_blocks": 2048, "candidate_block_size": 8,
                    "owns_k": layer == SOURCE_LAYER, "is_candidate_source": layer == SOURCE_LAYER,
                    "uses_candidates": layer != SOURCE_LAYER}
        if (type(instance) is not reference_module.Indexer
                or getattr(instance.forward, "__func__", None) is not original
                or any(type(getattr(instance, k, None)) is not type(v)
                       or getattr(instance, k) != v for k, v in expected.items())):
            raise ValueError(f"unsupported reference Indexer at layer {layer}")
        instances.append((layer, instance))
    if len({id(instance) for _, instance in instances}) != len(instances):
        raise ValueError("each CED layer must own a distinct Indexer instance")
    handle._reference, handle._shared, handle._world = reference_module, reference_module.shared_attn, world
    handle._original, handle._core, handle._backend = original, core, backend
    namespace = dict(vars(reference_module))
    namespace["_compact_finish"] = handle._finish
    handle._prefix = _compile_method(prefix_node, namespace, filename)
    for layer, instance in instances:
        bound_original = instance.forward
        if layer == SOURCE_LAYER:
            def wrapped(self, x, qr, latent, start_pos, offset, _old=bound_original):
                return handle._source(self, _old, x, qr, latent, start_pos, offset)
        else:
            def wrapped(self, x, qr, latent, start_pos, offset, _old=bound_original, _layer=layer):
                return handle._consumer(_layer, self, _old, x, qr, latent, start_pos, offset)
        installed = MethodType(wrapped, instance)
        handle._bindings.append((instance, instance.__dict__.get("forward", _ABSENT), installed))
        instance.forward = installed
    _ACTIVE[reference_module] = handle
    handle.active = True
    return handle
