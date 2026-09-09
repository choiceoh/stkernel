"""Required TP SF6 Q0 startup comparison using one real packed weight owner.

Import is CPU-inert. This is a bounded numerical/graph canary, not throughput
or sanitizer acceptance. The existing EP runtime/fixtures are never invoked.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import struct
from threading import RLock
import time

# Reuse only the established arithmetic acceptance and bounded diagnostics.
# Their .02/.04 floors and 3*stock-noise limits must not diverge by backend.
from .glm53_ep_local_selftest import (
    check_control, compare, _tensor_identity, _inputs, _memory,
    _cache_snapshot, _restore_scratch_caches,
)

SEED = 905321
CASES = (("balanced4096", 4096, "balanced"),
         ("concentrated6912", 6912, "concentrated"),
         ("zeros4097", 4097, "zeros"),
         ("duplicate8192", 8192, "duplicate"))
_LOCK = RLock()
_STATES = {}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _tensor_state(tensor):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError as exc:
        if "version counter" not in str(exc):
            raise
        version = "inference-no-version-counter"
    return (id(tensor), tensor.data_ptr(), version, tuple(tensor.shape),
            tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))


def _backings(experts, layer):
    views = experts._sf6_weight_views
    return dict(w13=layer.w13_weight, w2=layer.w2_weight,
                sf1=views.sfb1_packed, sf2=views.sfb2_packed,
                fc1_alpha=views.w1_alpha, fc2_alpha=views.w2_alpha,
                fc1_input=experts.g1_alphas, fc2_input=experts._fc2_input_scale)


def tp_sf6_q0_selftest_eligible(experts, layer):
    """Pure owner selection; repeated finalization may select a cached owner."""
    views = getattr(experts, "_sf6_weight_views", None)
    return bool(not getattr(experts, "_use_ep", True)
                and getattr(experts, "_sf6_finalized", False)
                and views is not None and getattr(views, "tiled", False)
                and getattr(views, "packed_only", False)
                and getattr(getattr(views, "reform_scales", None), "enabled", False)
                and tuple(getattr(layer.w13_weight, "shape", ())) == (288,1024,2048)
                and tuple(getattr(layer.w2_weight, "shape", ())) == (288,4096,256)
                and (getattr(experts,"global_num_experts",None),getattr(experts,"num_local_experts",None),
                     getattr(experts,"hidden_dim",None),getattr(experts,"intermediate_size_per_partition",None),
                     getattr(experts,"topk",None)) == (288,288,4096,512,8))


def _caller_state(experts, layer):
    views = experts._sf6_weight_views
    fields = ("w1_scale", "w2_scale", "w1_sf_mma", "w2_sf_mma",
              "g1_alphas", "g2_alphas", "_fc2_input_scale")
    return dict(wrapper=id(experts._wrapper), owner=id(views),
                generation=experts._sf6_generation,
                tensors={name: _tensor_state(value) for name, value in _backings(experts, layer).items()},
                views={name: _tensor_state(getattr(views, name)) for name in
                       ("w13_fp4", "down_fp4", "sfb1_packed", "sfb2_packed")},
                caller={name: _tensor_state(getattr(experts, name, None)) for name in fields},
                raw_parameters=(layer.w13_weight_scale, layer.w2_weight_scale))


def _runtime(experts, layer):
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_gated_sf6_q0 as q0
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import _b12x_require_packed_owner
    device = layer.w13_weight.device
    if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1)
            or torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("TP Q0 self-test requires SM121 outside graph capture")
    if (experts._use_ep or experts._wrapper is not None
            or not experts._sf6_finalized
            or (experts.global_num_experts, experts.num_local_experts,
                experts.hidden_dim, experts.intermediate_size_per_partition, experts.topk)
               != (288, 288, 4096, 512, 8)
            or (experts._activation_str, experts._swiglu_alpha,
                experts._swiglu_beta, experts._swiglu_limit)
               != ("swigluoai_uninterleave", 1., 0., 10.)):
        raise RuntimeError("TP Q0 self-test requires the finalized pre-inference TP owner")
    if not md._TP_SF6_Q0_ENABLED or not q0.stock_contract_matches():
        raise RuntimeError("TP Q0 self-test selection/source contract mismatch")
    views = experts._sf6_weight_views
    _b12x_require_packed_owner(views)
    if (not views.tiled or tuple(layer.w13_weight.shape) != (288, 1024, 2048)
            or tuple(layer.w2_weight.shape) != (288, 4096, 256)
            or any(value is not None for value in (experts.w1_scale, experts.w2_scale,
                   experts.w1_sf_mma, experts.w2_sf_mma,
                   layer.w13_weight_scale, layer.w2_weight_scale))):
        raise RuntimeError("TP Q0 self-test cannot reconstruct or retain raw SF6 scales")
    for tensor in _backings(experts, layer).values():
        if tensor is None or tensor.device != device or not tensor.is_contiguous():
            raise RuntimeError("TP Q0 hashing requires the original contiguous backing tensors")
    from . import glm53_ep_local_selftest as numerical
    files = dict(selftest=Path(__file__), numerical=Path(numerical.__file__),
                 dispatch=Path(md.__file__), candidate=Path(q0.__file__),
                 sf6=Path(q0._sf6.__file__), stock=Path(q0._sf6._stock.__file__),
                 wrapper=Path(inspect.getfile(type(experts))))
    source = {name: dict(path=str(path), sha256=_sha(path.read_bytes())) for name, path in files.items()}
    import flashinfer
    distribution = importlib.metadata.distribution("cuda-bindings")
    metadata = [Path(distribution.locate_file(p)) for p in distribution.files or ()
                if p.name == "METADATA" and p.parent.name.startswith("cuda_bindings-")]
    if len(metadata) != 1:
        raise RuntimeError("TP Q0 cannot identify CUDA bindings metadata")
    bindings = importlib.import_module("cuda.bindings")
    versions = {}
    for name, module in (("torch",torch),("flashinfer",flashinfer),("cuda.bindings",bindings)):
        filename = getattr(module,"__file__",None)
        versions[name] = dict(version=str(getattr(module,"__version__","unknown")),
                              path=filename,sha256=_sha(Path(filename).read_bytes()) if filename else None)
    versions["cuda.bindings"].update(version=distribution.version,
        metadata_path=str(metadata[0]),metadata_sha256=_sha(metadata[0].read_bytes()))
    return torch, md, device, dict(source=source, versions=versions)


def _route_rows(rows, kind, changed=False):
    """All TP IDs remain valid; zero router weights still allocate Q0 rows."""
    if kind not in ("balanced", "concentrated", "zeros", "duplicate"):
        raise ValueError("unknown TP Q0 fixture")
    offset = 137 if changed else 0
    return [[(6 if changed else 5) if kind == "duplicate" else
             (slot + (3 if changed else 0)) % 288 if kind == "concentrated" else
             (token*37 + slot*31 + offset) % 288 for slot in range(8)]
            for token in range(rows)]


def _routing(torch, rows, kind, generator, device, changed):
    ids = torch.tensor(_route_rows(rows, kind, changed), dtype=torch.int32, device=device)
    values = torch.rand(rows, 8, dtype=torch.bfloat16, generator=generator, device=device)
    values /= values.sum(dim=1, keepdim=True)
    weights = values.float()
    if kind == "zeros":
        parity = int(changed)
        weights[parity::2, :] = 0.
        weights[1-parity::2, ::2] = 0.
        weights[1-parity, 0] = -0.
    return ids, weights


def _weight_bits(value):
    return struct.pack("<f", value).hex()


def _route_metadata(counts, bases, tokens, weights, input_ids, input_weights):
    """Validate every route including duplicates and both signed zeros."""
    rows = len(input_ids)
    expected = Counter((expert, token, _weight_bits(weight))
                       for token, (ids, values) in enumerate(zip(input_ids, input_weights))
                       for expert, weight in zip(ids, values))
    if len(counts) != 288 or len(bases) != 289 or any(not 0 <= e < 288 for e, _, _ in expected):
        raise AssertionError("TP Q0 route metadata geometry mismatch")
    observed = Counter()
    sample = []
    selected = {0, rows//2, rows-1}
    tile_base = 0
    for expert, count in enumerate(counts):
        if not 0 <= count <= rows*8 or bases[expert] != tile_base:
            raise AssertionError("TP Q0 count/prefix mismatch")
        tile_base += (count + 127)//128
        for physical in range(bases[expert]*128, bases[expert]*128 + count):
            if physical >= len(tokens) or physical >= len(weights):
                raise AssertionError("TP Q0 token map escaped workspace")
            token, weight = tokens[physical], weights[physical]
            if not 0 <= token < rows:
                raise AssertionError("TP Q0 token map escaped input")
            record = (expert, token, _weight_bits(weight))
            observed[record] += 1
            if token in selected:
                sample.append((record, physical))
    if bases[-1] != tile_base or observed != expected or sum(counts) != rows*8:
        raise AssertionError("TP Q0 route multiset differs from all original top8 routes")
    return sorted(sample)


def _q0(torch, workspace, ids, weights):
    counts = workspace.row_counts.cpu().tolist()
    bases = workspace.expert_tile_base.cpu().tolist()
    sample = _route_metadata(counts, bases, workspace.token_map.cpu().tolist(),
                             workspace.token_weights.cpu().tolist(),
                             ids.cpu().tolist(), weights.cpu().tolist())
    packed = workspace.packed_input.reshape(-1, 2048)
    sf = workspace.scale_flat
    records = []
    for key, physical in sample:
        base = (physical//128)*32768 + (physical%32)*16 + ((physical//32)%4)*4
        offsets = [base + (block//4)*512 + block%4 for block in range(256)]
        if max(offsets) >= sf.numel():
            raise AssertionError("TP Q0 scale row escaped workspace")
        scale_indices = torch.tensor(offsets, dtype=torch.int64, device=sf.device)
        records.append((*key, _tensor_identity(packed[physical])["sha256"],
                        _tensor_identity(sf.index_select(0, scale_indices))["sha256"]))
    records.sort()
    return dict(row_counts=counts, routes=sum(counts), payload_sample=records,
                scope="all route IDs/weights/counts/prefixes; A/SFA bytes only for tokens 0,T//2,T-1")


def _selected_keys(md):
    keys = [key for key in md._DYNAMIC_KERNEL_CACHE
            if len(key) >= 19 and key[0] == "dynamic" and key[3:7] == (288,4096,512,8)
            and "sf6_direct_prefill_v1" in key]
    candidates = [key for key in keys if key[-1] == "glm53_tp_sf6_q0_v1"]
    if not candidates:
        raise AssertionError("TP Q0 candidate compiled cache key is absent")
    pairs = []
    for candidate in candidates:
        baseline = candidate[:-1]
        if baseline not in keys:
            raise AssertionError("TP Q0 baseline/candidate differ beyond Q0 selection")
        pairs.append(dict(baseline=repr(baseline), candidate=repr(candidate)))
    return pairs


def _case(torch, md, device, experts, workspace, case, sink):
    name, rows, kind = case
    generator = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn(rows, 4096, dtype=torch.bfloat16, device=device, generator=generator)*.5
    ids, weights = _routing(torch, rows, kind, generator, device, False)
    input_gs = experts.g1_alphas.float().clone()
    down_gs = experts._fc2_input_scale.float().clone()
    out = torch.empty_like(x)
    side = torch.cuda.Stream(device=device)
    graph = None
    sink.update(case=name, rows=rows, phase="initial", controls=[], candidate=[], q0=[], inputs=[])
    views = experts._sf6_weight_views
    scales = dict(fc1_input=input_gs, fc1_alpha=views.w1_alpha,
                  fc2_input=down_gs, fc2_alpha=views.w2_alpha)
    mapping = torch.arange(288, dtype=torch.int32, device=device)
    pointers = tuple(t.data_ptr() for t in (x, ids, weights, input_gs, down_gs, out))

    def call(candidate):
        got = md.launch_sm120_dynamic_moe(
            workspace=workspace, weights=views, a=x, topk_ids=ids, topk_weights=weights,
            input_gs=input_gs, down_input_scale=down_gs, scatter_output=out,
            num_experts=288, num_tokens=rows, k=4096, n=512, top_k=8,
            activation="swigluoai_uninterleave", swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
            activation_precision="fp4", quant_mode="nvfp4", _tp_sf6_q0_override=candidate)
        if got is not out or out.dtype != torch.bfloat16:
            raise AssertionError("TP Q0 launcher changed output identity/dtype")

    def eager(candidate):
        out.fill_(float("nan"))
        call(candidate)
        torch.cuda.synchronize(device)
        return out.clone()

    primary = None
    try:
        for changed in (False, True):
            phase = "changed" if changed else "initial"
            if changed:
                x.mul_(-.75)
                new_ids, new_weights = _routing(torch, rows, kind, generator, device, True)
                ids.copy_(new_ids); weights.copy_(new_weights)
                # Small private activation-scale vector exercises unequal cache
                # values. Actual model weights/alpha/SF6 storage stays untouched.
                input_gs[::2].mul_(.875); input_gs[1::2].mul_(1.125)
            if tuple(t.data_ptr() for t in (x, ids, weights, input_gs, down_gs, out)) != pointers:
                raise AssertionError("TP Q0 changed fixture replaced storage")
            sink["phase"] = phase
            identity = _inputs(x, ids, weights, scales)
            sink["inputs"].append(identity)
            b1 = eager(False)
            baseline_q0 = _q0(torch, workspace, ids, weights)
            b2, b3 = eager(False), eager(False)
            sink["controls"].append([check_control(b1,b2), check_control(b1,b3), check_control(b2,b3)])
            context = dict(result=sink, third=b3, inputs=x, route_ids=ids, route_weights=weights,
                           expert_map=mapping, scales=scales)
            for label in ("C1-eager", "C2-graph-current", "C3-graph-side"):
                sink["phase"] = phase+"-"+label
                if label == "C1-eager":
                    candidate = eager(True)
                else:
                    if graph is None:
                        graph = torch.cuda.CUDAGraph()
                        side.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.graph(graph, stream=side):
                            call(True)
                        torch.cuda.current_stream(device).wait_stream(side)
                    out.fill_(float("nan"))
                    if label == "C3-graph-side":
                        side.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.stream(side):
                            graph.replay()
                        torch.cuda.current_stream(device).wait_stream(side)
                    else:
                        graph.replay()
                    torch.cuda.synchronize(device)
                    candidate = out.clone()
                sink["candidate"].append(dict(phase=sink["phase"], **compare(candidate,b1,b2,failure_context=context)))
                q0 = _q0(torch, workspace, ids, weights)
                if q0 != baseline_q0:
                    raise AssertionError("TP Q0 route/sample bytes differ from same-input stock")
                sink["q0"].append(dict(phase=sink["phase"], sha256=_sha(json.dumps(q0,sort_keys=True).encode()),
                                       routes=q0["routes"], scope=q0["scope"]))
                zero_rows = (weights == 0).all(dim=1)
                if bool(zero_rows.any()) and (not bool((b1[zero_rows] == 0).all())
                        or not bool((candidate[zero_rows] == 0).all())):
                    raise AssertionError("TP Q0 all-zero-weight output is not exactly zero")
                del candidate
            if _inputs(x, ids, weights, scales) != identity:
                raise AssertionError("TP Q0 call mutated its fixture inputs")
            del b1, b2, b3
        sink.update(verdict="PASS", phase="complete", graph_replay=True)
    except BaseException as exc:
        primary = exc
        sink.update(verdict="FAIL", error=repr(exc))
    finally:
        try:
            torch.cuda.synchronize(device)
        except BaseException as exc:
            sink.update(verdict="FAIL", cleanup_error=repr(exc))
            if primary is None:
                primary = exc
        graph = None
    if primary is not None:
        raise primary


def ensure_tp_sf6_q0_selftest(experts, *, layer):
    """After final owner release, before wrappers/readiness; one real layer/rank."""
    with _LOCK:
        torch, md, device, provenance = _runtime(experts, layer)
        state = _caller_state(experts, layer)
        key = (os.getpid(), str(device), id(experts._sf6_weight_views),
               _sha(json.dumps(provenance,sort_keys=True).encode()), SEED, CASES)
        previous = _STATES.get(key)
        if previous is not None:
            if previous["verdict"] != "PASS":
                raise RuntimeError("TP Q0 startup canary previously failed or is reentrant")
            if previous["caller_state"] != state:
                raise RuntimeError("TP Q0 cached canary owner changed")
            return previous
        receipt = dict(schema=1, verdict="RUNNING", phase="admission", started_at=time.time(),
                       actual_packed_owner=True, geometry=dict(E=288,K=4096,I=512,top8=8),
                       source=provenance, caller_state=state, cases=[],
                       performance_acceptance=False, full_sanitizer_acceptance=False,
                       scope="one real TP packed-only layer per rank; four cases through8192; not all layers/sanitizer")
        _STATES[key] = receipt
        cache_before = _cache_snapshot(md)
        before_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        receipt["memory_before"] = _memory(torch,device)
        primary = None
        workspace = None
        try:
            receipt["weights_before"] = {name:_tensor_identity(value) for name,value in _backings(experts,layer).items()}
            workspace = md.allocate_sm120_dynamic_workspace(
                state_E=288,weight_E=288,routed_rows=8192*8,k=4096,n=512,num_topk=8,
                device=device,activation="swigluoai_uninterleave",quant_mode="nvfp4",tile_m=128)
            for case in CASES:
                receipt["phase"] = case[0]
                cell = dict(verdict="RUNNING",started_at=time.time())
                receipt["cases"].append(cell)
                _case(torch,md,device,experts,workspace,case,cell)
                cell["completed_at"] = time.time()
            receipt["cache_pairs"] = _selected_keys(md)
        except BaseException as exc:
            primary = exc
            receipt.update(verdict="FAIL",error=repr(exc))
        finally:
            try:
                torch.cuda.synchronize(device)
                workspace = None
                receipt["scratch_cache_removed"] = _restore_scratch_caches(md,cache_before)
                if "weights_before" in receipt:
                    receipt["weights_after"] = {name:_tensor_identity(value) for name,value in _backings(experts,layer).items()}
                    if receipt["weights_before"] != receipt["weights_after"]:
                        raise AssertionError("TP Q0 canary changed actual weight/scale backing bytes")
                if _caller_state(experts,layer) != state:
                    raise AssertionError("TP Q0 canary changed caller owner/descriptor storage")
                if (not torch.equal(torch.get_rng_state(),before_rng[0])
                        or not torch.equal(torch.cuda.get_rng_state(device),before_rng[1])):
                    raise AssertionError("TP Q0 canary changed global RNG state")
                receipt["caller_preserved"] = True
            except BaseException as exc:
                receipt.update(verdict="FAIL",cleanup_error=repr(exc))
                if primary is None:
                    primary = exc
        receipt.update(completed_at=time.time(),memory_after=_memory(torch,device))
        if primary is not None:
            print("[tp-sf6-q0-selftest] FAIL "+json.dumps(receipt,sort_keys=True),flush=True)
            raise RuntimeError("TP SF6 Q0 startup canary failed; readiness refused") from primary
        receipt.update(verdict="PASS",phase="complete")
        print("[tp-sf6-q0-selftest] PASS "+json.dumps(receipt,sort_keys=True),flush=True)
        return receipt
