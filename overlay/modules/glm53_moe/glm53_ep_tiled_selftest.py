"""One actual EP layer: stock row-major references before in-place tiling.

No model requests, timings-as-performance, weight replicas, or relaxed limits.
The two hooks straddle the owner's irreversible startup weight relayout.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
from threading import RLock
import time
from types import SimpleNamespace

from .glm53_ep_local_selftest import (
    _cache_snapshot, _restore_scratch_caches,
    _tensor_identity, _inputs, _memory, check_control, compare, row_errors,
    ROW_L2_FLOOR, ROW_PEAK_FLOOR,
)

SEED = 905329
CASES = (("mixed6", 6, "mixed"), ("balanced12", 12, "balanced"),
         ("concentrated24", 24, "concentrated"), ("zeros32", 32, "zeros"),
         ("remote33", 33, "remote"), ("balanced2128", 2128, "balanced"),
         ("balanced4096", 4096, "balanced"),
         ("concentrated6912", 6912, "concentrated"),
         ("balanced8192", 8192, "balanced"),
         # Append so the original cases keep their SEED + case_index inputs.
         # SPEC_K=3 verifies 4 rows/request: batches 1..4 use M4/8/12/16.
         ("mixed4", 4, "mixed"), ("balanced8", 8, "balanced"),
         ("concentrated16", 16, "concentrated"))
_LOCK = RLock()
_STATES = {}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def route_rows(rows, kind, offset, changed=False):
    """Global top8 IDs; all-remote remains remote in both phases."""
    if kind not in {case[2] for case in CASES} or offset not in (0, 72, 144, 216):
        raise ValueError("unsupported EP tiled fixture")
    shift = 3 if changed else 0
    result = []
    for token in range(rows):
        if kind == "remote":
            row = [(offset + 72 + (token + slot + shift) % 216) % 288 for slot in range(8)]
        elif kind == "concentrated":
            row = [offset + (slot + shift) % 72 for slot in range(8)]
        elif kind == "mixed" and token % 3 == 0:
            row = [offset+shift, (offset+72) % 288, offset+1, -1,
                   offset+71, 288, offset+2, offset+2]
        else:
            row = [(token*37 + slot*31 + (137 if changed else 0)) % 288 for slot in range(8)]
        result.append(row)
    return result


def _backings(owner, layer):
    return dict(w13=layer.w13_weight, w2=layer.w2_weight,
                sf1=owner.w1_sf_mma, sf2=owner.w2_sf_mma,
                fc1_alpha=owner.g1_alphas, fc2_alpha=owner.g2_alphas,
                fc2_input=owner._fc2_input_scale)


def _identities(owner, layer):
    return {name: _tensor_identity(value) for name, value in _backings(owner, layer).items()}


def _runtime(owner, layer):
    if (not owner._use_ep or not owner._ep_no_dummy or
            (owner.global_num_experts, owner.num_local_experts, owner.hidden_dim,
             owner.intermediate_size_per_partition, owner.topk) != (288, 72, 4096, 2048, 8)):
        raise RuntimeError("EP tiled self-test needs the exact unpadded actual owner")
    # This hook deliberately precedes allocation of the old micro workspaces.
    # Its admission belongs to the tiled owner, independently of the old knob.
    import torch
    from . import moe_dispatch as md
    from . import moe_dynamic_ep_local as local
    from . import glm53_ep_local_selftest as numerical
    device = layer.w13_weight.device
    if (device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1)
            or torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("EP tiled self-test requires SM121 outside graph capture")
    if not md._GLM53_EP_TILED or not local.stock_contract_matches():
        raise RuntimeError("EP tiled self-test selection/source contract differs")
    if (owner._kernel_num_experts != 72 or
            (owner._activation_str, owner._swiglu_alpha, owner._swiglu_beta, owner._swiglu_limit)
            != ("swigluoai_uninterleave", 1., 0., 10.)):
        raise RuntimeError("EP tiled self-test requires the exact serving activation")
    prefix = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x."
    tiled = importlib.import_module(prefix + "glm53_ep_tiled")
    decode = importlib.import_module(prefix + "moe_static_ep_tiled")
    decode.ep_tiled_source_contract()
    files = dict(tiled_selftest=Path(__file__), numerical=Path(numerical.__file__),
        wrapper=Path(inspect.getfile(type(owner))), dispatch=Path(md.__file__),
        local=Path(local.__file__), stock=Path(local._stock.__file__),
        tiled_owner=Path(tiled.__file__), tiled_decode=Path(decode.__file__))
    for name in ("moe_static_kernel_v4", "moe_static_kernel_v5",
                 "moe_reform_sf_pack", "moe_sf_pack", "moe_dynamic_gated_sf6"):
        files[name] = Path(importlib.import_module(prefix+name).__file__)
    versions = {}
    for name in ("torch", "cuda.bindings", "flashinfer"):
        module = importlib.import_module(name)
        filename = getattr(module, "__file__", None)
        entry = dict(version=str(getattr(module, "__version__", "unknown")),
                     path=filename, sha256=_sha(Path(filename).read_bytes()) if filename else None)
        if name == "cuda.bindings":
            distribution = importlib.metadata.distribution("cuda-bindings")
            paths = [Path(distribution.locate_file(p)) for p in distribution.files or ()
                     if p.name == "METADATA" and p.parent.name.startswith("cuda_bindings-")]
            if len(paths) != 1:
                raise RuntimeError("EP tiled cannot identify CUDA bindings metadata")
            entry.update(version=distribution.version, metadata_path=str(paths[0]),
                         metadata_sha256=_sha(paths[0].read_bytes()))
        elif filename is None:
            raise RuntimeError("EP tiled cannot identify imported runtime: " + name)
        versions[name] = entry
    provenance = dict(source={name:dict(path=str(path),sha256=_sha(path.read_bytes()))
                             for name,path in files.items()}, versions=versions)
    return dict(torch=torch, md=md, device=device, tiled=tiled, decode=decode, provenance=provenance)


def _key(context):
    return (os.getpid(), str(context["device"]),
            _sha(json.dumps(context["provenance"], sort_keys=True).encode()), SEED)


def _publish_failure(receipt, exc):
    receipt.update(verdict="FAIL", error=repr(exc), completed_at=time.time())
    print("[ep-tiled-selftest] FAIL " + json.dumps(receipt, sort_keys=True), flush=True)


def _same_weights_and_scales(before, after, *, relayout):
    if before.keys() != after.keys():
        raise AssertionError("actual owner backing set changed")
    for name in before:
        a, b = before[name], after[name]
        if name in ("w13", "w2") and relayout:
            if {k:v for k,v in a.items() if k != "sha256"} != {k:v for k,v in b.items() if k != "sha256"}:
                raise AssertionError("in-place tiling replaced actual weight storage: " + name)
        elif a != b:
            raise AssertionError("self-test changed immutable caller backing: " + name)


def _reference_owner(owner):
    # Descriptor-backed alpha properties must never be overwritten on a copy
    # of the real owner. This object only borrows immutable tensors/methods.
    cls = type("_EPTiledStockReference", (SimpleNamespace,),
               {"_apply_ep_compact": type(owner)._apply_ep_compact})
    return cls(num_local_experts=72, _kernel_num_experts=72, max_num_tokens=8192,
               w1_sf_mma=owner.w1_sf_mma, w2_sf_mma=owner.w2_sf_mma,
               g1_alphas=owner.g1_alphas, g2_alphas=owner.g2_alphas,
               _fc2_input_scale=owner._fc2_input_scale,
               _activation_str=owner._activation_str, _swiglu_alpha=owner._swiglu_alpha,
               _swiglu_beta=owner._swiglu_beta, _swiglu_limit=owner._swiglu_limit)


def _fill(context, buffers, case_index, changed, offset):
    torch, device = context["torch"], context["device"]
    _, rows, kind = CASES[case_index]
    x, ids, weights, output = (value[:rows] for value in buffers)
    generator = torch.Generator(device=device).manual_seed(SEED + case_index)
    x.copy_(torch.randn(x.shape, dtype=torch.bfloat16, device=device, generator=generator))
    x.mul_(-.375 if changed else .5)
    ids.copy_(torch.tensor(route_rows(rows, kind, offset, changed), dtype=torch.int32, device=device))
    weights.copy_(torch.rand(weights.shape, dtype=torch.float32, device=device, generator=generator))
    weights.div_(weights.sum(dim=1, keepdim=True))
    if kind in ("zeros", "mixed"):
        weights[:, ::2] = 0
        weights[0, 0] = -0.0
    if changed:
        weights.mul_(.875)
    return x, ids, weights, output


def _scales(owner):
    return dict(fc1_alpha=owner.g1_alphas, fc2_alpha=owner.g2_alphas,
                fc2_input=owner._fc2_input_scale)


def _check_rng(context, original):
    torch, device = context["torch"], context["device"]
    if not torch.equal(original[0], torch.get_rng_state()) or not torch.equal(original[1], torch.cuda.get_rng_state(device)):
        raise AssertionError("EP tiled self-test changed global RNG state")


def _capture_references(context, owner, layer, receipt):
    torch, md, device = context["torch"], context["md"], context["device"]
    if tuple(layer.w13_weight.shape) != (72,4096,2048) or tuple(layer.w2_weight.shape) != (72,4096,1024):
        raise RuntimeError("EP reference expects original row-major model weights")
    if getattr(layer.w13_weight, "_b12x_tile_major", False):
        raise RuntimeError("row-major reference requested after tiling")
    original = _identities(owner, layer)
    receipt["weights_before"] = original
    rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    cache = _cache_snapshot(md)
    refs, buffers = [], None
    try:
        buffers = (torch.empty((8192,4096),dtype=torch.bfloat16,device=device),
                   torch.empty((8192,8),dtype=torch.int32,device=device),
                   torch.empty((8192,8),dtype=torch.float32,device=device),
                   torch.empty((8192,4096),dtype=torch.bfloat16,device=device))
        stock = _reference_owner(owner)
        offset = int(owner.local_expert_offset)
        for index, (name, rows, kind) in enumerate(CASES):
            cell = dict(case=name, rows=rows, kind=kind, verdict="RUNNING",
                        phase="stock-reference", started_at=time.time(), controls=[], inputs=[],
                        candidate=[], reference_outputs=[], graph_replay=rows <= 33 or rows == 8192)
            receipt["cases"].append(cell)
            phases = []
            for changed in (False, True):
                x, ids, weights, out = _fill(context, buffers, index, changed, offset)
                identity = _inputs(x, ids, weights, _scales(owner))
                cell["inputs"].append(identity)
                local = (ids >= offset) & (ids < offset+72)
                mapped = torch.where(local, ids-offset, torch.full_like(ids,72))
                values = torch.where(local, weights, torch.zeros_like(weights))
                outputs = []
                for label in ("B1", "B2", "B3"):
                    out.fill_(float("nan"))
                    got = stock._apply_ep_compact(out, x, layer.w13_weight, layer.w2_weight, mapped, values)
                    if got is not out:
                        raise AssertionError("stock reference did not write its output")
                    torch.cuda.synchronize(device)
                    outputs.append(out.clone())
                cell["reference_outputs"].append({label:_tensor_identity(value)
                    for label,value in zip(("B1","B2","B3"),outputs)})
                cell["controls"].append([check_control(outputs[0],outputs[1]),
                    check_control(outputs[0],outputs[2]),check_control(outputs[1],outputs[2])])
                if kind == "remote" and any(not bool((value == 0).all()) for value in outputs):
                    raise AssertionError("all-remote stock output is not exact zero")
                phases.append((outputs[0].cpu(), outputs[1].cpu()))
                if _inputs(x,ids,weights,_scales(owner)) != identity:
                    raise AssertionError("stock reference changed actual inputs/scales")
                del outputs
            refs.append(phases)
            cell["phase"] = "references-complete"
        _same_weights_and_scales(original, _identities(owner,layer), relayout=False)
        _check_rng(context,rng)
        receipt["reference_bytes"] = sum(t.numel()*t.element_size() for phases in refs for pair in phases for t in pair)
        return dict(buffers=buffers, references=refs, rng=rng, offset=offset, original=original)
    finally:
        torch.cuda.synchronize(device)
        receipt["reference_scratch_restored"] = _restore_scratch_caches(md,cache)


def before_relayout(owner, layer):
    """Before weight mutation; return CPU references and one input buffer set."""
    with _LOCK:
        context = _runtime(owner,layer)
        key = _key(context)
        previous = _STATES.get(key)
        if previous is not None:
            if previous["verdict"] != "PASS":
                raise RuntimeError("EP tiled canary previously failed or remains in progress")
            return dict(cached=True, receipt=previous)
        receipt = dict(schema=1, verdict="RUNNING", phase="before-relayout", started_at=time.time(),
                       source=context["provenance"], geometry=dict(E=72,K=4096,I=2048,top8=8),
                       cases=[], actual_weight_owner=True, performance_acceptance=False,
                       full_sanitizer_acceptance=False,
                       scope="first actual EP layer per process/device/source; stock compact references before in-place tiling")
        _STATES[key] = receipt
        try:
            receipt["memory_before"] = _memory(context["torch"],context["device"])
            handle = _capture_references(context,owner,layer,receipt)
            handle.update(context=context, receipt=receipt, owner_id=id(owner), layer_id=id(layer), key=key)
            receipt["phase"] = "awaiting-relayout"
            return handle
        except BaseException as exc:
            _publish_failure(receipt,exc)
            raise RuntimeError("EP tiled stock reference failed; readiness refused") from exc


def _failure_rows(candidate, baseline, repeat, inputs):
    """Bounded raw diagnostics; the established comparison remains authoritative."""
    import torch
    error, peak = row_errors(candidate,baseline)
    noise, peak_noise = row_errors(repeat,baseline)
    ll = torch.maximum(3*noise,torch.full_like(noise,ROW_L2_FLOOR))
    pl = torch.maximum(3*peak_noise,torch.full_like(peak_noise,ROW_PEAK_FLOOR))
    indices = ((error > ll) | (peak > pl)).nonzero().flatten()[:8].cpu().tolist()
    result = []
    for row in indices:
        columns = (candidate[row].float()-baseline[row].float()).abs().topk(8).indices.cpu().tolist()
        result.append(dict(row=row,l2=float(error[row]),peak=float(peak[row]),
            l2_limit=float(ll[row]),peak_limit=float(pl[row]),columns=columns,
            raw_bf16={name:value[row].view(torch.int16)[columns].cpu().tolist()
                      for name,value in (("B1",baseline),("B2",repeat),("C",candidate),("X",inputs))}))
    return dict(rows=result, max_rows=8, max_columns=8, B3_scope="hash and original controls only; output not retained")


def _packed_identity(owner, layer):
    """Bind the admitted SF6 owner while original loader aliases still exist.

    enabled is published by prepare_reform_scales only after both complete
    device roundtrips. This records that source-bound preparation contract;
    it does not run another pack/unpack or claim raw sources are released.
    """
    views = owner._ep_tiled_weight_views
    scales = getattr(views, "reform_scales", None)
    if (not getattr(views, "tiled", False) or not getattr(views, "packed_only", False)
            or not getattr(scales, "enabled", False)):
        raise AssertionError("EP tiled canary requires both admitted packed-only SF6 planes")
    for name in ("w1_scale_storage", "w2_scale_storage", "_w13_sf_storage",
                 "_down_sf_storage", "sfb_w13_ptr", "sfb_down_ptr"):
        if getattr(views, name, None) is not None:
            raise AssertionError("EP tiled SF6 view retains raw scale field: " + name)
    if views.sfb1_packed is not scales.fc1 or views.sfb2_packed is not scales.fc2:
        raise AssertionError("EP tiled view does not own its admitted SF6 planes")
    raw = (owner.w1_scale, owner.w2_scale, owner.w1_sf_mma, owner.w2_sf_mma,
           layer.w13_weight_scale, layer.w2_weight_scale)
    if any(value is None for value in raw):
        raise AssertionError("EP tiled canary must precede final model raw-scale release")
    raw_addresses = {value.untyped_storage().data_ptr() for value in raw}
    planes = {}
    for name, value, blocks in (("fc1", scales.fc1, 512), ("fc2", scales.fc2, 256)):
        if (tuple(value.shape) != (72,blocks,1552) or str(value.dtype) != "torch.uint8"
                or value.device != layer.w13_weight.device or not value.is_contiguous()
                or value.untyped_storage().data_ptr() in raw_addresses):
            raise AssertionError("EP tiled SF6 plane geometry/device/source alias mismatch: " + name)
        planes[name] = _tensor_identity(value)
    return dict(owner_id=id(views), scales_id=id(scales), planes=planes,
        raw_sources_retained=True, raw_release_acceptance=False,
        preparation_contract=dict(
            scope="source-bound mandatory full device roundtrip in prepare_reform_scales; not an additional canary roundtrip",
            both_planes_enabled=True, stage_raw_bytes=2048, stage_packed_bytes=1552,
            fc1_stages=72*512, fc2_stages=72*256,
            raw_bytes=72*768*2048, packed_bytes=72*768*1552))


def _cache_evidence(context, owner, rows):
    """Require the real launcher's isolated direct SF6 artifact namespace."""
    if rows <= 32:
        ws = owner._ep_tiled_workspace
        geometry = context["decode"].ep_tiled_geometry(
            rows, ws.static.max_rows, ws.scratch.max_active_clusters)
        expected = ("glm53_ep_static_tiled_fp32_v1", rows, ws.static.max_rows,
            ws.scratch.max_active_clusters, "torch.int32", False, True,
            geometry["fc1"], geometry["fc2"], "nvfp4", "sf6_v1",
            "swigluoai_uninterleave", 1., 0., 10.,
            "bf16_scatter" if geometry["reform"] else "fp32_scatter")
        if geometry["reform"]:
            expected += ("glm53_ep_static_sf6_a_ring_v1",
                         "glm53_ep_static_sf6_word_unpack_v1",
                         "glm53_ep_static_bf16_scatter_v1")
        if expected not in context["decode"]._EP_TILED_KERNEL_CACHE:
            raise AssertionError("native EP tiled decode artifact was not selected/warmed")
        return dict(scope="exact native shape cache key", keys=[repr(expected)])
    keys = [key for key in context["md"]._DYNAMIC_KERNEL_CACHE
            if len(key) == 20 and key[:7] == ("dynamic","fp4","nvfp4",72,4096,2048,8)
            and str(key[9]) == "torch.int32" and key[10:16] ==
                (False,True,"swigluoai_uninterleave",1.,0.,10.)
            and key[17] is True and key[-2:] ==
                ("glm53_ep_prefill_local_fp32_v2","glm53_ep_tiled_sf6_v1")]
    if not keys:
        raise AssertionError("tiled EP prefill artifact namespace is absent")
    return dict(scope="matching runtime-shaped dynamic keys; source binds per-shape selection",
                keys=sorted(map(repr,keys)))


def _validate_candidate(context, owner, layer, handle):
    torch, device, tiled = context["torch"],context["device"],context["tiled"]
    receipt = handle["receipt"]
    if owner._ep_tiled_weight_views is None or owner._ep_tiled_workspace is None:
        raise RuntimeError("EP tiled actual owner/workspace was not prepared")
    after = _identities(owner,layer)
    _same_weights_and_scales(handle["original"],after,relayout=True)
    receipt["weights_after_relayout"] = after
    receipt["packed_before"] = _packed_identity(owner,layer)
    mapping = torch.full((288,),-1,dtype=torch.int32,device=device)
    mapping[handle["offset"]:handle["offset"]+72] = torch.arange(72,dtype=torch.int32,device=device)
    for index, cell in enumerate(receipt["cases"]):
        graph, side = None, torch.cuda.Stream(device=device)
        try:
            for changed in (False,True):
                x,ids,weights,out = _fill(context,handle["buffers"],index,changed,handle["offset"])
                identity = _inputs(x,ids,weights,_scales(owner))
                if identity != cell["inputs"][int(changed)]:
                    raise AssertionError("tiled replay does not use the original input bytes/addresses")
                b1,b2 = (value.to(device) for value in handle["references"][index][int(changed)])
                def call():
                    result = tiled.launch_ep_tiled(owner,out,x,layer.w13_weight,layer.w2_weight,ids,weights,mapping)
                    if result is not out:
                        raise AssertionError("EP tiled candidate declined or replaced output")
                for label in ("C1-eager","C2-graph-current" if cell["graph_replay"] else "C2-eager",
                              "C3-graph-side" if cell["graph_replay"] else "C3-side"):
                    cell["phase"] = ("changed" if changed else "initial")+"-"+label
                    out.fill_(float("nan"))
                    if label == "C1-eager" or label == "C2-eager":
                        call()
                    elif label.startswith("C2-graph"):
                        if graph is None:
                            graph = torch.cuda.CUDAGraph()
                            side.wait_stream(torch.cuda.current_stream(device))
                            with torch.cuda.graph(graph,stream=side):
                                call()
                            torch.cuda.current_stream(device).wait_stream(side)
                        graph.replay()
                    else:
                        side.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.stream(side):
                            graph.replay() if graph is not None else call()
                        torch.cuda.current_stream(device).wait_stream(side)
                    torch.cuda.synchronize(device)
                    try:
                        result = compare(out,b1,b2)
                    except BaseException:
                        try:
                            cell["first_failure_rows"] = _failure_rows(out,b1,b2,x)
                        except BaseException as exc:
                            cell["diagnostic_error"] = repr(exc)
                        raise
                    cell["candidate"].append(dict(phase=cell["phase"],**result))
                    if label == "C1-eager":
                        cell["cache_evidence"] = _cache_evidence(context,owner,cell["rows"])
                    if cell["kind"] == "remote" and not bool((out == 0).all()):
                        raise AssertionError("all-remote tiled output is not exact zero")
                if _inputs(x,ids,weights,_scales(owner)) != identity:
                    raise AssertionError("EP tiled candidate changed actual inputs/scales")
            cell.update(verdict="PASS",phase="complete",completed_at=time.time())
        finally:
            torch.cuda.synchronize(device)
            graph = None
    _same_weights_and_scales(after,_identities(owner,layer),relayout=False)
    receipt["packed_after"] = _packed_identity(owner,layer)
    if receipt["packed_before"] != receipt["packed_after"]:
        raise AssertionError("EP tiled canary changed its actual packed SF6 owner/bytes")
    receipt["actual_packed_owner"] = True
    receipt["raw_release_acceptance"] = False
    _check_rng(context,handle["rng"])
    receipt["caller_preserved"] = True


def after_relayout(owner, layer, reference):
    """After new owner/workspaces exist; PASS is required before readiness."""
    with _LOCK:
        if reference.get("cached"):
            if reference["receipt"]["verdict"] != "PASS":
                raise RuntimeError("cached EP tiled canary is not PASS")
            return reference["receipt"]
        context, receipt = reference["context"],reference["receipt"]
        if receipt["verdict"] != "RUNNING" or reference["owner_id"] != id(owner) or reference["layer_id"] != id(layer):
            raise RuntimeError("EP tiled canary handle does not match pending owner")
        cache = _cache_snapshot(context["md"])
        primary = None
        try:
            if _key(_runtime(owner,layer)) != reference["key"]:
                raise RuntimeError("EP tiled runtime/source changed across relayout")
            _validate_candidate(context,owner,layer,reference)
        except BaseException as exc:
            primary = exc
        finally:
            try:
                context["torch"].cuda.synchronize(context["device"])
                reference.pop("references",None)
                reference.pop("buffers",None)
                receipt["candidate_scratch_restored"] = _restore_scratch_caches(context["md"],cache)
                receipt["memory_after"] = _memory(context["torch"],context["device"])
            except BaseException as exc:
                receipt["cleanup_error"] = repr(exc)
                if primary is None:
                    primary = exc
        if primary is not None:
            _publish_failure(receipt,primary)
            raise RuntimeError("EP tiled numerical canary failed; readiness refused") from primary
        receipt.update(verdict="PASS",phase="complete",completed_at=time.time())
        print("[ep-tiled-selftest] PASS "+json.dumps(receipt,sort_keys=True),flush=True)
        return receipt
