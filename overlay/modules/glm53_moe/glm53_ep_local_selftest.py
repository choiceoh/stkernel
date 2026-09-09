"""Required, bounded EP startup numerics; no requests or component timing.

CPU import is inert. Runtime failures remain sticky for this process/source/device.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import struct
from threading import RLock
import time
from types import SimpleNamespace

from itertools import islice
import math
import struct

MAX_BAD_ROWS = 8
MAX_COLUMNS = 8
OUTPUTS = ("B1", "B2", "B3", "C1")


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16(word):
    return struct.unpack("<f", struct.pack("<I", (int(word) & 0xffff) << 16))[0]


def build_failure_record(rows, *, total_bad_rows):
    """Format bounded CPU records, selecting worst FP32 differences from BF16.

    Metrics and limits are the values from the actual failed comparison;
    they are never inferred from the global maxima or recomputed in Python.
    Full captured rows are discarded after selecting at most eight columns.
    """
    captured = []
    for row in islice(rows, MAX_BAD_ROWS):
        raw = row["raw_bf16"]
        fields = OUTPUTS + (("X",) if "X" in raw else ())
        widths = {len(raw[name]) for name in fields}
        if len(widths) != 1 or not next(iter(widths)):
            raise ValueError("diagnostic output rows must have matching nonempty widths")
        metrics = row["metrics"]
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("diagnostic metrics must be finite")
        violations = dict(l2=metrics["relative_l2"] > metrics["l2_limit"],
                          peak=metrics["relative_peak"] > metrics["peak_limit"])
        if not any(violations.values()):
            raise ValueError("diagnostic row did not fail its original limits")
        if len(row["routes"]) != 8:
            raise ValueError("diagnostics require the original top8 routes")
        # Match the original comparison's FP32 subtraction. Stable column
        # ordering only breaks equal-magnitude ties for this diagnostic view.
        delta = [abs(_f32(_bf16(c) - _bf16(b)))
                 for b, c in zip(raw["B1"], raw["C1"])]
        columns = sorted(range(len(delta)), key=lambda col: (-delta[col], col))[:MAX_COLUMNS]
        captured.append(dict(
            row_id=int(row["row_id"]), metrics=dict(metrics), violations=violations,
            routes=row["routes"],
            worst_columns=[dict(column=col, absolute_delta_f32=delta[col],
                                raw_bf16_u16={name: int(raw[name][col]) & 0xffff
                                              for name in fields}) for col in columns]))
    if (not captured or type(total_bad_rows) is not int
            or total_bad_rows < len(captured)):
        raise ValueError("diagnostic bad-row count is inconsistent")
    return dict(schema=1, diagnostic_only=True, first_failure_preserved=True,
                total_bad_rows=total_bad_rows, captured_bad_rows=len(captured),
                truncated=total_bad_rows > len(captured),
                selection="first bad rows in ascending row order; worst absolute-delta columns",
                input_column_scope="X is the same input-column index, not a causal attribution",
                raw_dtype="bfloat16", rows=captured)


def capture_failure(candidate, baseline, repeat, third, *, bad, error, peak,
                    noise, peak_noise, l2_limits, peak_limits, total_bad_rows,
                    route_ids, route_weights, expert_map, scales, inputs):
    """Copy only after FAIL: <=8 full output rows, route metadata and scales.

    At H4096 the four outputs plus input snapshot total at most 320 KiB. Only the
    worst eight columns are retained in JSON. No model/kernel call is made.
    """
    import torch
    if any(t.dtype != torch.bfloat16 for t in (baseline, repeat, third, candidate, inputs)):
        raise ValueError("raw failure capture requires BF16 outputs")
    if route_ids.shape[1] != 8 or route_weights.shape[1] != 8:
        raise ValueError("raw failure capture requires top8 routing")
    indices = bad.nonzero(as_tuple=False).flatten()[:MAX_BAD_ROWS].detach().cpu().tolist()
    mapping = expert_map.detach().cpu().tolist()
    scale_values = {name: values.detach().cpu().tolist() for name, values in scales.items()}

    def records():
        for row_id in indices:
            b1 = baseline[row_id].float()
            metrics = dict(relative_l2=float(error[row_id]), relative_peak=float(peak[row_id]),
                           stock_relative_l2=float(noise[row_id]),
                           stock_relative_peak=float(peak_noise[row_id]),
                           l2_limit=float(l2_limits[row_id]), peak_limit=float(peak_limits[row_id]),
                           b1_l2_norm=float(b1.norm()), b1_absmax=float(b1.abs().amax()))
            ids = route_ids[row_id].detach().cpu().tolist()
            weights = route_weights[row_id].detach().cpu().tolist()
            routes = []
            for slot, (expert, weight) in enumerate(zip(ids, weights)):
                mapped = mapping[expert] if 0 <= expert < len(mapping) else -1
                values = {name: items[mapped] if 0 <= mapped < len(items) else None
                          for name, items in scale_values.items()}
                routes.append(dict(slot=slot, global_expert_id=int(expert),
                                   local_expert_id=int(mapped), weight=weight, scales=values))
            raw = {name: tensor[row_id].detach().cpu().view(torch.int16).tolist()
                   for name, tensor in zip(OUTPUTS + ("X",),
                                           (baseline, repeat, third, candidate, inputs))}
            yield dict(row_id=row_id, metrics=metrics, routes=routes, raw_bf16=raw)

    return build_failure_record(records(), total_bad_rows=total_bad_rows)


ROW_L2_FLOOR = .02
ROW_PEAK_FLOOR = .04

def row_errors(actual, reference):
    import torch
    assert actual.shape == reference.shape and actual.ndim == 2, "output shape mismatch"
    a, b = actual.float(), reference.float()
    assert all(bool(torch.isfinite(t).all()) for t in (a, b)), "nonfinite output"
    delta = a - b
    return (delta.norm(dim=1) / b.norm(dim=1).clamp_min(1e-6),
            delta.abs().amax(dim=1) / b.abs().amax(dim=1).clamp_min(1e-6))

def check_control(baseline, repeat):
    """Stock variability itself must stay within the fixed numerical floors."""
    l2, peak = row_errors(repeat, baseline)
    bad = (l2 > ROW_L2_FLOOR) | (peak > ROW_PEAK_FLOOR)
    result = dict(bad_rows=int(bad.sum()), max_row_relative_l2=float(l2.max()),
                  max_row_relative_abs=float(peak.max()))
    assert not result["bad_rows"], dict(verdict="UNSTABLE_STOCK_CONTROL", **result)
    return result

def compare(candidate, baseline, repeat, *, failure_context=None):
    import torch
    # A noisy reference must never authorize arbitrarily noisy candidates.
    check_control(baseline, repeat)
    error, max_error = row_errors(candidate, baseline)
    noise, max_noise = row_errors(repeat, baseline)
    bad = ((error > torch.maximum(3*noise, torch.full_like(noise, ROW_L2_FLOOR)))
           | (max_error > torch.maximum(3*max_noise, torch.full_like(max_noise, ROW_PEAK_FLOOR))))
    result = dict(bad_rows=int(bad.sum()), max_row_relative_l2=float(error.max()),
                  max_row_relative_abs=float(max_error.max()),
                  stock_max_row_relative_l2=float(noise.max()),
                  stock_max_row_relative_abs=float(max_noise.max()))
    if result["bad_rows"] and failure_context is not None:
        # The original failure is permanent even if diagnostic copying fails.
        # No host tensor copies or additional comparisons occur on success.
        sink = failure_context["result"]
        sink["verdict"] = "FAIL"
        if "candidate_first_failure" not in sink:
            sink["candidate_first_failure"] = dict(
                verdict="CANDIDATE_NUMERICS_FAIL", phase=sink.get("phase"), **result)
            try:
                sink["candidate_failure_diagnostics"] = capture_failure(
                    candidate, baseline, repeat, failure_context["third"],
                    bad=bad, error=error, peak=max_error, noise=noise, peak_noise=max_noise,
                    l2_limits=torch.maximum(3*noise, torch.full_like(noise, ROW_L2_FLOOR)),
                    peak_limits=torch.maximum(3*max_noise, torch.full_like(max_noise, ROW_PEAK_FLOOR)),
                    total_bad_rows=result["bad_rows"],
                    route_ids=failure_context["route_ids"], route_weights=failure_context["route_weights"],
                    expert_map=failure_context["expert_map"], scales=failure_context["scales"],
                    inputs=failure_context["inputs"])
            except BaseException as exc:
                sink["candidate_failure_diagnostics_error"] = repr(exc)
    assert not result["bad_rows"], dict(verdict="CANDIDATE_NUMERICS_FAIL", **result)
    return result


SEED = 905308
CASES = (("concentrated6912", 6912, "concentrated"),
         ("balanced4096", 4096, "balanced"),
         ("remote4096", 4096, "remote"),
         ("duplicate4096", 4096, "duplicate"),
         ("zeros4097", 4097, "zeros"),
         ("balanced8192", 8192, "balanced"))
_LOCK = RLock()
_STATES = {}
_LOG = logging.getLogger(__name__)
_CACHE_NAMES = ("_WORKSPACE_CACHE", "_WEIGHT_CACHE", "_W4A16_WEIGHT_CACHE",
                "_PADDED_WEIGHT_CACHE", "_SF_PACKED", "_SF_PACK_DUMMY")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _tensor_bytes(tensor):
    import torch
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _tensor_identity(tensor):
    import torch
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    # Hash large synthetic weights without a second full CPU-sized copy.
    for offset in range(0, raw.numel(), 8*1024*1024):
        digest.update(raw[offset:offset+8*1024*1024].cpu().numpy().tobytes())
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=digest.hexdigest(), data_ptr=tensor.data_ptr())


def _inputs(x, ids, weights, scales):
    return {name: _tensor_identity(value) for name, value in
            dict(X=x, ids=ids, weights=weights, **scales).items()}


def _runtime(wrapper, device):
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dynamic_ep_local as local
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import glm53_ep_route_remap as remap
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_micro_kernel as micro
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1):
        raise RuntimeError("EP self-test requires the serving SM121 device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EP self-test cannot run under graph capture")
    if not md._GLM53_EP_PREFILL_LOCAL or not local.stock_contract_matches():
        raise RuntimeError("EP self-test source/dispatcher admission differs")
    if (wrapper._kernel_num_experts != 72 or wrapper.hidden_dim != 4096
            or wrapper.intermediate_size_per_partition != 2048
            or wrapper._activation_str != "swigluoai_uninterleave"
            or (wrapper._swiglu_alpha, wrapper._swiglu_beta, wrapper._swiglu_limit) != (1., 0., 10.)):
        raise RuntimeError("EP self-test requires exact serving geometry")
    if wrapper._ep_fixed_workspace is None or wrapper._ep_zero_weight_workspace is None:
        raise RuntimeError("EP self-test needs both pinned decode workspaces")
    files = {"selftest": Path(__file__), "wrapper": Path(inspect.getfile(type(wrapper))),
             "dispatch": Path(md.__file__), "local": Path(local.__file__),
             "stock": Path(local._stock.__file__), "remap": Path(remap.__file__),
             "micro": Path(micro.__file__)}
    source = {name: dict(path=str(path), sha256=_sha(path.read_bytes()))
              for name, path in files.items()}
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
                raise RuntimeError("self-test cannot identify CUDA bindings distribution metadata")
            entry.update(version=distribution.version,
                         metadata_path=str(paths[0]), metadata_sha256=_sha(paths[0].read_bytes()))
        elif filename is None:
            raise RuntimeError("self-test cannot identify imported runtime source: " + name)
        versions[name] = entry
    return torch, md, remap, device, dict(source=source, versions=versions)


def _cache_snapshot(md):
    return {name: dict(getattr(md, name)) for name in _CACHE_NAMES if hasattr(md, name)}


def _restore_scratch_caches(md, before):
    """Restore exact pre-test object bindings, including replaced smaller workspaces.

    Startup executes serially under our lock. Compiled kernels are deliberately
    retained. No global cache-clear function is called.
    """
    removed = {}
    for name, original in before.items():
        cache = getattr(md, name)
        removed[name] = sum(key not in original or original[key] is not value
                            for key, value in cache.items())
        for key in tuple(cache):
            if key not in original:
                del cache[key]
        for key, value in original.items():
            cache[key] = value
        if set(cache) != set(original) or any(cache[key] is not value for key, value in original.items()):
            raise RuntimeError("EP self-test scratch cache restoration failed")
    return removed


def _weights(torch, device):
    from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout
    gen = torch.Generator().manual_seed(SEED)
    e, h, n = 72, 4096, 2048
    w13 = torch.randint(0, 256, (e, 2*n, h//2), dtype=torch.uint8, generator=gen).to(device)
    w2 = torch.randint(0, 256, (e, h, n//2), dtype=torch.uint8, generator=gen).to(device)
    # onepass9 ran inside vLLM's BF16 default-dtype context. Keep that exact
    # failed fixture explicit; changing RNG dtype would change its input data.
    s13 = (torch.rand(e, 2*n, h//16, dtype=torch.bfloat16, generator=gen)*.05+.01).to(torch.float8_e4m3fn).to(device)
    s2 = (torch.rand(e, h, n//16, dtype=torch.bfloat16, generator=gen)*.05+.01).to(torch.float8_e4m3fn).to(device)
    sf13 = flashinfer_convert_sf_to_mma_layout(s13.reshape(e*2*n, h//16), m=2*n, k=h, num_groups=e)
    sf2 = flashinfer_convert_sf_to_mma_layout(s2.reshape(e*h, n//16), m=h, k=n, num_groups=e)
    return w13, sf13, w2, sf2


def _routing(torch, rows, kind, gen, device, changed=False):
    token = torch.arange(rows, device=device)[:, None]
    slot = torch.arange(8, device=device)[None, :]
    if kind == "remote" and not changed:
        global_ids = (72 + (token + slot) % 216).expand(rows, 8)
    elif kind == "concentrated":
        global_ids = ((slot + (3 if changed else 0)) % 72).expand(rows, 8)
    elif kind == "duplicate":
        global_ids = torch.full((rows, 8), 6 if changed else 5, device=device)
    else:
        global_ids = (token*37 + slot*31 + (137 if changed else 0)) % 288
    weights = torch.rand(rows, 8, dtype=torch.bfloat16, device=device, generator=gen)
    weights /= weights.sum(dim=1, keepdim=True)
    if kind == "zeros":
        weights[:, ::2] = 0
    return global_ids.to(torch.int32).contiguous(), weights


def _synthetic_wrapper(caller, torch, device, sf13, sf2):
    # The real wrapper's alpha/scale properties alias quant_config descriptors.
    # Bind methods to a fresh namespace, never copy or mutate that descriptor.
    scales = torch.linspace(.8, 1.2, 72, dtype=torch.bfloat16, device=device)
    methods = ("_remap_ep_tensors", "_apply_ep_compact", "_apply_ep_local_prefill",
               "_apply_ep_fixed", "_apply_ep_zero_weight_micro", "_ep_tail_padded_micro",
               "_ep_tail_buffers", "_try_apply_ep_fused_short_decode")
    fixture_type = type("_EPFixture", (SimpleNamespace,),
                        {name: getattr(type(caller), name) for name in methods})
    # Methods live on the throwaway type: no obj -> bound method -> obj cycle
    # may retain large scratch/scale allocations between the bounded cases.
    obj = fixture_type(num_local_experts=72, _kernel_num_experts=72,
        local_expert_offset=0, max_num_tokens=8192, _ep_zero_weight_micro=True,
        _activation_str="swigluoai_uninterleave", _swiglu_alpha=1.,
        _swiglu_beta=0., _swiglu_limit=10., w1_sf_mma=sf13, w2_sf_mma=sf2,
        g1_alphas=scales, g2_alphas=scales.flip(0).contiguous(), _fc2_input_scale=scales,
        _ep_fixed_workspace=caller._ep_fixed_workspace,
        _ep_zero_weight_workspace=caller._ep_zero_weight_workspace, _ep_tail_key=None)
    return obj


def _scratch(obj, torch, rows, device, *, weights_dtype):
    for name, dtype in (("_ep_ids", torch.int32), ("_ep_scales", weights_dtype),
            ("_ep_long", torch.int64), ("_ep_mapped", torch.int32), ("_ep_remote", torch.bool),
            ("_ep_tmp_a", torch.bool), ("_ep_tmp_b", torch.bool)):
        setattr(obj, name, torch.empty((rows, 8), dtype=dtype, device=device))


def _q0_records(counts, bases, tokens, weights, packed, scale, *, rows, expected):
    """Canonical valid-route hashes; uninitialized padding is never evidence."""
    result = []
    observed = []
    for expert, count in enumerate(counts):
        if count < 0 or count > rows*8 or bases[expert] < 0:
            raise AssertionError("invalid Q0 row metadata")
        for row in range(bases[expert]*128, bases[expert]*128 + count):
            token = tokens[row]
            if not 0 <= token < rows:
                raise AssertionError("Q0 token map escaped fixture")
            wb = struct.pack("<f", weights[row]).hex()
            observed.append((expert, token, wb))
            a = packed[row*2048:(row+1)*2048]
            base = (row//128)*32768 + (row%32)*16 + ((row//32)%4)*4
            offsets = [base + (sf//4)*512 + sf%4 for sf in range(256)]
            if len(a) != 2048 or max(offsets) >= len(scale):
                raise AssertionError("Q0 packed/scale storage escaped fixture")
            sf = bytes(scale[offset] for offset in offsets)
            result.append((expert, token, wb, _sha(a), _sha(sf)))
    if sorted(observed) != sorted(expected):
        raise AssertionError("Q0 route IDs/weights differ from original local nonzero routes")
    result.sort()
    return result


def _q0(md, device, rows, ids, weights):
    key = (72, 72, 4096, 2048, 8, str(device), "dynamic", "nvfp4", "swigluoai_uninterleave", 128)
    workspace = md._WORKSPACE_CACHE.get(key)
    if workspace is None or workspace.routed_rows_capacity < rows*8:
        raise AssertionError("missing exact EP-local dynamic workspace")
    expected = [(expert, token, struct.pack("<f", weight).hex())
                for token, (route, values) in enumerate(zip(ids.cpu().tolist(), weights.cpu().tolist()))
                for expert, weight in zip(route, values) if 0 <= expert < 72 and weight != 0]
    records = _q0_records(workspace.row_counts.cpu().tolist(), workspace.expert_tile_base.cpu().tolist(),
        workspace.token_map.cpu().tolist(), workspace.token_weights.cpu().tolist(),
        _tensor_bytes(workspace.packed_input), _tensor_bytes(workspace.packed_input_scale),
        rows=rows, expected=expected)
    return records, dict(valid_routes=len(records), sha256=_sha(json.dumps(records).encode()),
                         row_counts=workspace.row_counts.cpu().tolist(),
                         scope="candidate valid Q0 routes; not a stock packed-input equivalence proof")


def _compare_q0(first, current):
    if first != current:
        differences = [dict(index=i, first=a, current=b)
                       for i, (a, b) in enumerate(zip(first, current)) if a != b][:8]
        raise AssertionError(dict(verdict="Q0_REPLAY_BYTES_CHANGED", first_rows=len(first),
                                  current_rows=len(current), differences=differences))


def _prepare_check(obj, torch, remap, x, ids, weights, expert_map):
    actual = obj._ep_tail_buffers(8, x, ids, weights, ids_dtype=torch.int32)
    if actual is None:
        raise RuntimeError("T6 self-test staging allocation failed")
    px, pi, pw, _ = actual
    if not remap.try_prepare_ep_short_decode(x, ids, weights, expert_map=expert_map,
            num_local_experts=72, local_expert_offset=0, pad_x=px, pad_ids=pi, pad_weights=pw):
        raise AssertionError("T6 fused preparation did not admit its exact fixture")
    mapped, mapped_weights = obj._remap_ep_tensors(ids, weights, expert_map)
    expected_x = torch.cat((x, x[:1].expand(2, -1)))
    expected_ids = torch.cat((mapped, mapped[:1].expand(2, -1)))
    expected_w = torch.cat((mapped_weights, torch.zeros_like(mapped_weights[:2])))
    proof = {}
    for name, candidate, reference in (("X", px, expected_x), ("ids", pi, expected_ids),
                                        ("weights", pw, expected_w)):
        if candidate.dtype != reference.dtype or tuple(candidate.shape) != tuple(reference.shape):
            raise AssertionError(dict(verdict="T6_PREP_METADATA_FAIL", tensor=name,
                actual_dtype=str(candidate.dtype), reference_dtype=str(reference.dtype),
                actual_shape=tuple(candidate.shape), reference_shape=tuple(reference.shape)))
        a, b = _tensor_bytes(candidate), _tensor_bytes(reference)
        proof[name] = dict(actual_sha256=_sha(a), reference_sha256=_sha(b), exact=a == b)
        if a != b:
            first = next((i for i, (aa, bb) in enumerate(zip(a, b)) if aa != bb), min(len(a), len(b)))
            raise AssertionError(dict(verdict="T6_PREP_BYTES_FAIL", tensor=name, byte=first,
                                      actual=a[first:first+1].hex(), reference=b[first:first+1].hex()))
    return proof


def _case(caller, torch, md, remap, device, tensors, case, sink):
    name, rows, kind = case
    w13, sf13, w2, sf2 = tensors
    obj = _synthetic_wrapper(caller, torch, device, sf13, sf2)
    gen = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn(rows, 4096, dtype=torch.bfloat16, device=device, generator=gen)*.5
    ids, weights = _routing(torch, rows, kind, gen, device)
    _scratch(obj, torch, rows, device, weights_dtype=weights.dtype)
    if kind == "short":
        ids.copy_(torch.tensor([[72+i for i in range(8)], list(range(8)),
            [0, 72, 1, 144, 2, -1, 287, 3], [71, 0, 288, -1, 72, 1, 2, 3],
            [5]*8, [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32, device=device))
        weights[2, ::2] = 0
        weights[5, 0] = -0.0
    expert_map = torch.arange(288, dtype=torch.int32, device=device)
    expert_map[72:] = -1
    out = torch.empty_like(x)
    side = torch.cuda.Stream(device=device)
    scales = dict(fc1_input=obj.g1_alphas, fc1_alpha=obj.g1_alphas,
                  fc2_input=obj._fc2_input_scale, fc2_alpha=obj.g2_alphas)
    sink.update(case=name, phase="initial", controls=[], candidate=[], q0=[], inputs=[],
                self_variation=[], preparation=[])

    def eager(candidate, nondefault=False):
        out.fill_(float("nan"))
        def call():
            if candidate and kind == "short":
                got = obj._try_apply_ep_fused_short_decode(out, x, w13, w2, ids, weights, expert_map)
                if got is None:
                    raise AssertionError("T6 candidate silently fell back")
            else:
                mapped, values = obj._remap_ep_tensors(ids, weights, expert_map,
                                                     fuse_local_prefill=candidate)
                method = obj._apply_ep_local_prefill if candidate else (
                    obj._apply_ep_fixed if kind == "short" else obj._apply_ep_compact)
                got = method(out, x, w13, w2, mapped, values)
            if got is not out:
                raise AssertionError("self-test did not write the supplied output")
        if nondefault:
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                call()
            torch.cuda.current_stream(device).wait_stream(side)
        else:
            call()
        torch.cuda.synchronize(device)
        return out.clone()

    for changed in (False, True):
        if changed:
            pointers = tuple(t.data_ptr() for t in (x, ids, weights, obj.g1_alphas))
            x.mul_(-.75)
            obj.g1_alphas.mul_(1.125)
            new_ids, new_weights = _routing(torch, rows, kind, gen, device, changed=True)
            ids.copy_(new_ids); weights.copy_(new_weights)
            if pointers != tuple(t.data_ptr() for t in (x, ids, weights, obj.g1_alphas)):
                raise AssertionError("changed fixture replaced original storage")
        sink["phase"] = "changed" if changed else "initial"
        identity = _inputs(x, ids, weights, scales)
        sink["inputs"].append(identity)
        if kind == "short":
            sink["preparation"].append(_prepare_check(obj, torch, remap, x, ids, weights, expert_map))
            if not changed:
                # The full numerical fixture remains onepass9's BF16 one.
                # These byte-only cases use the wrapper's actual same-dtype
                # legacy-remap scratch without additional MoE executions.
                try:
                    for dtype in (torch.float32, torch.float16):
                        _scratch(obj, torch, rows, device, weights_dtype=dtype)
                        proof = _prepare_check(obj, torch, remap, x, ids, weights.to(dtype), expert_map)
                        sink["preparation"].append(dict(input_weights_dtype=str(dtype), tensors=proof))
                finally:
                    _scratch(obj, torch, rows, device, weights_dtype=weights.dtype)
        b1, b2, b3 = eager(False), eager(False), eager(False)
        sink["controls"].append([check_control(b1, b2), check_control(b1, b3), check_control(b2, b3)])
        context = dict(result=sink, third=b3, inputs=x, route_ids=ids, route_weights=weights,
                       expert_map=expert_map, scales=scales)
        first_q0, first_candidate = None, None
        for label, nondefault in (("C1", False), ("C2", False), ("C3", True)):
            sink["phase"] = ("changed" if changed else "initial") + "-" + label
            candidate = eager(True, nondefault)
            if kind != "short":
                q0, proof = _q0(md, device, rows, ids, weights)
                sink["q0"].append(dict(phase=sink["phase"], **proof))
                if first_q0 is None:
                    first_q0 = q0
                else:
                    _compare_q0(first_q0, q0)
            sink["candidate"].append(compare(candidate, b1, b2, failure_context=context))
            if first_candidate is None:
                first_candidate = candidate
            else:
                l2, peak = row_errors(candidate, first_candidate)
                sink["self_variation"].append(dict(phase=sink["phase"],
                    max_row_relative_l2=float(l2.max()), max_row_relative_abs=float(peak.max()),
                    scope="candidate-self variation only; original numerical gate unchanged"))
            if kind == "remote" and not changed:
                if not bool((b1 == 0).all()) or not bool((candidate == 0).all()):
                    raise AssertionError("all-remote output was not exact zero")
        if _inputs(x, ids, weights, scales) != identity:
            raise AssertionError("self-test mutated fixture inputs while comparing")
    sink.update(verdict="PASS", phase="complete")


def _run(wrapper, torch, md, remap, device, receipt):
    tensors = _weights(torch, device)
    receipt["weights"] = {name: _tensor_identity(tensor) for name, tensor in
                          zip(("w13", "sf13", "w2", "sf2"), tensors)}
    receipt["cases"] = []
    for case in (*CASES, ("short6", 6, "short")):
        cell = dict(started_at=time.time(), memory_before=_memory(torch, device))
        receipt["cases"].append(cell)
        before = _cache_snapshot(md)
        primary = None
        try:
            _case(wrapper, torch, md, remap, device, tensors, case, cell)
        except BaseException as exc:
            primary = exc
        finally:
            # Free fixture-scale cache views after each case. Otherwise their
            # per-case alpha pointers retain another 108 MiB of converted SF.
            try:
                torch.cuda.synchronize(device)
                _restore_scratch_caches(md, before)
            except BaseException as exc:
                cell["cleanup_error"] = repr(exc)
                if primary is None:
                    primary = exc
            cell.update(completed_at=time.time(), memory_after=_memory(torch, device))
            cell["duration_s"] = cell["completed_at"] - cell["started_at"]
        if primary is not None:
            raise primary
    receipt["micro_keys"] = _micro_keys(md)


def _memory(torch, device):
    return dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                process_peak_allocated=torch.cuda.max_memory_allocated(device),
                process_peak_reserved=torch.cuda.max_memory_reserved(device))


def _micro_keys(md):
    candidate, control = [], []
    for key in md._MICRO_KERNEL_CACHE:
        if len(key) <= 17:
            continue
        if key[2:9] == (72, 72, 8, 4096, 2048, 8, 64):
            if (key[10] != (16, 128) or key[17] != 72 or
                    key[-4:] != ("glm53_ep_micro_scatter_fp32_v1", "glm53_ep_micro_direct_scatter_v1",
                                 "glm53_ep_micro_shared_fc1_a_v1", "glm53_ep_micro_m16_v1")):
                raise AssertionError("T6 padded candidate did not select exact M16 sentinel shared-A direct FP32 micro")
            candidate.append(key)
        if key[2:9] == (72, 72, 8, 4096, 2048, 1, 8):
            if (key[10] != (64, 128) or key[17] is not None or
                    key[-1] != "glm53_ep_micro_scatter_fp32_v1"):
                raise AssertionError("T6 fixed control did not retain M64 shared FP32 micro")
            control.append(key)
    if not candidate or not control:
        raise AssertionError("missing compiled T6 M16/M64 keys")
    # Cache keys contain torch.dtype objects; preserve their exact repr without
    # making the mandatory startup receipt depend on a custom JSON encoder.
    return dict(candidate=[repr(key) for key in candidate], control=[repr(key) for key in control])


def _caller_state(wrapper):
    fields = ("g1_alphas", "g2_alphas", "w1_scale", "w2_scale", "w1_sf_mma",
              "w2_sf_mma", "_fc2_input_scale")
    result = {}
    for name in fields:
        value = getattr(wrapper, name, None)
        if value is None:
            continue
        try:
            version = value._version
        except RuntimeError as exc:
            if "version counter" not in str(exc):
                raise
            version = "inference-no-version-counter"
        result[name] = (id(value), value.data_ptr(), version, tuple(value.shape))
    return result


def ensure_ep_local_selftest(wrapper, *, device):
    """Run once after both pinned decode workspaces exist, before readiness.

    Each worker executes independently. Any exception permanently rejects this
    process/source/device. The caller must propagate it and may publish model
    readiness only after every rank has returned PASS. No production weights,
    RNG state, HTTP requests, or external probe modules are used or modified.
    """
    with _LOCK:
        torch, md, remap, device, provenance = _runtime(wrapper, device)
        key = (os.getpid(), str(device), _sha(json.dumps(provenance, sort_keys=True).encode()),
               SEED, CASES, "short6-v1")
        previous = _STATES.get(key)
        if previous is not None:
            if previous["verdict"] != "PASS":
                raise RuntimeError("EP startup self-test previously failed or is reentrant: " + previous["verdict"])
            return previous
        receipt = dict(schema=1, verdict="RUNNING", pid=os.getpid(), device=str(device),
            seed=SEED, started_at=time.time(), provenance=provenance,
            performance_acceptance=False, full_sanitizer_acceptance=False,
            scope="bounded synthetic serving-runtime startup canary; no component timing",
            compiled_kernel_cache_retained=True)
        _STATES[key] = receipt
        before = _cache_snapshot(md)
        caller_before = None
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        primary = None
        try:
            caller_before = _caller_state(wrapper)
            _run(wrapper, torch, md, remap, device, receipt)
        except BaseException as exc:
            primary = exc
            receipt.update(verdict="FAIL", error=repr(exc))
        finally:
            try:
                torch.cuda.synchronize(device)
                receipt["scratch_cache_removed"] = _restore_scratch_caches(md, before)
                receipt["scratch_cache_restored"] = True
                if caller_before is not None and _caller_state(wrapper) != caller_before:
                    raise RuntimeError("EP self-test mutated caller scale/descriptor storage")
                receipt["caller_scale_storage_unchanged"] = True
            except BaseException as exc:
                receipt.update(verdict="FAIL", cleanup_error=repr(exc))
                if primary is None:
                    primary = exc
        receipt["completed_at"] = time.time()
        receipt["memory"] = dict(allocated_before=allocated_before, reserved_before=reserved_before,
            allocated_after=torch.cuda.memory_allocated(device), reserved_after=torch.cuda.memory_reserved(device),
            process_peak_allocated=torch.cuda.max_memory_allocated(device),
            process_peak_reserved=torch.cuda.max_memory_reserved(device),
            scope="process peaks include earlier model loading; no reset or canary-only peak claim")
        if primary is not None:
            _LOG.error("[ep-local-selftest] FAIL %s", json.dumps(receipt, sort_keys=True))
            raise RuntimeError("EP startup self-test failed; serving readiness refused") from primary
        receipt["verdict"] = "PASS"
        # Required evidence must survive third-party logger thresholds. Cached
        # PASS returns above, so each process/source/device publishes only once.
        print("[ep-local-selftest] PASS " + json.dumps(receipt, sort_keys=True), flush=True)
        return receipt
