#!/usr/bin/env python3
"""CPU differential oracle for the V4.1 candidate indexer's compact scores.

Executes the pinned vendor Indexer.forward/select_candidate_blocks AST with
synthetic, already-quantized BF16 Q/K and projection outputs. No model weights,
TileLang, vLLM, distributed processes, GPU, or downloads are needed. Collective
element counts describe communication volume, never GPU speed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

REFERENCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
REFERENCE_REVISION = "fb2764a5cf321eaa5070ca8f9e892818f477c16d"
ROOT = Path(__file__).resolve().parents[1]


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference(path, world_size):
    """Only execute the three original definitions; no vendor import side effects."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("reference source SHA256 differs from the reviewed V4.1 release")
    tree = ast.parse(raw, filename=str(path))
    names = {"Indexer", "select_candidate_blocks", "apply_rotary_emb"}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    assert {node.name for node in selected} == names
    module = ModuleType("dsv41_pinned_reference")
    ns = vars(module)
    ns.update(__file__=str(path), torch=torch, ModelArgs=object,
              nn=nn, F=F, world_size=world_size, fp4_block_size=32,
              shared_attn=SimpleNamespace(),
              fp4_act_quant=lambda value, block, inplace: None,
              dist=SimpleNamespace(all_reduce=lambda value: None))
    unit = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec", dont_inherit=True), ns)
    # The score arithmetic is copied as AST nodes, without rewriting operations,
    # to observe the exact pre-collective BF16 tensor in both world-size cases.
    indexer = next(node for node in selected if isinstance(node, ast.ClassDef))
    forward = next(node for node in indexer.body if isinstance(node, ast.FunctionDef)
                   and node.name == "forward")
    assignments = [node for node in forward.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "index_score" for t in node.targets)]
    assert len(assignments) == 2
    score_fn = ast.parse("def source_score(q, index_k, weights):\n    pass\n").body[0]
    score_fn.body = assignments + [ast.Return(value=ast.Name(id="index_score", ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[score_fn], type_ignores=[])),
                 str(path), "exec"), ns)
    ns["_extracted_ast_sha256"] = hashlib.sha256(ast.dump(unit).encode()).hexdigest()
    ns["_module"] = module
    return ns


def bf16_sum(values):
    """Explicit deterministic CPU collective simulator, identical for both paths."""
    result = values[0].clone()
    for value in values[1:]:
        result.add_(value)
    return result


def tensor_sha(value):
    import torch
    raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def fixture(torch, shape, generator, kind):
    if kind == "ties":
        return torch.zeros(shape, dtype=torch.bfloat16)
    if kind == "fp4_like":
        levels = torch.tensor([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6]) / 8
        return levels[torch.randint(len(levels), shape, generator=generator)].to(torch.bfloat16)
    return (torch.randn(shape, generator=generator) * .25).to(torch.bfloat16)


def make_indexer(ns, q, raw_weights, width, ratio, topk, blocks, *, source=False):
    import torch
    cls = ns["Indexer"]
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj.owns_k = False
    obj.dim = 5120
    obj.q_lora_rank = 1280
    obj.compress_ratio = ratio
    obj.rope_head_dim = 64
    obj.n_heads = 32
    obj.n_local_heads = q.shape[2]
    obj.index_head_dim = 128
    obj.softmax_scale = 128 ** -.5
    obj.index_topk = topk
    obj.candidate_topk_blocks = blocks
    obj.candidate_block_size = 8
    obj.is_candidate_source = source
    obj.uses_candidates = not source
    obj.freqs_cis = torch.ones(width * ratio + q.shape[1], 32, dtype=torch.complex64)
    obj.wq_b = lambda unused: q.clone().flatten(2)
    obj.weights_proj = lambda unused: raw_weights.clone()
    return obj


def run_case(core, reference_path, spec, world_size, seed):
    import torch
    generator = torch.Generator().manual_seed(seed)
    b, queries, width = spec["batch"], spec["queries"], spec["width"]
    ratio, start = spec["ratio"], spec["start"]
    assert (start + queries) // ratio == width
    ns = load_reference(reference_path, world_size)
    key = fixture(torch, (b, width, 128), generator, spec["kind"])
    q_all = fixture(torch, (b, queries, 32, 128), generator, spec["kind"])
    raw_all = (torch.randn((b, queries, 32), generator=generator) * 3).to(torch.bfloat16)
    raw_all[..., 0] = -8  # rectification must precede the negative head weighting
    raw_all[..., 1] = 8
    weights_all = raw_all * (128 ** -.5 * 32 ** -.5)
    lens = ((torch.arange(1, queries + 1) // ratio).unsqueeze(-1)
            if start == 0 else (start + queries) // ratio)
    source_scores = fixture(torch, (b, queries, width), generator, spec["kind"])
    if start == 0:
        source_scores.masked_fill_(torch.arange(width) >= lens, -torch.inf)
    mask = ns["select_candidate_blocks"](source_scores, lens, spec["blocks"], 8)
    ids, core_mask = core.select_candidate_ids(source_scores.clone(), lens,
                                               spec["blocks"], 8, return_mask=True)
    assert torch.equal(core_mask, mask), "candidate source differs from the actual source selector"
    assert ids.dtype == torch.int32 and ids.shape[:2] == (b, queries)
    assert ids.shape[-1] == min(width, spec["blocks"] * 8)
    reconstructed = torch.zeros_like(mask)
    for bi in range(b):
        for qi in range(queries):
            valid = ids[bi, qi][ids[bi, qi] >= 0].long()
            assert torch.equal(valid, valid.sort().values) and len(valid.unique()) == len(valid)
            reconstructed[bi, qi, valid] = True
    assert torch.equal(reconstructed, mask), "IDs do not encode the original selected positions"

    head_count = 32 // world_size
    q_parts = q_all.split(head_count, dim=2)
    w_parts = weights_all.split(head_count, dim=2)
    raw_parts = raw_all.split(head_count, dim=2)
    dense_parts = [ns["source_score"](q.clone(), key, w) for q, w in zip(q_parts, w_parts)]
    assert all(x.dtype == torch.bfloat16 for x in dense_parts)
    dense_total = bf16_sum(dense_parts)
    safe = ids.clamp_min(0).long()
    gathered_parts = [x.gather(-1, safe).masked_fill(ids < 0, 0) for x in dense_parts]
    gathered_total = bf16_sum(gathered_parts)
    payloads = []
    references, actuals = [], []
    for rank, (q, weights, raw_weights) in enumerate(zip(q_parts, w_parts, raw_parts)):
        reference_calls = []
        def dense_reduce(value):
            assert torch.equal(value, dense_parts[rank])
            reference_calls.append(list(value.shape))
            value.copy_(dense_total)
        ns["dist"] = SimpleNamespace(all_reduce=dense_reduce)
        ns["shared_attn"] = SimpleNamespace(index_k=key, candidates=mask)
        obj = make_indexer(ns, q, raw_weights, width, ratio, spec["topk"], spec["blocks"])
        x = torch.zeros((b, queries, 1), dtype=torch.bfloat16)
        expected = obj.forward(x, x, None, start, spec["offset"])
        assert len(reference_calls) == (1 if world_size > 1 else 0)
        calls = []
        def compact_reduce(value):
            assert value.is_contiguous() and value.dtype == torch.bfloat16
            assert torch.equal(value, gathered_parts[rank]), "local compact BF16 score drift"
            calls.append(dict(shape=list(value.shape), elements=value.numel(), bytes=value.numel()*value.element_size()))
            value.copy_(gathered_total)
        compact = core.compact_index_scores(q.clone(), key, weights, ids,
                                             compact_reduce if world_size > 1 else None,
                                             query_chunk_size=1)
        assert torch.equal(compact, gathered_total), "reduced compact BF16 score drift"
        assert len(calls) == (1 if world_size > 1 else 0), "collective count changed"
        actual = core.compact_topk(compact, ids, lens, spec["offset"], spec["topk"],
                                   full_width=width, query_chunk_size=1)
        assert torch.equal(actual, expected), "full-domain topk/causal/offset mismatch"
        references.append(tensor_sha(expected)); actuals.append(tensor_sha(actual))
        payloads.extend(calls)
    return dict(name=spec["name"], world_size=world_size, kind=spec["kind"],
                batch=b, queries=queries, width=width, candidate_width=ids.shape[-1],
                negative_head_weights=True, source_selector_exact=True,
                compact_scores_bf16_exact=True, indices_exact=True,
                reference_index_sha256=references, candidate_index_sha256=actuals,
                input_sha256=dict(q=tensor_sha(q_all), key=tensor_sha(key), raw_weights=tensor_sha(raw_all)),
                collective_payloads=payloads, dense_payload_elements=b*queries*width,
                compact_payload_elements=ids.numel(),
                source_ast_sha256=ns["_extracted_ast_sha256"])


def cases():
    common = dict(batch=2, queries=1, ratio=1, blocks=3, topk=9, offset=128)
    return [
        dict(common, name="early_causal_partial", queries=17, width=17, start=0, blocks=1, kind="fp4_like"),
        dict(common, name="ratio2_zero_visible", queries=17, width=8, ratio=2, start=0, blocks=1, kind="random"),
        dict(common, name="partial_last_block", width=37, start=36, kind="fp4_like"),
        dict(common, name="random_negative_weights", width=49, start=48, kind="random"),
        dict(common, name="ties_full_domain", width=65, start=64, blocks=1, kind="ties"),
        dict(common, name="short_context", width=127, start=126, kind="fp4_like"),
        dict(common, name="changed_candidates_first", width=49, start=48, kind="random"),
        dict(common, name="changed_candidates_second", width=49, start=48, kind="random"),
        dict(common, name="released_long_decode", batch=1, width=32777, start=32776,
             blocks=2048, topk=512, kind="fp4_like"),
    ]


def adapter_cases(adapter, reference_path, world_size):
    """Drive actual patched instances across dense/compact transitions.

    Projection outputs and post-quantized keys are synthetic; all score,
    source-selection, masking, collective and final top-k operations execute
    the pinned reference forward or the installed adapter. The CPU collective
    order is specified, not a prediction about NCCL reduction trees.
    """
    import torch
    layers = (20, 24, 28, 32, 36)
    # Same geometry but changed data catches reuse of a prior source's IDs.
    steps = [
        ("long_decode_first", 1, 32777, 1, 32776, 0, True),
        ("long_decode_new_candidates", 1, 32777, 1, 32776, 0, True),
        ("long_decode_offset", 1, 32777, 1, 32776, 128, True),
        ("short_decode_fallback", 1, 127, 1, 126, 128, False),
        ("prefill_fallback", 1, 3, 3, 0, 128, False),
        ("batch2_long_decode_fallback", 2, 16385, 1, 16384, 128, False),
    ]
    rank_state = []
    for rank in range(world_size):
        ref, opt = load_reference(reference_path, world_size), load_reference(reference_path, world_size)
        models = []
        for ns in (ref, opt):
            model = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(indexer=None)) for _ in range(40)])
            for layer in layers:
                q = torch.zeros(1, 1, 32 // world_size, 128, dtype=torch.bfloat16)
                raw = torch.zeros(1, 1, 32 // world_size, dtype=torch.bfloat16)
                instance = make_indexer(ns, q, raw, 32780, 1, 512, 2048, source=layer == 20)
                instance.owns_k = layer == 20
                model.layers[layer].attn.indexer = instance
            models.append(model)
        disabled = adapter.install_reference_indexer(models[1], opt["_module"], enabled=False)
        assert disabled.active is False
        handle = adapter.install_reference_indexer(models[1], opt["_module"], enabled=True)
        assert handle.active is True
        rank_state.append((ref, opt, *models, handle))
    output, prior_masks = [], None
    try:
        for step_index, (name, batch, width, queries, start, offset, compact_expected) in enumerate(steps):
            rng = torch.Generator().manual_seed(710000 + step_index)
            key = fixture(torch, (batch, width, 128), rng, "fp4_like")
            qs = {layer: fixture(torch, (batch, queries, 32, 128), rng, "fp4_like") for layer in layers}
            raw = {layer: (torch.randn((batch, queries, 32), generator=rng)*3).to(torch.bfloat16) for layer in layers}
            for value in raw.values():
                value[..., 0], value[..., 1] = -8, 8
            q_parts = {layer: qs[layer].split(32 // world_size, dim=2) for layer in layers}
            raw_parts = {layer: raw[layer].split(32 // world_size, dim=2) for layer in layers}
            local_scores = {layer: [rank_state[r][0]["source_score"](
                q_parts[layer][r].clone(), key, raw_parts[layer][r] * (128**-.5 * 32**-.5))
                for r in range(world_size)] for layer in layers}
            total_scores = {layer: bf16_sum(local_scores[layer]) for layer in layers}
            masks, payloads, output_hashes = [], [], []
            x = torch.zeros(batch, queries, 5120, dtype=torch.bfloat16)
            qr = torch.zeros(batch, queries, 1280, dtype=torch.bfloat16)
            for rank, (ref, opt, ref_model, opt_model, handle) in enumerate(rank_state):
                # Keep each module's shared object stable, as the real runtime does.
                for ns, model in ((ref, ref_model), (opt, opt_model)):
                    ns["shared_attn"].index_k = key
                    for layer in layers:
                        instance = model.layers[layer].attn.indexer
                        instance.wq_b = lambda unused, q=q_parts[layer][rank]: q.clone().flatten(2)
                        instance.weights_proj = lambda unused, w=raw_parts[layer][rank]: w.clone()
                before = handle.counters
                expected = {}
                for layer in layers:
                    def reference_reduce(value, layer=layer):
                        assert torch.equal(value, local_scores[layer][rank])
                        value.copy_(total_scores[layer])
                    ref["dist"].all_reduce = reference_reduce
                    expected[layer] = ref_model.layers[layer].attn.indexer(x, qr, None, start, offset)
                source_mask = ref["shared_attn"].candidates
                masks.append(tensor_sha(source_mask))
                # Rebuild expected gathered values from the ORIGINAL mask, not
                # by calling the candidate's selector a second time.
                selected = torch.arange(width).expand(batch, queries, width).masked_fill(~source_mask, width).sort(dim=-1).values
                selected = selected[..., :min(width, 16384)]
                valid = selected < width
                safe = selected.clamp_max(width - 1)
                for layer in layers:
                    expect_compact = compact_expected and layer != 20
                    def candidate_reduce(value, layer=layer, expect_compact=expect_compact):
                        expected_local = local_scores[layer][rank]
                        total = total_scores[layer]
                        if expect_compact:
                            expected_local = expected_local.gather(-1, safe).masked_fill(~valid, 0)
                            total = bf16_sum([part.gather(-1, safe).masked_fill(~valid, 0)
                                              for part in local_scores[layer]])
                        assert value.dtype == torch.bfloat16 and value.is_contiguous()
                        assert torch.equal(value, expected_local), "adapter changed the local BF16 score payload"
                        payloads.append(dict(rank=rank, layer=layer, compact=expect_compact,
                                             shape=list(value.shape), elements=value.numel()))
                        value.copy_(total)
                    opt["dist"].all_reduce = candidate_reduce
                    actual = opt_model.layers[layer].attn.indexer(x, qr, None, start, offset)
                    assert torch.equal(actual, expected[layer]), f"adapter {name} layer {layer} drift"
                    output_hashes.append(dict(rank=rank, layer=layer, sha256=tensor_sha(actual)))
                assert torch.equal(opt["shared_attn"].candidates, source_mask)
                delta = {key: handle.counters[key] - before[key] for key in before}
                assert delta == {"source_compact_steps": int(compact_expected),
                                 "consumer_compact_calls": 4 if compact_expected else 0,
                                 "dense_fallback_calls": 0 if compact_expected else 5}
                assert handle._state is None, "a completed layer sequence retained stale candidate state"
            if name == "long_decode_new_candidates":
                assert masks != prior_masks, "candidate-change fixture did not change the actual source mask"
            prior_masks = masks
            assert len(payloads) == (5*world_size if world_size > 1 else 0)
            output.append(dict(name=name, world_size=world_size, width=width, batch=batch, queries=queries,
                               start_pos=start, offset=offset, compact_consumers=compact_expected,
                               source_mask_sha256=masks, output_sha256=output_hashes,
                               collective_payloads=payloads, exact_original_forward=True))
    finally:
        for ref, opt, ref_model, opt_model, handle in rank_state:
            handle.restore()
            assert handle.active is False
            for layer in layers:
                assert opt_model.layers[layer].attn.indexer.forward.__func__ is opt["Indexer"].forward
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; preserve the prior receipt")
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    core_path = ROOT / "overlay/modules/dsv41_model/dsv41_indexer.py"
    core = load_file(core_path, "dsv41_indexer")
    adapter_path = ROOT / "overlay/modules/dsv41_model/dsv41_reference_adapter.py"
    adapter = load_file(adapter_path, "dsv41_reference_adapter_oracle_candidate")
    receipt = dict(schema=1, scope="synthetic CPU arithmetic/selection/collective simulator",
                   reference_sha256=REFERENCE_SHA256, reference_revision=REFERENCE_REVISION,
                   core_sha256=hashlib.sha256(core_path.read_bytes()).hexdigest(),
                   adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                   probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   torch_version=torch.__version__, passed=False, gpu_executed=False,
                   model_equivalence=False, gpu_speedup=None,
                   scope_limits=["synthetic projection outputs and post-quantization Q/K",
                                 "identity RoPE fixture; original arithmetic executes",
                                 "deterministic CPU BF16 sum simulates TP, not NCCL",
                                 "GPU backend and model execution remain unvalidated"], rows=[])
    try:
        for i, spec in enumerate(cases()):
            for world in (1, 4):
                row = run_case(core, args.reference, spec, world, 20260910 + i)
                receipt["rows"].append(row)
                print(json.dumps({k: row[k] for k in ("name", "world_size", "width", "candidate_width", "indices_exact")}), flush=True)
        receipt["adapter_rows"] = []
        for world in (1, 4):
            receipt["adapter_rows"].extend(adapter_cases(adapter, args.reference, world))
        receipt["static_collective_elements"] = [dict(context=s, batch=1, queries=1,
            full_elements=s, compact_elements=2048*8, ratio=s/(2048*8),
            scope="collective tensor only; full-width final topk allocation remains")
            for s in (131072, 1048576)]
        receipt["cuda_initialized"] = torch.cuda.is_initialized()
        assert receipt["cuda_initialized"] is False
        receipt["passed"] = True
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
