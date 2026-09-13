#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read one isolated engine tree's declarations and execute its CPU byte layout.

Invoked by step_source in a fresh process: baseline and working-tree modules must
never share Python's import cache. No weights, model execution or CUDA allocation.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS


def expression(node, env):
    """Only declaration arithmetic and explicitly supplied constructors/functions."""
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Attribute, ast.Subscript,
               ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
               ast.Mod, ast.Pow, ast.LShift, ast.USub, ast.UAdd, ast.Not, ast.IfExp,
               ast.Compare, ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
               ast.BoolOp, ast.And, ast.Or, ast.Dict, ast.List, ast.Tuple, ast.Call, ast.keyword)
    for part in ast.walk(node):
        if not isinstance(part, allowed) or isinstance(part, ast.Attribute) and part.attr.startswith('_'):
            raise ValueError(f"unsupported declaration: {ast.unparse(node)}")
        if isinstance(part, ast.Call):
            function = ast.unparse(part.func)
            if function not in ('dict', 'bool', 'int', 'float', 'sched.chunk_for', 'ExecutionPlan', 'sched.Contract'):
                raise ValueError(f"unsupported declaration call: {function}")
    return eval(compile(ast.Expression(node), '<engine declaration>', 'eval'),
                {"__builtins__": {}, "dict": dict, "bool": bool, "int": int, "float": float}, env)


def assignment(tree, name):
    matches = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(matches) != 1:
        raise ValueError(f"expected one engine declaration of {name}, found {len(matches)}")
    return matches[0]


def probe(root: Path, settings: dict, config=None) -> dict:
    sys.path.insert(0, str(root))
    from engine.base import scheduler as sched
    from engine.base.kernel_shape import MEASURED
    from engine.profiles.glm53 import draft_policy, facts
    from engine.profiles.glm53.caches import layout, snapshot_layout, stage_bytes
    from engine.profiles.glm53.execution import ExecutionPlan

    boot = ast.parse((root / 'engine/profiles/glm53/boot.py').read_text())
    env = dict(facts=facts, sched=sched, ExecutionPlan=ExecutionPlan)
    for name in ('TOKEN_BUDGET', 'MAX_SEQS', 'MAX_WAIT_S'):
        env[name] = expression(assignment(ast.Module(body=boot.body, type_ignores=[]), name), env)
    declared = next(n for n in boot.body if isinstance(n, ast.FunctionDef) and n.name == 'declared')
    # Serving defaults can move into the pure policy module. Resolve the names
    # imported by this snapshot's declaration, including aliases; older boot
    # sources with inline defaults need no SERVING_POLICY symbol.
    for node in ast.walk(declared):
        if isinstance(node, ast.ImportFrom) and node.module == draft_policy.__name__:
            for alias in node.names:
                env[alias.asname or alias.name] = getattr(draft_policy, alias.name)
    env['gb10_defaults'] = expression(assignment(declared, 'gb10_defaults'), env)
    cfg = expression(assignment(declared, 'defaults'), env)
    unknown = settings.keys() - cfg.keys()
    if unknown:
        raise ValueError(f"unknown execution settings: {sorted(unknown)}")
    cfg.update(settings)
    for name in ('execution_overlap', 'early_observe', 'direct_mhc', 'prefill_project_tiles',
                 'nvme_mapped_staging', 'deferred_kda', 'terminal_mhc', 'draft_diagnostics'):
        if cfg[name] not in (0, 1):
            raise ValueError(f'{name} must be 0 or 1')
    draft_policy.DraftPolicy(cfg['draft_fc_precision'], cfg['draft_fc_calibration'], bool(cfg['draft_diagnostics']))
    env['cfg'] = cfg
    plans = [n for n in ast.walk(boot) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'ExecutionPlan']
    if len(plans) != 1:
        raise ValueError('expected one boot ExecutionPlan constructor')
    plan = expression(plans[0], env)
    k = facts.SPEC_K
    if config is not None:
        f = facts.architecture(config)
        f = dataclasses.replace(f, kda_state_dtype=cfg['kda_state_dtype'])
        shape = f.kernel_shape()
        basis = 'supplied checkpoint config, validated by this tree'
    else:
        # The existing oracle's GLM topology, made explicit. A checkpoint config
        # can replace it; code edits to layout/shape/defaults are still executed.
        shape = MEASURED
        kinds = tuple('dsa' if i % 4 == 3 else 'kda' for i in range(45))
        f = NS(layers=45, kinds=kinds, block=facts.BLOCK, chunk_align=facts.CHUNK_ALIGN,
               spec_k=k, kv_lora=shape.attention.head_dim, idx_dim=shape.indexer.head_dim,
               kpool=shape.indexer.pool, conv=shape.linear.conv, kda_dim=shape.linear.k_dim,
               kda_heads_local=shape.linear.heads * shape.tp // facts.TP,
               kda_state_dtype=cfg['kda_state_dtype'], dense=(0, 1, 2),
               is_dsa=lambda layer: kinds[layer] == 'dsa')
        basis = 'GLM53 reference topology (45 layers, 11 DSA); this tree kernel_shape.MEASURED'
    env.update(F=f, drafter=NS(k=k), plan=plan, max_seqs=env['MAX_SEQS'])
    build = next(n for n in boot.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    env['token_budget'] = expression(assignment(build, 'token_budget'), env)
    contract = expression(assignment(build, 'contract'), env)
    chunk = sched.chunk_for(contract.chunk_align, contract.token_budget, contract.draft_slots)
    if k < 0 or facts.TP <= 0 or contract.max_running <= 0 or chunk <= 0:
        raise ValueError('engine declares an unusable serving shape')
    draft = shape.drafter or MEASURED.drafter
    if k and draft is None:
        raise ValueError('the source has no reference drafter geometry')
    draft_shape = (draft.layers, draft.window, draft.kv_heads, draft.head_dim) if k and draft is not None else None
    lay = layout(f, range(f.layers), draft_shape)
    by_kind = {}
    for field in lay.fields:
        size = math.prod(field.shape) * (4 if field.dtype == 'f32' else 2)
        by_kind[field.name] = by_kind.get(field.name, 0) + size
    return dict(facts=dict(spec_k=k, tp=facts.TP, block=facts.BLOCK, chunk_align=facts.CHUNK_ALIGN,
                           kv_dtype=facts.KV_DTYPE, kda_state_dtype=cfg['kda_state_dtype']),
                contract=dataclasses.asdict(contract), execution=dataclasses.asdict(plan), settings=cfg,
                prefill_chunk=chunk,
                compute_tile=plan.tile_rows if plan.prefill_tiles > 1 else chunk,
                geometry=dict(basis=basis, drafter_basis='this tree kernel_shape reference; no drafter checkpoint loaded',
                              layers=f.layers, moe_layers=f.layers-len(f.dense),
                              experts=shape.moe.experts, topk=shape.moe.topk, hidden=shape.hidden,
                              inter_local=shape.moe.inter // facts.TP),
                memory=dict(kv_block_bytes=lay.block_bytes, kv_bytes_per_token=lay.block_bytes/f.block,
                            slot_bytes=lay.slot_bytes, state_fields_bytes=by_kind,
                            resident_slots_bytes=(contract.max_running+1)*lay.slot_bytes,
                            snapshot_bytes=snapshot_layout(f, range(f.layers), draft_shape)[0],
                            boundary_stage_bytes=stage_bytes(f, range(f.layers), contract.max_running)))


if __name__ == '__main__':
    request = json.load(sys.stdin)
    try:
        result = probe(Path(sys.argv[1]), request.get('settings', {}), request.get('config'))
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
        sys.exit(1)
