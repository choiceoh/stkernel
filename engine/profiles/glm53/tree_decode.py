"""Explicit eager GLM tree experiment: DFlash -> cost selection -> target -> commit.

No serving flag selects this path. It owns one sequence exclusively, requires
greedy decoding and FP32 KDA, and refuses insufficient reservations/budgets.
Projections and routed experts see the whole tree once per layer. DSA selection
uses branch-private pool/tail data and gathers only the selected latent rows;
it never copies the full prefix KV or lets siblings enter one another's mask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from engine.modules import tree_attention, tree_kda
from engine.modules.speculative_tree import Tree, dflash_candidates, select
from engine.profiles.glm53.net import K_NORM_EPS, O_NORM_EPS, Step


def propose(drafter, field, slot, anchor, context, *, nodes=8, width=2, expansion=31,
            predictor=None, runtime_id, bytes_per_expert, fixed_node_bytes, cost_weight=1.):
    """Return rank 0's cost-selected tree; all ranks run the same draft collectives."""
    device = field.device
    unary, cand, proj = drafter.candidate_rows(field, torch.tensor([slot], device=device),
        torch.tensor([anchor], device=device), torch.tensor([context], device=device))
    selection, error = None, None
    if drafter.target.comm.rank == 0:
        try:
            candidates = dflash_candidates(anchor, unary[0], cand[0], proj[0],
                drafter.p["candidate_selector.predecessor_codebook"],
                drafter.p["candidate_selector.successor_codebook"], drafter.selector_alpha,
                width=width, max_nodes=expansion, predictor=predictor, runtime_id=runtime_id)
            selection = select(candidates, min(nodes, len(candidates)), bytes_per_expert=bytes_per_expert,
                               fixed_node_bytes=fixed_node_bytes, cost_weight=cost_weight)
        except (ValueError, IndexError, RuntimeError) as exc:
            error = str(exc)
    selection, error = drafter.target.comm.broadcast_object((selection, error))
    if error is not None:
        raise ValueError("rank 0 tree selection failed: " + error)
    return selection


@torch.inference_mode()
def decode_once(drafter, caches, field, *, seq, slot, anchor, context, budget, runtime_id,
                bytes_per_expert, fixed_node_bytes, nodes=8, width=2, expansion=31,
                cost_weight=1., predictor=None, eos=frozenset(), persistent_mlp=None,
                max_scratch_bytes=256 << 20, temperature=0.):
    """One complete experimental step, including accepted target-feature observation.

    The caller reserves context + spec_k + 1 tokens, owns the sequence, and
    uses the returned context/last output as the next step's context/anchor.
    `field` is the existing drafter field; no second draft cache is allocated.
    """
    if temperature != 0 or type(budget) is not int or budget <= 0:
        raise ValueError("experimental tree decode requires greedy mode and a positive budget")
    selection = propose(drafter, field, slot, anchor, context, nodes=nodes, width=width, expansion=expansion,
        predictor=predictor, runtime_id=runtime_id, bytes_per_expert=bytes_per_expert,
        fixed_node_bytes=fixed_node_bytes, cost_weight=cost_weight)
    with Verification(drafter.target, caches, selection.tree, seq=seq, slot=slot, context=context,
                      max_scratch_bytes=max_scratch_bytes, persistent_mlp=persistent_mlp) as run:
        run.verify(aux_layers=drafter.F.aux_layers)
        result = run.commit(budget=budget, eos=eos, decodable=drafter.decodable,
                            predictor=predictor, runtime_id=runtime_id)
    count = len(result["path"])
    positions = (context + torch.arange(count, device=field.device))[None]
    drafter.observe_rows(field, torch.tensor([slot], device=field.device), positions, result["features"],
                         torch.tensor([count], device=field.device))
    result.update(proposal_mass=selection.proposal_mass, predicted_expert_bytes=selection.predicted_expert_bytes,
                  prediction_coverage=selection.prediction_coverage, tree_nodes=len(selection.tree.tokens),
                  tree_tokens=selection.tree.tokens, tree_parents=selection.tree.parents)
    return result


class Verification:
    """Owned verify/commit transaction; abort leaves canonical tensor contents intact."""
    def __init__(self, net, caches, tree: Tree, *, seq, slot, context,
                 max_scratch_bytes=256 << 20, temperature=0., persistent_mlp=None):
        if temperature != 0:
            raise ValueError("tree verification currently supports greedy decoding only")
        if getattr(caches, "_tree_pending", None) is not None:
            raise RuntimeError("a tree verification already owns these caches")
        if any(type(v) is not int or v < 0 for v in (seq, slot, context)) or slot == 0:
            raise ValueError("tree decode needs a real sequence, slot and context")
        F = net.F
        if (len(tree.tokens) > 32 or max(tree.depths) > F.spec_k or any(t >= F.vocab for t in tree.tokens)
                or getattr(F, "kda_state_dtype", "fp32") != "fp32"):
            raise ValueError("tree exceeds the W4A8 row/draft/vocabulary bound or requires non-FP32 state")
        self.net, self.caches, self.tree = net, caches, tree
        self.seq, self.slot, self.context = seq, slot, context
        self.persistent_mlp = persistent_mlp
        # Conservative peak: factors, initial/commit/boundary states, raw conv,
        # node activations and one DSA prefix-key/selection workspace. CUDA
        # addresses canonical/private KV directly; the CPU numerical oracle
        # still materializes rows when computing its reference attention.
        n, h, d = len(tree.tokens), F.kda_heads_local, F.kda_dim
        nk = sum(not F.is_dsa(L) for L in net.layers)
        nd = len(net.layers) - nk
        from engine.modules.sparse_attention import mla_sparse_mqa
        gathered_reference = caches.device.type != 'cuda' or net.lanes.mla_sparse is mla_sparse_mqa
        self.scratch_bound = (nk * (3*h*d*d*4 + n*h*d*(12 + 6))
            + nd*n*(F.kv_lora + 5*F.idx_dim + 4) + 16*n*F.hidden*F.hc
            + sum(F.is_moe(L) for L in net.layers)*n*F.topk_experts*8
            + (context // F.kpool + n) * (F.idx_dim + 2*n + 16) * 4
            + n * (F.topk + F.kpool) * (F.kv_lora * 12 + 32 if gathered_reference else 32))
        if type(max_scratch_bytes) is not int or self.scratch_bound > max_scratch_bytes:
            raise ValueError(f"tree scratch needs at most {self.scratch_bound} bytes; budget is {max_scratch_bytes}")
        if persistent_mlp is not None:
            persistent_mlp.validate(net, n)
            self.scratch_bound += max(plan.scratch_bytes for plan in persistent_mlp.plans.values())
            if self.scratch_bound > max_scratch_bytes:
                raise ValueError("tree and persistent MLP exceed the combined scratch budget")
        # Only mappings are prepared here; reservation belongs to the caller.
        dummy = torch.full((max(tree.depths) + 1,), tree.tokens[0], dtype=torch.int64, device=caches.device)
        caches.prepare(Step.prefill(dummy, context, seq, slot))
        for L in net.layers:
            if not F.is_dsa(L) and caches.kda(L, slot)[1].dtype != torch.float32:
                raise ValueError("tree KDA canonical storage must be FP32")
        descriptor = (tree, seq, slot, context)
        if any(other != descriptor for other in net.comm.gather_objects(descriptor)):
            raise ValueError("ranks disagree on the tree/sequence/context")
        self.epoch = caches.pool.epochs[seq]
        self.versions = caches.state._version, caches.paged._version
        self.kda, self.dsa, self.routes = {}, {}, {}
        self.topology = tree_kda.Topology(tree, caches.device, taps=F.conv)
        # Branch pool and latent lookup geometry is shared by every DSA layer.
        paths = [tree.path(i) for i in range(n)]
        self.path_nodes = torch.tensor([p+(p[-1],)*(max(tree.depths)+1-len(p)) for p in paths],
                                       device=caches.device)
        self.lengths = torch.tensor([context+len(p) for p in paths], dtype=torch.int32, device=caches.device)
        complete = [i for i, p in enumerate(paths) if (context+len(p)) % F.kpool == 0]
        pool_index = {node: j for j, node in enumerate(complete)}
        sources = [tuple(paths[i][j] if j >= 0 else j for j in range(len(paths[i])-F.kpool, len(paths[i])))
                   for i in complete]
        self.pool_sources = torch.tensor(sources, device=caches.device, dtype=torch.int64).reshape(-1, F.kpool)
        self.pool_width = (context+max(tree.depths)+1)//F.kpool-context//F.kpool
        branch_ids = [tuple(pool_index[i] for i in p if i in pool_index) for p in paths]
        self.branch_pools = torch.tensor([p+(0,)*(self.pool_width-len(p)) for p in branch_ids],
                                         device=caches.device, dtype=torch.int64).reshape(n, self.pool_width)
        counts = [(context+len(p))//F.kpool for p in paths]
        self.pool_groups = tuple((count, torch.tensor([i for i, c in enumerate(counts) if c == count], device=caches.device))
                                 for count in sorted(set(counts)) if count)
        self.branch_columns = self.branch_pools+context//F.kpool
        self.indexer_starts = torch.zeros(n, dtype=torch.int32, device=caches.device)
        self.indexer_ends = torch.full((n,), context//F.kpool+len(complete), dtype=torch.int32, device=caches.device)
        self.hidden = self.features = None
        self.closed = False
        caches._tree_pending = self

    def _check(self):
        c = self.caches
        if (self.closed or c._tree_pending is not self or c.pool.epochs[self.seq] != self.epoch
                or c.slots.owner[self.slot] != self.seq
                or (c.state._version, c.paged._version) != self.versions):
            raise RuntimeError("tree verification is closed or its canonical cache changed")

    def abort(self):
        if getattr(self.caches, "_tree_pending", None) is self:
            self.caches._tree_pending = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.abort()

    def _kda(self, L, x):
        net, F, c = self.net, self.net.F, self.caches
        name, p, n, h, d = f"L{L}.kda.", net.p, len(x), net.Hk, F.kda_dim
        raw, beta, fa, ga = net.linear(x, name + "in_proj").split([3*h*d, h, d, d], -1)
        g = net.linear(fa, name + "f_b").view(n, h, d)
        gate = net.linear(ga, name + "g_b").view(n, h, d)
        conv_ring, rec = c.kda(L, self.slot)
        y = tree_kda.conv(self.tree, raw, p[name + "conv"], conv_ring, topology=self.topology, context=self.context)
        q, k, v = (value.view(n, h, d) for value in y.split(h*d, -1))
        initial = rec[(self.context-1) % net.rec_ring] if self.context else torch.zeros_like(rec[0])
        core, factors = tree_kda.verify(self.tree, q, k, v, g, beta, p[name + "A_log"],
                                       p[name + "dt_bias"], initial, F.lower_bound, topology=self.topology)
        self.kda[L] = raw, factors
        output = net.lanes.kda_output_norm(core, gate, p[name + "o_norm"], O_NORM_EPS)
        return net.comm.all_reduce(net.linear(output.reshape(n, h*d), name + "o_proj"))

    def _dsa(self, L, x):
        net, F, c, ctx = self.net, self.net.F, self.caches, self.context
        p, name, idx, n = net.p, f"L{L}.mla.", f"L{L}.idx.", len(x)
        qa, kv = net.linear(x, name + "qkv_a").split([F.q_lora, F.kv_lora], -1)
        qr = net._norm(qa, p[name + "q_a_norm"], F.rms_eps)
        q = net.linear(qr, name + "q_b").view(n, net.Hl, F.qk_nope)
        latent = net._norm(kv, p[name + "kv_a_norm"], F.rms_eps).to(torch.float8_e4m3fn)
        iq = net.linear(qr, idx + "wq_b").view(n, F.idx_heads, F.idx_dim)
        w = x.float() @ p[idx + "w_heads"].float().T
        key = net.linear(x, idx + "wk")
        key = (Fn.layer_norm(key.float(), (F.idx_dim,), p[idx + "k_norm_w"], p[idx + "k_norm_b"], K_NORM_EPS).to(x.dtype)
               if net.lanes.layernorm is None else net.lanes.layernorm(key, p[idx + "k_norm_w"], p[idx + "k_norm_b"], K_NORM_EPS))
        gate = net.linear(x, idx + "gate")
        iq8, qs = net.lanes.indexer_quant(iq.reshape(-1, F.idx_dim))
        iq8 = iq8.view(n, F.idx_heads, F.idx_dim)
        we = net.lanes.head_gate(w, qs.view(n, F.idx_heads), F.idx_scale)
        wb = p[name + "kv_b"].view(net.Hl, F.qk_nope + F.v_dim, F.kv_lora)
        qabs = tree_attention.absorb(q, wb[:, :F.qk_nope])
        width = F.topk + F.kpool - 1
        pk = torch.empty((0, F.idx_dim), dtype=torch.float8_e4m3fn, device=x.device)
        ps = torch.empty(0, dtype=torch.float32, device=x.device)
        if len(self.pool_sources):
            source = self.pool_sources
            tail = c.tail(L, self.slot)
            prefix = tail[(ctx+source) % len(tail)]
            kw = torch.where((source >= 0)[..., None], key[source.clamp_min(0)], prefix[:, :, 0])
            gw = torch.where((source >= 0)[..., None], gate[source.clamp_min(0)], prefix[:, :, 1])
            pk, ps = net.lanes.kpool_compress(kw, gw, p[idx + "ape"])
            ps = ps.reshape(-1)
        keys, scales = tree_attention.key_bank(c.pool_keys(L), c.pool_scales(L), pk, ps,
                                               *c.pool_map(L, self.seq), ctx//F.kpool)
        if len(keys):
            # Score the shared prefix once as a query batch. The extra bank
            # contains each completed branch pool once, never whole prefixes.
            logits = net.lanes.indexer_logits(iq8, keys, scales, we, self.indexer_ends, ks=self.indexer_starts).float()
            # Restore chronological columns before topk so siblings are absent
            # and ties have the same column order as the ordinary linear path.
            logical = torch.cat([logits[:, :ctx//F.kpool],
                logits.gather(1, self.branch_columns)], 1) if self.pool_width else logits[:, :ctx//F.kpool]
            pools = torch.full((n, F.topk//F.kpool), -1, dtype=torch.int32, device=x.device)
            # Topk ties can depend on the physical column count even when all
            # extra columns are -inf. Group equal lengths and slice exactly as
            # the linear decoder does (at most ceil(depth/kpool)+1 groups).
            for count, rows in self.pool_groups:
                take = min(count, F.topk//F.kpool)
                # Every sliced column is valid: retain even infinite logits,
                # as the ordinary selector does with its explicit valid count.
                pools[rows, :take] = logical[rows, :count].topk(take, dim=-1, sorted=False).indices.to(torch.int32)
        else:
            pools = torch.full((n, F.topk//F.kpool), -1, dtype=torch.int32, device=x.device)
        slots = torch.empty((n, width), dtype=torch.int32, device=x.device)
        valid = torch.empty(n, dtype=torch.int32, device=x.device)
        tree_attention.pool_slots(pools, self.lengths, F.kpool, *c.token_map(L, self.seq),
                                  self.path_nodes, ctx, slots, valid)
        output = net.lanes.mla_sparse(qabs, c.latent(L), slots, valid, F.mla_scale, 1., branch=latent)
        self.dsa[L] = latent, key, gate, pk, ps
        out = tree_attention.absorb(output, wb[:, F.qk_nope:], transpose=True)
        return net.comm.all_reduce(net.linear(out.reshape(n, net.Hl*F.v_dim), name + "o_proj"))

    @torch.inference_mode()
    def verify(self, *, aux_layers=()):
        self._check()
        if self.hidden is not None or any(L not in self.net.layers for L in aux_layers):
            raise ValueError("verify once, with auxiliary layers belonging to this target")
        net, F, n = self.net, self.net.F, len(self.tree.tokens)
        x = net.embed(torch.tensor(self.tree.tokens, dtype=torch.int64, device=self.caches.device))
        res = x[:, None, :].expand(n, F.hc, F.hidden).contiguous()
        post = comb = None
        aux = {}
        def observe(layer, routes):
            self.routes[layer] = routes.detach().clone()
        try:
            for L in net.layers:
                if post is None:
                    post, comb, x = net._hc_pre(L, res, "attn")
                else:
                    res, post, comb, x = net._hc_post_pre(L, x, res, post, comb, "attn")
                x = self._dsa(L, x) if F.is_dsa(L) else self._kda(L, x)
                res, post, comb, x = net._hc_post_pre(L, x, res, post, comb, "ffn")
                if F.is_moe(L):
                    x = net._moe(L, x, route_observer=observe)
                elif self.persistent_mlp is not None:
                    x = self.persistent_mlp(net, L, x)
                else:
                    x = net._dense(L, x)
                if L in aux_layers:
                    aux[L] = net.lanes.mhc_post(x, res, post, comb).float().mean(1).to(x.dtype)
            hidden = net.lanes.mhc_post(x, res, post, comb).float().mean(1).to(x.dtype)
            self.hidden = net._norm(hidden, net.p["norm"], F.rms_eps)
            self.features = torch.cat([aux[L] for L in aux_layers], -1) if aux_layers else None
            self._check()
            return self.hidden
        except Exception:
            self.abort()
            raise

    @torch.inference_mode()
    def commit(self, *, budget, eos=frozenset(), decodable=None, predictor=None, runtime_id=""):
        """Commit only verified inputs; the returned last token is the next anchor."""
        self._check()
        if self.hidden is None:
            raise RuntimeError("verify before commit")
        if type(budget) is not int or budget <= 0:
            raise ValueError("commit requires a positive output budget")
        if predictor is not None and predictor.runtime_id != runtime_id:
            raise ValueError("route observations belong to a different runtime")
        net, F, c = self.net, self.net.F, self.caches
        targets = net.head_tokens(self.hidden, F.vocab if decodable is None else decodable).tolist()
        result = self.tree.greedy(targets, budget=budget, eos=eos)
        # Rank zero is authoritative even if a future head implementation ties differently.
        outputs, path = net.comm.broadcast_object(result)
        positions = self.context + torch.arange(len(path), device=c.device)
        writes = []
        # Materialize everything before the first canonical write. Prefix-boundary
        # cells are retained so the existing checkpoint/stage owner can consume them.
        for L, (raw, factors) in self.kda.items():
            states = [(self.context+i, factors.state(node)) for i, node in enumerate(path)
                      if i == len(path)-1 or (self.context+i+1) % F.block == 0]
            writes.append(("kda", L, raw[list(path)], states))
        new_pools = (self.context+len(path))//F.kpool-self.context//F.kpool
        pool_rows = self.branch_pools[path[-1], :new_pools]
        for L, (latent, key, gate, branch_keys, branch_scales) in self.dsa.items():
            pk, ps = branch_keys[pool_rows], branch_scales[pool_rows]
            pids = c.pool_slots(L, self.seq, self.context // F.kpool + torch.arange(len(pk), device=c.device)).long()
            writes.append(("dsa", L, latent[list(path)], key[list(path)], gate[list(path)], pk, ps, pids))
        features = self.features[list(path)] if self.features is not None else None
        # One host transfer for all layers; per-layer .cpu() introduced a
        # stream synchronization at every routed layer in the eager path.
        observed = (dict(zip(self.routes, torch.stack(tuple(self.routes.values())).cpu().tolist()))
                    if self.routes else {})
        route_uses = sum(len(row) for rows in observed.values() for row in rows)
        unique = len({(L, e) for L, rows in observed.items() for row in rows for e in row})
        predicted_nodes = predicted_uses = actual_known = hits = 0
        if predictor is not None:
            # Score predictions before adding this step's labels. Unknown
            # nodes are reported as missing coverage, not perfect recall.
            for node, token in enumerate(self.tree.tokens):
                parent = self.tree.parents[node]
                predecessor = self.tree.tokens[parent] if parent >= 0 else token
                predicted = predictor.predict(predecessor, token, self.tree.depths[node], runtime_id=runtime_id)
                if predicted:
                    actual = {(L, e) for L, rows in observed.items() for e in rows[node]}
                    predicted_nodes += 1
                    predicted_uses += len(predicted)
                    actual_known += len(actual)
                    hits += len(predicted & actual)
            # Validate/update bounded metadata before publishing any cache write.
            for node in path:
                parent = self.tree.parents[node]
                predecessor = self.tree.tokens[parent] if parent >= 0 else self.tree.tokens[0]
                routes = [(L, e) for L, rows in observed.items() for e in rows[node]]
                if routes:
                    predictor.observe(predecessor, self.tree.tokens[node], self.tree.depths[node], routes,
                                      runtime_id=runtime_id)
        self._check()
        for kind, L, *values in writes:
            if kind == "kda":
                raw, states = values
                conv, rec = c.kda(L, self.slot)
                conv[:, positions % net.conv_ring] = raw.T
                for position, state in states:
                    rec[position % net.rec_ring].copy_(state)
            else:
                latent, key, gate, pk, ps, pids = values
                c.latent(L)[c.token_slots(L, self.seq, positions).long()] = latent
                c.pool_keys(L)[pids], c.pool_scales(L)[pids] = pk, ps
                tail = c.tail(L, self.slot)
                tail[positions % len(tail), 0], tail[positions % len(tail), 1] = key, gate
        self.abort()
        return {"tokens": outputs, "path": path, "context": self.context + len(path),
                "features": features, "factor_bytes": sum(f.nbytes for _, f in self.kda.values()),
                "scratch_bound_bytes": self.scratch_bound, "actual_route_uses": route_uses,
                "actual_unique_experts": unique, "route_prediction_nodes": predicted_nodes,
                "route_prediction_recall": hits/actual_known if actual_known else None,
                "route_prediction_precision": hits/predicted_uses if predicted_uses else None,
                "mode": "experimental-eager-greedy-tree"}
