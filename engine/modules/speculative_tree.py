"""Bounded, ancestor-closed draft trees. Predictions change proposals, never the target.

This is an eager experiment: small candidate metadata crosses to the host once.
Greedy verification is exact relative to the target's argmax; sampled decoding
must keep the existing rejection sampler until it has a separate tree proof.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Tree:
    tokens: tuple[int, ...]
    parents: tuple[int, ...]

    def __post_init__(self):
        if (not isinstance(self.tokens, tuple) or not isinstance(self.parents, tuple)
                or not 1 <= len(self.tokens) <= 64 or len(self.tokens) != len(self.parents)):
            raise ValueError("a tree needs 1..64 tokens and one parent per token")
        seen = set()
        for i, (token, parent) in enumerate(zip(self.tokens, self.parents)):
            if type(token) is not int or token < 0 or type(parent) is not int:
                raise ValueError("tree token and parent IDs must be integers")
            if (i == 0 and parent != -1) or (i > 0 and not 0 <= parent < i):
                raise ValueError("one root and topologically ordered parents are required")
            if (parent, token) in seen:
                raise ValueError("duplicate siblings would double-count proposal mass")
            seen.add((parent, token))

    def path(self, node: int) -> tuple[int, ...]:
        if type(node) is not int or not 0 <= node < len(self.tokens):
            raise ValueError("node is outside this tree")
        path = []
        while node >= 0:
            path.append(node)
            node = self.parents[node]
        return tuple(reversed(path))

    @property
    def depths(self):
        return tuple(len(self.path(i)) - 1 for i in range(len(self.tokens)))

    def greedy(self, target_tokens, *, budget: int, eos: frozenset[int] = frozenset()):
        """Emit target tokens; return the input-node path whose states to commit.

        Root is the already emitted anchor, at the next uncomputed position.
        Every iteration computes one new output from one verified input node.
        The last output (including EOS) is not committed as an input yet.
        """
        if type(budget) is not int or budget <= 0 or len(target_tokens) != len(self.tokens):
            raise ValueError("greedy verification needs a positive output budget and one target per node")
        if any(type(t) is not int or t < 0 for t in target_tokens):
            raise ValueError("target argmax tokens must be nonnegative integers")
        children = {(p, t): i for i, (p, t) in enumerate(zip(self.parents, self.tokens)) if i}
        node, outputs, path = 0, [], []
        while len(outputs) < budget:
            path.append(node)
            token = target_tokens[node]
            outputs.append(token)
            if token in eos or (node, token) not in children:
                break
            node = children[node, token]
        return tuple(outputs), tuple(path)


@dataclass(frozen=True)
class Candidate:
    token: int
    parent: int
    probability: float                 # conditional proposal mass, not measured target acceptance
    experts: frozenset[tuple[int, int]] = frozenset()  # (layer, expert), never prune target routes


@dataclass(frozen=True)
class Selection:
    tree: Tree
    source_nodes: tuple[int, ...]
    proposal_mass: float
    predicted_expert_bytes: int
    prediction_coverage: float


def select(candidates: tuple[Candidate, ...], budget: int, *, bytes_per_expert: int,
           fixed_node_bytes: int, cost_weight: float = 1.0) -> Selection:
    """Greedy marginal proposal-mass / cost selection, with ancestor closure.

    The fixed positive charge accounts for attention, shared MLP and metadata.
    A node with unknown routes is charged the largest known route footprint,
    so missing predictions cannot masquerade as free experts. With no route
    observations the ranking reduces to proposal mass alone.
    """
    full = Tree(tuple(c.token for c in candidates), tuple(c.parent for c in candidates))
    if (type(budget) is not int or not 1 <= budget <= len(candidates)
            or type(bytes_per_expert) is not int or bytes_per_expert <= 0
            or type(fixed_node_bytes) is not int or fixed_node_bytes <= 0
            or not math.isfinite(cost_weight) or cost_weight < 0):
        raise ValueError("tree selection needs a valid budget and finite positive byte costs")
    mass = []
    child_mass = Counter()
    for i, c in enumerate(candidates):
        if not math.isfinite(c.probability) or not 0 <= c.probability <= 1:
            raise ValueError("candidate probabilities must lie in [0, 1]")
        if not isinstance(c.experts, frozenset) or any(
                not isinstance(e, tuple) or len(e) != 2 or any(type(v) is not int or v < 0 for v in e)
                for e in c.experts):
            raise ValueError("expert identities must be immutable (layer, expert) pairs")
        if i:
            child_mass[c.parent] += c.probability
        mass.append(c.probability * (mass[c.parent] if i else 1))
    if candidates[0].probability != 1 or any(v > 1 + 1e-6 for v in child_mass.values()):
        raise ValueError("root mass must be one and sibling mass cannot exceed one")
    chosen, touched = [0], set(candidates[0].experts)
    unknown = max((len(c.experts) for c in candidates), default=0)
    charged = (len(touched) if touched else unknown) * bytes_per_expert
    while len(chosen) < budget:
        eligible = [i for i, c in enumerate(candidates) if i not in chosen and c.parent in chosen]
        def cost(i):
            extra = len(candidates[i].experts - touched) if candidates[i].experts else unknown
            return extra * bytes_per_expert
        best = max(eligible, key=lambda i: (mass[i] / (fixed_node_bytes + cost_weight * cost(i)), mass[i], -i))
        charged += cost(best)
        touched.update(candidates[best].experts)
        chosen.append(best)
    chosen.sort()                       # kernels consume a topological order
    remap = {old: new for new, old in enumerate(chosen)}
    tree = Tree(tuple(full.tokens[i] for i in chosen),
                tuple(-1 if i == 0 else remap[full.parents[i]] for i in chosen))
    return Selection(tree, tuple(chosen), sum(mass[i] for i in chosen), charged,
                     sum(bool(candidates[i].experts) for i in chosen) / len(chosen))


class RouteTable:
    """Bounded, conservative empirical predictor keyed by predecessor/token/depth.

    It stores the union of observed target routes, not fabricated router labels.
    Poor coverage or route recall is visible in records; this deliberately small
    baseline can be replaced by a calibrated hidden-state head. A runtime ID is
    mandatory: observations from a different model/pack/selector are refused.
    """
    def __init__(self, runtime_id: str, *, capacity=256, min_observations=2, max_routes=512):
        if (not isinstance(runtime_id, str) or not runtime_id or type(capacity) is not int or capacity <= 0
                or type(min_observations) is not int or min_observations <= 0
                or type(max_routes) is not int or max_routes <= 0):
            raise ValueError("route table needs a runtime identity and positive bounds")
        self.runtime_id, self.capacity, self.min_observations = runtime_id, capacity, min_observations
        self.max_routes = max_routes
        self.entries = OrderedDict()

    def _key(self, predecessor, token, depth, runtime_id):
        if runtime_id != self.runtime_id:
            raise ValueError("route observations belong to a different runtime")
        if any(type(v) is not int or v < 0 for v in (predecessor, token, depth)):
            raise ValueError("route predictor keys must be nonnegative integers")
        return predecessor, token, depth

    def predict(self, predecessor, token, depth, *, runtime_id):
        key = self._key(predecessor, token, depth, runtime_id)
        count, routes = self.entries.get(key, (0, frozenset()))
        return routes if count >= self.min_observations else frozenset()

    def observe(self, predecessor, token, depth, experts, *, runtime_id):
        key = self._key(predecessor, token, depth, runtime_id)
        routes = frozenset(experts)
        if not routes or len(routes) > self.max_routes or any(
                not isinstance(e, tuple) or len(e) != 2 or any(type(v) is not int or v < 0 for v in e) for e in routes):
            raise ValueError("route observations need actual layer/expert identities")
        count, previous = self.entries.pop(key, (0, frozenset()))
        if len(previous | routes) > self.max_routes:
            # A diffuse/stale entry must warm up again instead of growing without bound.
            count, previous = 0, frozenset()
        self.entries[key] = (count + 1, previous | routes)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


def dflash_candidates(anchor, unary, tokens, projection, predecessor, successor, alpha, *,
                      width=2, max_nodes=31, predictor=None, runtime_id=""):
    """Expand the existing DFlash selector scores without a second draft model.

    Tensor math stays on the input device. Only the bounded candidate lattice
    is copied to CPU; this entry point is intentionally outside CUDA capture.
    Sibling probabilities are normalized over the full candidate support,
    including children subsequently pruned by width or node budget.
    """
    import torch
    if (unary.ndim != 2 or tokens.shape != unary.shape or projection.ndim != 2
            or projection.shape[0] != unary.shape[0] or len(alpha) != len(unary)
            or type(width) is not int or not 1 <= width <= unary.shape[1]
            or type(max_nodes) is not int or not 1 <= max_nodes <= 64
            or type(anchor) is not int or anchor < 0 or anchor >= len(predecessor)):
        raise ValueError("invalid DFlash tree candidate geometry")
    if (tokens.dtype != torch.int64 or not torch.isfinite(unary).all() or not torch.isfinite(projection).all()
            or any(not math.isfinite(a) for a in alpha) or predecessor.ndim != 2 or successor.shape != predecessor.shape
            or projection.shape[1] != predecessor.shape[1] or torch.any((tokens < 0) | (tokens >= len(predecessor)))):
        raise ValueError("candidate IDs must be int64 and selector scores finite")
    # Construct all predecessor/candidate edges once, rather than per tree node.
    prev = torch.cat([torch.full_like(tokens[:1], anchor), tokens[:-1]], 0)
    edge = torch.einsum("kpr,kcr->kpc", predecessor[prev].float() * projection[:, None, :].float(),
                        successor[tokens].float())
    scale = torch.tensor(alpha, device=unary.device).view(-1, 1, 1)
    probs = torch.softmax(unary[:, None, :].float() + edge * scale, -1).cpu()
    ids = tokens.cpu().tolist()
    if not torch.isfinite(probs).all() or any(len(set(row)) != len(row) for row in ids):
        raise ValueError("invalid or duplicate selector candidate support")
    root_routes = predictor.predict(anchor, anchor, 0, runtime_id=runtime_id) if predictor else frozenset()
    out = [Candidate(anchor, -1, 1., root_routes)]
    frontier = [(0, 0)]
    for depth in range(len(ids)):
        next_frontier = []
        for parent, previous_column in frontier:
            row = probs[depth, previous_column]
            for column in sorted(range(len(row)), key=lambda j: (-float(row[j]), j))[:width]:
                if len(out) == max_nodes:
                    return tuple(out)
                token = ids[depth][column]
                routes = (predictor.predict(out[parent].token, token, depth + 1, runtime_id=runtime_id)
                          if predictor else frozenset())
                next_frontier.append((len(out), column))
                out.append(Candidate(token, parent, float(row[column]), routes))
        frontier = next_frontier
    return tuple(out)
