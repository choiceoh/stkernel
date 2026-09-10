"""Where each checkpoint tensor goes. Routing only -- no tensors are moved.

`load_weights` in a vLLM model consumes a stream of (name, tensor). This is the
part of it that decides, for one name, which module and which parameter it
belongs to and whether this rank wants it at all. Keeping that separable is what
makes it checkable: probes/dsv41_loader_route.py feeds it the 96,085 names from
the index and requires every one to be claimed exactly once, and requires the
set a rank claims to equal what tools/dsv41_preshard.py writes for that rank.

That second requirement is the point. The builder and the loader never run
together -- one writes files on srv4 today, the other reads them inside a
container weeks later -- so the only thing keeping them agreed is a test that
holds them to the same partition. They already share `expert_rank`; this adds
the name-level check on top.

Routing is by PATTERN, not by walking the shape plan. Walking the plan would
make coverage true by construction and prove nothing; matching names means a
tensor the plan never predicted arrives unclaimed and says so.

Three destinations:

  PARAM      a module parameter. `scale` rides with its `weight` -- they are
             one quantized tensor in two pieces and a loader that treats the
             scale as an independent parameter loads it into a module that has
             no such attribute.
  ENGRAM     not a parameter at all. The two lookup tables are 188.8 GiB and
             live on the SSD; dsv41_engram reads rows from them. A loader that
             materializes them is a loader that OOMs, which is the whole
             reason this profile exists.
  SKIP       nothing in the checkpoint routes here today. It exists so that an
             unrecognized name is a refusal rather than a silent drop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PARAM = "param"
ENGRAM = "engram"

# A whitelist of shapes a name may take. They carry no destination: the split
# is uniform (last segment is the attribute), so what these are FOR is refusing
# a name nobody has thought about.
_RULES = [
    # engram: the two tables leave, the small projections stay
    (re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$"), ENGRAM, None),
    (re.compile(r"^layers\.(\d+)\.engram\.(q_weight|k_weight)$"),
     None, None),
    (re.compile(r"^layers\.(\d+)\.engram\.wkv\.(weight|scale)$"),
     None, None),
    # routed experts, the only rank-local tensors
    (re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"),
     None, None),
    # everything else in a layer or an mtp head
    (re.compile(r"^(layers|mtp)\.(\d+)\.(.+)\.(weight|scale|bias|bias_vl)$"),
     None, None),
    (re.compile(r"^(layers|mtp)\.(\d+)\.(hc_\w+|attn\.attn_sink)$"),
     None, None),
    # the tower, the aligner, and the top level
    (re.compile(r"^(vision|aligner)\.(.+)\.(weight|bias)$"), None, None),
    (re.compile(r"^(embed|head|norm)\.weight$"), None, None),
    (re.compile(r"^(image_start|image_end|image_newline)$"), None, None),
]

_EXPERT = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.")


@dataclass(frozen=True)
class Route:
    """`param` is None for a bare tensor whose name IS the attribute."""

    kind: str
    module: str
    param: "str | None"
    expert: "int | None" = None


class UnroutedTensor(KeyError):
    """A name no rule claims. Never a warning: a dropped weight is silent."""


def route(name: str) -> Route:
    for pattern, template, _ in _RULES:
        m = pattern.match(name)
        if not m:
            continue
        if template is ENGRAM:
            head, _, tail = name.rpartition(".")
            return Route(ENGRAM, head, tail)
        # The patterns are a whitelist -- their job is to make an unknown name
        # fail rather than be split blindly -- but the split itself is uniform:
        # the last segment is the attribute and everything before it is the
        # module. Treating `hc_attn_fn` and friends as attribute-less collapses
        # every one of a layer's bare tensors onto one destination, which loads
        # whichever arrived last and drops the rest.
        module, _, param = name.rpartition(".")
        e = _EXPERT.match(name)
        return Route(PARAM, module, param, int(e.group(3)) if e else None)
    raise UnroutedTensor(
        f"no rule claims {name!r}. A loader that skipped it would leave a "
        f"module at its initialization values and say nothing.")


def wanted_by(name: str, rank: int, world_size: int,
              n_routed_experts: int, dspark_experts: int,
              mtp: str = "replicate") -> bool:
    """Does this rank load this tensor?

    Only routed experts are rank-local; everything else is replicated, which is
    what tools/dsv41_preshard.py writes and why the two must agree. The engram
    tables are wanted by no rank: they are not loaded at all.

    `mtp` is the DSpark block, and it is a MODE rather than a fact. This used
    to shard its 128 experts unconditionally while the builder replicated them,
    a disagreement of 1,728 tensors per rank -- and it went unnoticed because
    the probe that compares the two excluded `mtp.` from the comparison. The
    default here is `replicate` because that is what the builder writes and
    what this fleet's draft_tensor_parallel_size=1 means; `ep` exists so that
    a build made with --mtp ep can be read by a loader told the same thing,
    and neither side may pick on its own.
    """
    from dsv41_layers import expert_rank

    r = route(name)
    if r.kind == ENGRAM:
        return False
    if r.expert is None:
        return True
    if name.startswith("mtp."):
        if mtp == "replicate":
            return True
        if mtp != "ep":
            raise ValueError(f"mtp must be 'replicate' or 'ep', not {mtp!r}")
        return expert_rank(r.expert, dspark_experts, world_size) == rank
    return expert_rank(r.expert, n_routed_experts, world_size) == rank
