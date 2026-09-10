"""Reading a pre-sharded rank file. The loader half of tools/dsv41_preshard.py.

The builder writes one safetensors per rank holding only that rank's tensors,
with expert names left at their GLOBAL ids -- `layers.5.ffn.experts.100.w1` is
still 100 in rank 1's file, not 4. That is deliberate: a renumbered file is
indistinguishable from a correct one once written, so the renumbering is done
HERE, where the rank is known and the mapping can be checked.

Which means this module has exactly two jobs, and both are refusals:

  1. refuse a file that is not this rank's. A pre-shard carries no shape that
     distinguishes it -- rank1of4 and rank0of4 have identical dtypes, identical
     shapes and identical sizes -- so the only evidence is `__metadata__`, and
     a file without it (or copied under another name) has to be rejected rather
     than trusted. Loading rank 1's experts as rank 0's produces a model that
     runs at full speed and answers wrong.
  2. refuse an expert this rank does not own, rather than renumber it into
     range. `local_expert` is the inverse of `expert_rank` only inside the
     rank's block; outside it, arithmetic still returns an integer.

Nothing here loads a tensor. `iter_weights` streams (name, tensor) pairs in the
shape `load_weights` wants, and the checking above happens before the first one
is yielded -- a loader that discovers the mismatch halfway has already written
into half the model.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from dsv41_layers import expert_rank

RANK_FILE = re.compile(r"^rank(\d+)of(\d+)\.safetensors$")
_EXPERT = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(.+)$")


class PreshardMismatch(RuntimeError):
    """The file on disk is not the one this rank asked for."""


@dataclass(frozen=True)
class Layout:
    """What a rank file says it is. `stated` is False when it says nothing."""

    rank: int
    world_size: int
    dense: str
    mtp: str
    n_routed_experts: int
    dspark_experts: int
    stated: bool = True

    def describe(self) -> str:
        src = "__metadata__" if self.stated else "the FILENAME only"
        return (f"rank {self.rank}/{self.world_size}, dense={self.dense}, "
                f"mtp={self.mtp}, {self.n_routed_experts} experts/layer "
                f"(from {src})")


def read_header(path: Path) -> tuple[dict, int]:
    """The safetensors header and the offset its data section starts at."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def read_layout(path: Path, header: "dict | None" = None) -> Layout:
    """What the file claims. Falls back to the filename, and says so.

    A filename-only layout is not an error here -- files built before the
    metadata existed are still correct -- but it IS weaker evidence, and
    `require` is what decides whether that is good enough.
    """
    if header is None:
        header, _ = read_header(path)
    meta = header.get("__metadata__")
    m = RANK_FILE.match(path.name)
    if meta and "rank" in meta and "world_size" in meta:
        return Layout(rank=int(meta["rank"]),
                      world_size=int(meta["world_size"]),
                      dense=meta.get("dense", "replicate"),
                      mtp=meta.get("mtp", "replicate"),
                      n_routed_experts=int(meta.get("n_routed_experts", 0)),
                      dspark_experts=int(meta.get("dspark_experts", 0)))
    if not m:
        raise PreshardMismatch(
            f"{path.name} has no __metadata__ and its name does not say which "
            f"rank it holds. There is nothing left to check it against.")
    return Layout(rank=int(m.group(1)), world_size=int(m.group(2)),
                  dense="replicate", mtp="replicate", n_routed_experts=0,
                  dspark_experts=0, stated=False)


def require(layout: Layout, *, rank: int, world_size: int,
            n_routed_experts: int = 0, dense: str = "replicate",
            mtp: str = "replicate", allow_unstated: bool = False) -> None:
    """Raise unless the file is the one this rank should be reading."""
    if not layout.stated and not allow_unstated:
        raise PreshardMismatch(
            f"the rank file carries no __metadata__, so the only claim it "
            f"makes about its contents is its name. Rebuild it with the "
            f"current tools/dsv41_preshard.py, or pass allow_unstated=True to "
            f"accept the filename as evidence.")
    bad = []
    if layout.world_size != world_size:
        bad.append(f"world size {layout.world_size}, this job runs "
                   f"{world_size}")
    if layout.rank != rank:
        bad.append(f"rank {layout.rank}, this process is rank {rank}")
    if layout.dense != dense:
        bad.append(f"dense={layout.dense}, expected {dense}")
    if layout.mtp != mtp:
        bad.append(f"mtp={layout.mtp}, expected {mtp}")
    if (layout.stated and n_routed_experts
            and layout.n_routed_experts != n_routed_experts):
        bad.append(f"{layout.n_routed_experts} experts/layer, config says "
                   f"{n_routed_experts}")
    if bad:
        raise PreshardMismatch(
            "the pre-shard on disk is not this rank's: "
            + "; ".join(bad)
            + ". Every rank file has the same shapes and the same size, so "
              "nothing downstream would have noticed.")


def local_expert(expert: int, n_routed_experts: int, world_size: int,
                 rank: int) -> int:
    """Global expert id -> this rank's index for it.

    The inverse of `expert_rank`, and only within the rank's own block. Out of
    it the arithmetic still yields an integer -- expert 0 on rank 1 of 4 with
    384 experts gives -96 -- which indexes a parameter list from the end and
    loads a real tensor into the wrong slot.
    """
    owner = expert_rank(expert, n_routed_experts, world_size)
    if owner != rank:
        raise PreshardMismatch(
            f"expert {expert} belongs to rank {owner}, not rank {rank}. A "
            f"pre-shard that contains it was built for another rank.")
    return expert - rank * (n_routed_experts // world_size)


def localize(name: str, *, rank: int, world_size: int, n_routed_experts: int,
             dspark_experts: int = 0, mtp: str = "replicate") -> str:
    """Rewrite one checkpoint name for this rank. Non-expert names pass through.

    A REPLICATED expert is not renumbered. Under `--mtp replicate` every rank
    holds all 128 of the DSpark block's experts, so its local index IS its
    global one -- renumbering there would map expert 32 onto slot -64 and
    refuse a name that is legitimately present.
    """
    m = _EXPERT.match(name)
    if not m:
        return name
    prefix, layer, expert, tail = m.groups()
    if prefix == "mtp":
        if mtp == "replicate":
            return name
        if mtp != "ep":
            raise ValueError(f"mtp must be 'replicate' or 'ep', not {mtp!r}")
    total = dspark_experts if prefix == "mtp" else n_routed_experts
    if not total:
        raise PreshardMismatch(
            f"{name} is an expert tensor but the expert count for {prefix!r} "
            f"is 0; nothing can be renumbered without it.")
    return (f"{prefix}.{layer}.ffn.experts."
            f"{local_expert(int(expert), total, world_size, rank)}.{tail}")


def find(model_dir, rank: int, world_size: int) -> "Path | None":
    """The rank file, or None if this is a stock repo rather than a pre-shard.

    None is not a failure: it is how a caller learns to fall back to the
    48-shard checkpoint. A rank file that is MISSING while its siblings exist
    is a failure, and is reported as one.
    """
    d = Path(model_dir)
    mine = d / f"rank{rank}of{world_size}.safetensors"
    if mine.is_file():
        return mine
    siblings = sorted(p.name for p in d.glob("rank*of*.safetensors"))
    if siblings:
        raise PreshardMismatch(
            f"{d} holds {siblings} but not {mine.name}. Pre-sharded weights "
            f"are per rank: this rank has nothing to load, and falling back to "
            f"the stock checkpoint would need 475 GiB on a node that does not "
            f"have it.")
    return None


def iter_weights(path, *, rank: int, world_size: int, n_routed_experts: int,
                 dspark_experts: int = 0, dense: str = "replicate",
                 mtp: str = "replicate", allow_unstated: bool = False,
                 device: str = "cpu"):
    """Stream (local_name, tensor) for `load_weights`, checked before the first.

    The check runs to completion over the whole NAME set first. Discovering a
    foreign expert on the last tensor of a 85 GiB file is discovering it after
    the model has been populated from that file.
    """
    from safetensors import safe_open

    path = Path(path)
    header, _ = read_header(path)
    header.pop("__metadata__", None)
    layout = read_layout(path)
    require(layout, rank=rank, world_size=world_size,
            n_routed_experts=n_routed_experts, dense=dense, mtp=mtp,
            allow_unstated=allow_unstated)
    names = sorted(header)
    renamed = {n: localize(n, rank=rank, world_size=world_size,
                           n_routed_experts=n_routed_experts,
                           dspark_experts=dspark_experts, mtp=mtp)
               for n in names}
    if len(set(renamed.values())) != len(renamed):
        dupes = [n for n in renamed
                 if list(renamed.values()).count(renamed[n]) > 1]
        raise PreshardMismatch(
            f"renumbering collapsed {len(dupes)} names onto the same "
            f"destination, e.g. {dupes[:2]}")
    with safe_open(str(path), framework="pt", device=device) as fh:
        for name in names:
            yield renamed[name], fh.get_tensor(name)
