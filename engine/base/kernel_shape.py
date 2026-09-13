"""The kernel shape descriptor (base): every model constant a kernel lane leans on, stated once by
the profile and read by the kernels -- so a lane is bound to a SHAPE, not to a model.

Until 2026-09-13 the kernel package spelled GLM-5.3's geometry in its own source: the MoE
admission gate compared against (288, 4096, 512, 8), the one-shot transport checked
`shape[1] != 4096`, the MK mHC wrapper sized its workspace for hc 4 x hidden 4096, the draft
kernels refused any head but 128. Those numbers are one model's, and a second profile (CHARTER
D5: Qwen3.8-Flash-Next) could not reach the same kernels without editing them. This module is
the seam: a profile derives a `KernelShape` from its facts, `bind()`s it at boot, and each lane
reads `bound()` instead of a literal.

Two rules make it safe (D3):

- **Binding is once per process.** A lane keys compiled kernels, weight views and captured graphs
  on the shape; rebinding a different one after that would misread bytes. `bind()` refuses a
  second, different shape. Equal rebinding is a no-op.
- **Unbound means the measured cell, never a guess.** `bound()` returns `MEASURED` -- the GB10
  TP4 cell the kernels were compiled and measured for (its numbers are the ones that used to be
  literals) -- when no profile has bound anything. A probe that runs a kernel standalone gets the
  kernel's own cell; a profile whose shape differs must bind, and a lane whose kernel is compiled
  for one cell (MLA's 16x512 latent, the Hadamard-128 indexer, the mHC kernel's hidden 4096/5120)
  refuses a bound shape it cannot serve, by name, instead of running it.

A `KernelShape` is per RANK: heads, intermediate and expert counts are what one of the `tp`
ranks holds, except where a field says otherwise (`moe.inter` is the model's, `moe.inter_local`
this rank's -- the dispatcher meets both spellings).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


def _positive(owner: str, **values) -> None:
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{owner}.{name} must be a positive int, got {value!r}")


def _power_of_two(owner: str, **values) -> None:
    for name, value in values.items():
        if type(value) is not int or value <= 0 or value & (value - 1):
            raise ValueError(f"{owner}.{name} must be a power of two, got {value!r}")


@dataclass(frozen=True)
class Device:
    """The box the lanes are compiled for. Every rank of this fleet is a GB10 (SM121, 48 SMs)."""
    capability: tuple = (12, 1)
    sms: int = 48

    def __post_init__(self):
        if len(self.capability) != 2 or any(type(v) is not int or v < 0 for v in self.capability):
            raise ValueError("Device.capability is (major, minor)")
        object.__setattr__(self, "capability", tuple(self.capability))
        _positive("Device", sms=self.sms)


@dataclass(frozen=True)
class Comm:
    """What the collectives carry: `world` ranks exchanging [rows, hidden] BF16 activations."""
    world: int
    hidden: int

    def __post_init__(self):
        _positive("Comm", world=self.world, hidden=self.hidden)


@dataclass(frozen=True)
class Attention:
    """The full-attention lane. `kind` "mla": `heads` query heads per rank over one shared latent of
    `head_dim` (kv_lora_rank); "gqa": `heads` query heads and `kv_heads` KV heads per rank at `head_dim`."""
    kind: str
    heads: int
    head_dim: int
    kv_heads: int = 1

    def __post_init__(self):
        if self.kind not in ("mla", "gqa"):
            raise ValueError(f"Attention.kind must be 'mla' or 'gqa', got {self.kind!r}")
        _positive("Attention", heads=self.heads, head_dim=self.head_dim, kv_heads=self.kv_heads)
        if self.heads % self.kv_heads:
            raise ValueError("Attention.heads must be a multiple of kv_heads")


@dataclass(frozen=True)
class LinearAttention:
    """The gated delta rule lane (KDA, GDN). Per rank: `heads` key heads, `v_heads` value heads (a
    multiple), key/value widths, the causal conv width, and where the decay lives -- "channel"
    (KDA: one log-decay per key channel) or "head" (GDN: one per head; modules/linear_attention)."""
    heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv: int
    decay: str = "channel"

    def __post_init__(self):
        _positive("LinearAttention", heads=self.heads, v_heads=self.v_heads, conv=self.conv)
        _power_of_two("LinearAttention", k_dim=self.k_dim, v_dim=self.v_dim)
        if self.v_heads % self.heads:
            raise ValueError("LinearAttention.v_heads must be a multiple of heads")
        if self.decay not in ("channel", "head"):
            raise ValueError(f"LinearAttention.decay must be 'channel' or 'head', got {self.decay!r}")


@dataclass(frozen=True)
class Indexer:
    """The sparse indexer: `heads` index heads (replicated), `head_dim` per head (the FP8 key
    width), `pool` tokens per compressed key, `topk` positions a query may attend to."""
    heads: int
    head_dim: int
    pool: int
    topk: int

    def __post_init__(self):
        _positive("Indexer", heads=self.heads, pool=self.pool, topk=self.topk)
        _power_of_two("Indexer", head_dim=self.head_dim)
        if self.topk % self.pool:
            raise ValueError("Indexer.topk must be whole pools")


@dataclass(frozen=True)
class MoE:
    """The routed-expert lane. `experts` in the model, `experts_local` on this rank (TP: all of
    them, sliced; EP: a slice of them, whole), `inter` the model's intermediate and `inter_local`
    this rank's, `topk` experts per token, the quantisation and gated activation the kernels are
    admitted for, the dense/shared MLP width served through the E=1 lane, and an optional measured
    pin for the prefill kernel's M tile (None: the row-count table)."""
    experts: int
    experts_local: int
    hidden: int
    inter: int
    inter_local: int
    topk: int
    quant: str
    activation: str
    swiglu_limit: "float | None"
    dense_inter_local: int = 0
    dynamic_tile_m: "int | None" = None

    def __post_init__(self):
        _positive("MoE", experts=self.experts, experts_local=self.experts_local, hidden=self.hidden,
                  inter=self.inter, inter_local=self.inter_local, topk=self.topk)
        if type(self.dense_inter_local) is not int or self.dense_inter_local < 0:
            raise ValueError("MoE.dense_inter_local must be a non-negative int")
        if self.experts_local > self.experts or self.inter_local > self.inter:
            raise ValueError("a rank cannot hold more experts or a wider intermediate than the model has")
        if self.dynamic_tile_m is not None and self.dynamic_tile_m not in (16, 32, 64, 128):
            raise ValueError("MoE.dynamic_tile_m must be 16/32/64/128 or None")


@dataclass(frozen=True)
class Drafter:
    """The speculative drafter's attention geometry (the draft kernels take `head_dim` from it)."""
    head_dim: int
    kv_heads: int
    layers: int
    window: int

    def __post_init__(self):
        _power_of_two("Drafter", head_dim=self.head_dim)
        _positive("Drafter", kv_heads=self.kv_heads, layers=self.layers, window=self.window)


@dataclass(frozen=True)
class KernelShape:
    comm: Comm
    hidden: int
    hc: int                     # residual streams of the hyper-connection ([T, hc, hidden])
    tp: int
    attention: Attention
    linear: LinearAttention
    indexer: Indexer
    moe: MoE
    spec_k: int                 # draft tokens verified per decode step (1 for an MTP head)
    device: Device = Device()
    drafter: "Drafter | None" = None

    def __post_init__(self):
        _positive("KernelShape", hidden=self.hidden, hc=self.hc, tp=self.tp, spec_k=self.spec_k)
        if self.comm.hidden != self.hidden or self.moe.hidden != self.hidden:
            raise ValueError("KernelShape: comm.hidden and moe.hidden must equal hidden")
        if self.comm.world != self.tp:
            raise ValueError("KernelShape: comm.world must equal tp")
        if self.moe.inter_local * self.tp != self.moe.inter and self.moe.inter_local != self.moe.inter:
            raise ValueError("KernelShape: moe.inter_local is inter/tp (TP-sharded) or inter (EP, whole experts)")

    def describe(self) -> str:
        m, a, l, i = self.moe, self.attention, self.linear, self.indexer
        return (f"hidden {self.hidden} hc {self.hc} tp {self.tp} | {a.kind} {a.heads}x{a.head_dim} | "
                f"linear {l.heads}/{l.v_heads}x{l.k_dim}x{l.v_dim} conv {l.conv} decay/{l.decay} | "
                f"indexer {i.heads}x{i.head_dim} pool {i.pool} top {i.topk} | "
                f"moe {m.experts}({m.experts_local} local) I{m.inter}/{m.inter_local} top{m.topk} {m.quant} {m.activation} | "
                f"spec {self.spec_k}" + (f" | drafter {self.drafter.head_dim}" if self.drafter else ""))


# The cell the kernels were compiled and measured for: GLM-5.3-Flash on four GB10s (TP=4). These
# are the numbers that were literals in the kernel package until 2026-09-13; a profile whose
# shape equals this one runs the same code it always did (tests/test_engine_kernel_shape pins
# the GLM profile's derivation to it).
MEASURED = KernelShape(
    comm=Comm(world=4, hidden=4096),
    hidden=4096, hc=4, tp=4,
    attention=Attention(kind="mla", heads=16, head_dim=512, kv_heads=1),
    linear=LinearAttention(heads=16, v_heads=16, k_dim=128, v_dim=128, conv=4, decay="channel"),
    indexer=Indexer(heads=32, head_dim=128, pool=4, topk=2048),
    moe=MoE(experts=288, experts_local=288, hidden=4096, inter=2048, inter_local=512, topk=8,
            quant="nvfp4", activation="swigluoai_uninterleave", swiglu_limit=10.0, dense_inter_local=3072),
    spec_k=6,
    device=Device(capability=(12, 1), sms=48),
    drafter=Drafter(head_dim=128, kv_heads=8, layers=5, window=2048),
)

_BOUND: "KernelShape | None" = None
_DRAFTER: "Drafter | None" = None


def bind(shape: KernelShape) -> KernelShape:
    """Declare the process's kernel shape. Once: a different shape after the first is refused."""
    global _BOUND
    if not isinstance(shape, KernelShape):
        raise TypeError(f"bind() takes a KernelShape, got {type(shape).__name__}")
    if _BOUND is not None and _BOUND != shape:
        raise RuntimeError("kernel shape already bound to a different shape in this process:\n"
                           f"  bound: {_BOUND.describe()}\n  asked: {shape.describe()}")
    _BOUND = shape
    return shape


def bound() -> KernelShape:
    """The bound shape, or the measured cell when no profile has bound one."""
    return MEASURED if _BOUND is None else _BOUND


def is_bound() -> bool:
    return _BOUND is not None


def bind_drafter(drafter: Drafter) -> Drafter:
    """Declare the drafter's geometry when the drafter loads (after the model's shape is known)."""
    global _DRAFTER
    if not isinstance(drafter, Drafter):
        raise TypeError(f"bind_drafter() takes a Drafter, got {type(drafter).__name__}")
    if _DRAFTER is not None and _DRAFTER != drafter:
        raise RuntimeError(f"drafter shape already bound to a different one: bound {_DRAFTER}, asked {drafter}")
    _DRAFTER = drafter
    return drafter


def drafter() -> Drafter:
    """The bound drafter geometry; the bound shape's own; else the measured cell's."""
    if _DRAFTER is not None:
        return _DRAFTER
    shape = bound()
    return shape.drafter if shape.drafter is not None else MEASURED.drafter


def reset() -> None:
    """Tests only: forget the process's bindings."""
    global _BOUND, _DRAFTER
    _BOUND = _DRAFTER = None


def fields_of(shape) -> dict:
    """A flat {name: value} view for logs and tables."""
    return {f.name: getattr(shape, f.name) for f in fields(shape)}


# ---- the record: the wizard writes it beside the rank files when a model is taken in; a boot reads it -----------
#
# Deriving the shape from config.json is free, so a boot could do it every time -- and did, until 2026-09-13.
# What a per-model record adds is not speed: it is the place for decisions the config cannot yield (the measured
# pins: moe.dynamic_tile_m, and the cells a new model gets measured into), and the admission table -- which lanes
# serve this model, which refuse it by name, which want a measurement -- produced when the model is taken in
# (preshard) instead of discovered by a dying boot. A boot binds the record when one is present and its config
# hash still matches the checkpoint (a stale record dies, D3: rerun the wizard), and derives as before when none is.

RECORD = "kernel_shape.json"
RECORD_VERSION = 1
PROFILES = {"glm53": "engine.profiles.glm53.facts", "qwen38": "engine.profiles.qwen38.shapes"}   # kernel_shape_of(ckpt)


def to_dict(shape: KernelShape) -> dict:
    return asdict(shape)


def from_dict(d: dict) -> KernelShape:
    return KernelShape(
        comm=Comm(**d["comm"]), hidden=d["hidden"], hc=d["hc"], tp=d["tp"],
        attention=Attention(**d["attention"]), linear=LinearAttention(**d["linear"]),
        indexer=Indexer(**d["indexer"]), moe=MoE(**d["moe"]), spec_k=d["spec_k"],
        device=Device(capability=tuple(d["device"]["capability"]), sms=d["device"]["sms"]),
        drafter=Drafter(**d["drafter"]) if d.get("drafter") else None)


def pin(shape: KernelShape, key: str, value) -> KernelShape:
    """A measured pin: `moe.dynamic_tile_m=32`. Two-level keys into the shape's parts; validated by the part."""
    part, _, name = key.partition(".")
    if not name or part not in {f.name for f in fields(KernelShape)}:
        raise ValueError(f"pin key must be <part>.<field> of KernelShape, got {key!r}")
    owner = getattr(shape, part)
    if owner is None or name not in {f.name for f in fields(owner)}:
        raise ValueError(f"{part} has no field {name!r}")
    return replace(shape, **{part: replace(owner, **{name: value})})


def config_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_record(ranks_dir, shape: KernelShape, *, profile: str, config_sha256: str, admission=None) -> Path:
    """The model's record beside its rank files: the shape (with its pins), the config it was derived from, and the
    admission table the wizard printed. Written on the host at preshard time; containers mount the rank files read-only."""
    if not isinstance(shape, KernelShape):
        raise TypeError("write_record takes a KernelShape")
    path = Path(ranks_dir) / RECORD
    record = {"version": RECORD_VERSION, "profile": profile, "config_sha256": config_sha256,
              "written": _dt.date.today().isoformat(), "shape": to_dict(shape),
              "admission": [asdict(v) for v in (admission or [])]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def read_record(ranks_dir) -> "dict | None":
    path = Path(ranks_dir) / RECORD
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("version") != RECORD_VERSION or "shape" not in record or "config_sha256" not in record:
        raise RuntimeError(f"{path}: not a kernel shape record this engine reads (version {RECORD_VERSION})")
    return record


def bind_recorded(ranks_dir, config_path, derive) -> "tuple[KernelShape, str]":
    """Bind the model's recorded shape when the record is present and still describes this checkpoint's config;
    derive (and bind) when there is no record. Returns (shape, "record" | "derived")."""
    record = read_record(ranks_dir)
    if record is None:
        return bind(derive()), "derived"
    actual = config_sha256(config_path)
    if record["config_sha256"] != actual:
        raise RuntimeError(f"{Path(ranks_dir) / RECORD} was written for another config.json "
                           f"({record['config_sha256'][:12]} != {actual[:12]}): the checkpoint changed since the "
                           "shape wizard ran -- rerun it (python3 -m engine.base.kernel_shape wizard)")
    return bind(from_dict(record["shape"])), "record"


def derive_for(profile: str, ckpt) -> KernelShape:
    """The profile's own derivation of a checkpoint's shape (engine/profiles/<profile>: kernel_shape_of)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    return importlib.import_module(PROFILES[profile]).kernel_shape_of(ckpt)


def _pin_value(raw: str):
    if raw.lower() in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def wizard(profile: str, ckpt, *, ranks=None, pins=(), write=False) -> dict:
    """Judge a checkpoint before any boot: derive its shape, apply measured pins, table every lane's admission, and
    (with write) record it beside the rank files. Returns {"shape", "admission", "path"}."""
    from engine.kernels import cells
    shape = derive_for(profile, ckpt)
    for raw in pins:
        key, sep, value = raw.partition("=")
        if not sep:
            raise ValueError(f"a pin is <part>.<field>=<value>, got {raw!r}")
        shape = pin(shape, key.strip(), _pin_value(value.strip()))
    verdicts = cells.admission(shape)
    path = None
    if write:
        path = write_record(ranks if ranks is not None else ckpt, shape, profile=profile,
                            config_sha256=config_sha256(Path(ckpt) / "config.json"), admission=verdicts)
    return {"shape": shape, "admission": verdicts, "path": path}


def main(argv=None) -> int:
    import argparse
    from engine.kernels import cells
    ap = argparse.ArgumentParser(prog="python3 -m engine.base.kernel_shape",
                                 description="the shape wizard: judge a checkpoint's kernel shape before any boot and "
                                             "record it beside its rank files")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wizard", help="derive, judge and (with --write) record")
    w.add_argument("--profile", required=True, choices=sorted(PROFILES))
    w.add_argument("--ckpt", required=True, help="the checkpoint directory (config.json)")
    w.add_argument("--ranks", help="the rank-file directory to record into (default: the checkpoint directory)")
    w.add_argument("--pin", action="append", default=[], metavar="PART.FIELD=VALUE",
                   help="a measured pin, e.g. moe.dynamic_tile_m=32 (repeatable)")
    w.add_argument("--write", action="store_true", help=f"write {RECORD}")
    s = sub.add_parser("show", help="print an existing record")
    s.add_argument("--ranks", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        record = read_record(a.ranks)
        if record is None:
            print(f"  no {RECORD} under {a.ranks}")
            return 1
        shape = from_dict(record["shape"])
        print(f"  {record['profile']} recorded {record['written']} for config {record['config_sha256'][:12]}")
        print(f"  shape: {shape.describe()}")
        print(cells.table([cells.Verdict(**v) for v in record["admission"]]))
        return 0
    result = wizard(a.profile, a.ckpt, ranks=a.ranks, pins=a.pin, write=a.write)
    print(f"  {a.profile} @ {a.ckpt}")
    print(f"  shape: {result['shape'].describe()}")
    print(cells.table(result["admission"]))
    if result["path"] is not None:
        print(f"  recorded -> {result['path']}")
    return 0


__all__ = ["Device", "Comm", "Attention", "LinearAttention", "Indexer", "MoE", "Drafter", "KernelShape",
           "MEASURED", "bind", "bound", "is_bound", "bind_drafter", "drafter", "reset", "replace", "fields_of",
           "RECORD", "to_dict", "from_dict", "pin", "config_sha256", "write_record", "read_record", "bind_recorded",
           "derive_for", "wizard", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
