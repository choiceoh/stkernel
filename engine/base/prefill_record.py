"""A passed prefill memory gate, kept per node: a restart of the same boot reuses its far pass (base).

Before the door opens, a profile runs its largest prefill chunk at both ends of the served context and
every rank votes each row against the allocator ceiling, the OS reserve and the box's kill lines. The
far end is both the peak and the long pass -- 30.2 s of a same-build restart on 2026-09-15, 9.89 GiB of
workspace against 6.57 GiB at position 0, both to the byte in four boots of three builds -- and a boot of
the same build, configuration, weights and node allocates the same bytes for it. So a boot that ran
every pass and passed keeps what it measured, and a later boot may skip the far pass when all of this holds:

- **The key matches.** The profile names everything that shapes those bytes or the kernels the pass
  dispatches: the engine tree, the runtime packages, the weight files and checkpoint metadata, the
  declared configuration and the node. Records are kept by key, so the production boot and a ticket's
  boot with another tree or configuration on the same node do not overwrite each other's. Without a
  record for this key the reason names what differs from the newest record the rank has.
- **The near pass still runs, and vouches for the record.** Its peak must stay within TOLERANCE_BYTES
  of the recorded one: the record has to describe this boot, not a boot like it.
- **This box is judged, not the recorded one.** The far peak is projected from the near pass that just
  ran (the recorded far-minus-near step on top of today's near peak, never below the recorded far
  peak), and must fit under this boot's allocator ceiling while the free and available bytes the near
  pass just read, less what the peak would add, stay above the OS reserve and the SIGTERM line. A box
  with less room than that runs the pass.
- **Every rank agrees.** One collective; a rank that cannot reuse -- no record, less room, an error
  computing any of it -- makes every rank run the pass.
- **Only a boot that ran the far pass writes**, and only once every row of its ledger has passed.

What a reused boot no longer proves on the day is that far pass itself: execution and finite output at
the far end with the largest chunk. The profile keeps a short continuation pass in its place.

This is a cache, not a knob: removing the directory runs the full gate, and so does `force_full`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import time

SCHEMA = 1
TOLERANCE_BYTES = 64 << 20
"""How far above its record the near pass may peak and still vouch for it: allocator rounding, not drift."""
KEEP = 32
"""Records a rank keeps, most recently used first: production's, and those of the tickets booted between its
restarts (dozens a day). A reuse refreshes its record's mtime, so the build that keeps restarting keeps its record."""

ROW_FIELDS = ("phase", "seconds", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
              "peak_workspace_bytes", "immediately_free_bytes", "available_bytes", "oom_margin_bytes", "passed")
_SAFETENSORS_HEADER_LIMIT = 100 << 20
_SMALL_FILE = 64 << 20
_MIB, _GIB = 1 << 20, 1 << 30


def digest_tree(root) -> str:
    """sha256 over every file under `root`: relative path and content, bytecode caches left out."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                       and p.suffix not in (".pyc", ".pyo")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def file_identity(path) -> dict:
    """What a file holds, cheaply. A safetensors file: its size, its mtime and its header's hash -- the
    tensors' names, dtypes, shapes and offsets in milliseconds, where the 44 GiB body would take minutes,
    and the mtime for a body rewritten in place. Anything else is small and hashed whole, without its
    mtime: the launcher copies the checkpoint metadata afresh at every launch."""
    path = Path(path)
    stat = path.stat()
    with path.open("rb") as stream:
        if path.suffix == ".safetensors":
            head = stream.read(8)
            if len(head) != 8:
                raise ValueError(f"{path}: not a safetensors file")
            (length,) = struct.unpack("<Q", head)
            if length > _SAFETENSORS_HEADER_LIMIT or 8 + length > stat.st_size:
                raise ValueError(f"{path}: implausible safetensors header of {length} bytes")
            header = hashlib.sha256(head + stream.read(length)).hexdigest()
            return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=header)
        if stat.st_size > _SMALL_FILE:
            raise ValueError(f"{path}: {stat.st_size} bytes is not a metadata file")
        return dict(size=stat.st_size, sha256=hashlib.sha256(stream.read()).hexdigest())


def optional_identity(path) -> "dict | None":
    path = Path(path)
    return file_identity(path) if path.is_file() else None


def package_versions(names) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def node_identity(device=None) -> dict:
    """The box: its hostname, the GPU (UUID, name, memory) and the driver the process loaded."""
    import torch
    index = torch.cuda.current_device() if device is None else device
    props = torch.cuda.get_device_properties(index)
    driver = Path("/proc/driver/nvidia/version")
    return dict(hostname=socket.gethostname(), gpu_uuid=str(getattr(props, "uuid", "")), gpu_name=props.name,
                gpu_total_memory=int(props.total_memory),
                driver=hashlib.sha256(driver.read_bytes()).hexdigest() if driver.is_file() else None)


def key_of(components: dict) -> str:
    return hashlib.sha256(json.dumps(components, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def changed(old, new, prefix: str = "") -> "list[str]":
    """The dotted names at which two component trees differ."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return [prefix.rstrip(".") or "components"] if old != new else []
    names = []
    for name in sorted(set(old) | set(new), key=str):
        names += changed(old.get(name), new.get(name), f"{prefix}{name}.")
    return names


@dataclass(frozen=True)
class Limits:
    """The lines THIS boot is held to: its allocator ceiling and the box's reserve and kill line."""
    baseline_reserved_bytes: int
    arena_bytes: int
    allocator_limit_bytes: int
    os_reserve_bytes: int
    sigterm_bytes: int

    @classmethod
    def of(cls, memory) -> "Limits":
        return cls(int(memory.baseline_reserved), int(memory.arena_bytes), int(memory.allocator_limit_bytes),
                   int(memory.os_reserve_bytes), int(memory.sigterm_bytes))


@dataclass(frozen=True)
class Verdict:
    reuse: bool
    reason: str
    peak_workspace_bytes: int = 0           # the projected far peak, outside the arena
    immediately_free_bytes: int = 0         # this box at that peak
    available_bytes: int = 0


class PrefillRecord:
    """One rank's record for this boot's key: loaded at construction, judged against the near pass, written after a full gate."""

    def __init__(self, root, rank: int, components: "dict | None", *, force_full: bool = False,
                 error: "str | None" = None):
        self.root, self.rank = Path(root), int(rank)
        self.components = components
        self.key = key_of(components) if components is not None else None
        self.path = self.root / f"prefill-rank{self.rank}-{self.key[:24] if self.key else 'unkeyed'}.json"
        self.force_full = bool(force_full)
        self.error = error
        self.full = None                    # (near phase, far phase) once this boot ran and passed the far pass
        self.reused = None                  # the far phase this boot skipped on every rank's vote
        self.record, self.reason = self._load()

    @classmethod
    def build(cls, root, rank: int, components, *, force_full: bool = False) -> "PrefillRecord":
        """Never raises. A key that cannot be computed is a full gate, and this rank still casts its vote."""
        try:
            return cls(root, rank, components(), force_full=force_full)
        except Exception as exc:                                  # noqa: BLE001 -- recorded as the reason
            return cls(root, rank, None, force_full=force_full, error=f"{type(exc).__name__}: {exc}")

    def _kept(self) -> "list[Path]":
        """This rank's records, most recently used first; one removed meanwhile is simply not listed."""
        stamped = []
        try:
            for path in self.root.glob(f"prefill-rank{self.rank}-*.json"):
                try:
                    stamped.append((path.stat().st_mtime_ns, path))
                except OSError:
                    continue
        except OSError:
            return []
        return [path for _, path in sorted(stamped, key=lambda item: item[0], reverse=True)]

    def _load(self):
        if self.error is not None:
            return None, f"no key for this boot ({self.error})"
        if self.force_full:
            return None, "the full gate was asked for"
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return None, self._missing()
        except OSError as exc:
            return None, f"the record is unreadable ({exc})"
        try:
            record = json.loads(text)
        except ValueError:
            return None, "the record is not JSON"
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            return None, "the record's schema differs"
        if record.get("key") != self.key:
            return None, "the record under this key names another"
        rows = record.get("rows")
        if not isinstance(rows, dict) or any(not isinstance(rows.get(name), dict) or rows[name].get("passed") is not True
                                             for name in ("near", "far")):
            return None, "the record's rows are incomplete"
        return record, "the record matches"

    def _missing(self) -> str:
        """No record for this key: say what differs from the newest this rank keeps, which is what a person asks."""
        for path in self._kept():
            try:
                newest = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            names = changed(newest.get("components"), self.components)
            return (f"no record for this key; the newest ({path.name}) differs at "
                    + (", ".join(names[:8]) + (" ..." if len(names) > 8 else "") if names else "no named component"))
        return "no record on this node"

    def verdict(self, near: dict, limits: Limits) -> Verdict:
        """May this rank skip the far pass, given the near pass it just ran? Host arithmetic only; never raises."""
        try:
            return self._verdict(near, limits)
        except Exception as exc:                                  # noqa: BLE001 -- a verdict that fails is a full gate
            return Verdict(False, f"the verdict failed ({type(exc).__name__}: {exc})")

    def _verdict(self, near: dict, limits: Limits) -> Verdict:
        if self.record is None:
            return Verdict(False, self.reason)
        was_near, was_far = self.record["rows"]["near"], self.record["rows"]["far"]
        if near.get("phase") != was_near.get("phase") or near.get("passed") is not True:
            return Verdict(False, f"the near pass is {near.get('phase')!r}, the record's is {was_near.get('phase')!r}")
        now, near_peak, far_peak = (int(near["peak_workspace_bytes"]), int(was_near["peak_workspace_bytes"]),
                                    int(was_far["peak_workspace_bytes"]))
        if now > near_peak + TOLERANCE_BYTES:
            return Verdict(False, f"the near pass peaked {(now - near_peak) / _MIB:.0f} MiB above its record")
        projected = max(far_peak, now + (far_peak - near_peak))
        ceiling = limits.allocator_limit_bytes - limits.baseline_reserved_bytes - limits.arena_bytes
        if projected > ceiling:
            return Verdict(False, f"the projected far peak {projected / _GIB:.2f} GiB is over this boot's "
                                  f"{ceiling / _GIB:.2f} GiB workspace ceiling")
        extra = max(0, limits.baseline_reserved_bytes + limits.arena_bytes + projected - int(near["reserved_bytes"]))
        free = int(near["immediately_free_bytes"]) - extra
        available = int(near["available_bytes"]) - extra
        if free < limits.os_reserve_bytes:
            return Verdict(False, f"this box would have {free / _GIB:.2f} GiB immediately free at the projected peak, "
                                  f"under the {limits.os_reserve_bytes / _GIB:.2f} GiB OS reserve")
        if available < limits.sigterm_bytes:
            return Verdict(False, f"this box would have {available / _GIB:.2f} GiB available at the projected peak, "
                                  f"under the {limits.sigterm_bytes / _GIB:.2f} GiB SIGTERM line")
        return Verdict(True, f"the record matches: far peak {projected / _GIB:.2f} GiB projected, "
                             f"{available / _GIB:.2f} GiB available at it", projected, free, available)

    def ran_full(self, near_phase: str, far_phase: str) -> None:
        self.full = (near_phase, far_phase)

    def used(self, phase: str) -> None:
        """Every rank reused this record for `phase`: say so, and mark it recently used for the pruning in `write`."""
        self.reused = phase
        try:
            os.utime(self.path)
        except OSError:
            pass                                                  # a read-only cache still reuses; it only prunes worse

    def write(self, phases, **provenance) -> Path:
        """Keep this boot's gate under its key. Only after it ran the far pass, and only when every row passed."""
        if self.full is None or self.key is None:
            raise RuntimeError("only a boot that ran the full gate writes its record")
        if not phases or any(row.get("passed") is not True for row in phases) or any(row.get("reused") for row in phases):
            raise RuntimeError("a record is only written from a ledger in which every row ran and passed")
        by_phase = {row["phase"]: row for row in phases}
        rows = {}
        for name, phase in zip(("near", "far"), self.full):
            if phase not in by_phase:
                raise RuntimeError(f"the ledger has no {phase!r} row")
            rows[name] = {field: by_phase[phase].get(field) for field in ROW_FIELDS}
        payload = dict(schema=SCHEMA, key=self.key, rank=self.rank, components=self.components, rows=rows,
                       written_at=time.time(), **provenance)
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.root, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream, indent=1, sort_keys=True, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        for stale in self._kept()[KEEP:]:
            if stale != self.path:
                stale.unlink(missing_ok=True)
        return self.path

    def outcome(self) -> str:
        """One line for the boot log: what this boot did with its record."""
        if self.reused is not None:
            return f"reused {self.path} for {self.reused}"
        if self.full is not None:
            return f"ran the full gate ({self.reason})"
        return f"no far pass was judged ({self.reason})"
