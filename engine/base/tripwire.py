"""Every host collective says where it stands, and ranks that meet at different places die
saying so (45차, 2026-09-13).

Three fleets went down in thirty-six hours the same way and none of them said so. On 09-12
22:31 ranks 0/1 offered seven rows to an all_gather and ranks 2/3 six, and the four sat in it
for sixteen minutes. On 09-13 05:24 a prefix restore failed on ranks 2/3 only; they prefilled
on their own while ranks 0/1 waited in a transfer vote, and the death certificates read
"Connection closed by peer" and "unspecified launch failure" -- the symptoms on the survivors,
never the cause. The rule that ends the class is one sentence: a decision that gates a
collective is never taken from rank-local state. Where that rule is kept by construction
(the step broadcast, the votes) nothing here fires. Where it is broken by a bug not yet
written, this is what turns a hang into one attributed line.

A tagged host collective carries, beside its values, every rank's (sequence, site, count, mode)
in its own slot of a FIXED-LENGTH int64 vector. Fixed length, so two ranks at different sites
still complete the same all-reduce instead of pairing mismatched buffers; per-rank slots, so
every rank reads the same table and every rank raises the same `CollectiveDivergence`, or
none does. The sequence counts every tagged collective this process has made; a rank that
skipped or added one meets the others one call out of step and the tags say which call and
where. The step broadcast first meets the same fixed all-reduce, then carries
rank 0's stamp for followers to check; mismatched collective types never meet.

Cost: 80 int64 on the control group per vote, the same all-reduce the vote already was.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import zlib
from pathlib import Path

SLOTS = 64            # values a tagged collective may carry
TAG = 4               # per-rank tag: (sequence, site id, value count, 1 when an exchange)
STEP = "step"         # the broadcast's site


class CollectiveDivergence(RuntimeError):
    """The ranks reached different collectives. Raised on every rank alike, naming each rank's site."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details


SEQ_WRAP = 1 << 20    # the sequence rides the wire modulo this: every tag stays below 2^24, exact even where a
                      # fake or a test transport sums in float32 (LocalTP rounded 701907001 to 701907008)


def site_id(site: str) -> int:
    return (zlib.crc32(str(site).encode()) & 0xFFFF) + 1               # 1..65536; never 0: 0 is an empty slot


def layout(world: int) -> "tuple[int, int, int]":
    """(tag slots, values offset, values per rank in exchange mode) for a world."""
    world = max(1, int(world))
    return TAG * world, TAG * world, SLOTS // world


def pack(world: int, rank: int, seq: int, site: int, values, exchange: bool) -> "list[int]":
    """This rank's contribution: its tag in its slot, its values summed (vote) or in its block (exchange)."""
    tags, offset, block = layout(world)
    values = [int(v) for v in values]
    limit = block if exchange else SLOTS
    if len(values) > limit:
        raise ValueError(f"a tagged collective carries at most {limit} values in this mode, not {len(values)}")
    vector = [0] * (tags + SLOTS)
    vector[TAG * rank: TAG * rank + TAG] = [int(seq), int(site), len(values), int(bool(exchange))]
    start = offset + (block * rank if exchange else 0)
    vector[start: start + len(values)] = values
    return vector


def unpack(vector, world: int) -> "tuple[list[tuple[int, int, int]], list[int]]":
    """(each rank's (sequence, site, count, exchange), the values region) of a reduced vector."""
    tags, offset, _ = layout(world)
    vector = [int(v) for v in vector]
    if len(vector) != tags + SLOTS:
        raise ValueError(f"a tagged collective is {tags + SLOTS} long for {world} ranks, not {len(vector)}")
    return [tuple(vector[TAG * r: TAG * r + TAG]) for r in range(world)], vector[offset:]


def peers_agree(vector, world: int) -> "list[int]":
    """What the fleet returns when every peer stands where this rank stands and votes as it votes
    (for fakes): the tag copied into every slot, the values summed `world` times."""
    tags, offset, _ = layout(world)
    vector = [int(v) for v in vector]
    mine = next((vector[TAG * r: TAG * r + TAG] for r in range(world) if vector[TAG * r + 1]), [0] * TAG)
    if mine[3]:                                      # an exchange: every peer's block holds this rank's values
        rank = next(r for r in range(world) if vector[TAG * r + 1])
        block = SLOTS // world
        values = vector[offset + block * rank: offset + block * rank + mine[2]]
        region = [0] * SLOTS
        for r in range(world):
            region[block * r: block * r + len(values)] = values
        return list(mine) * world + region
    return list(mine) * world + [v * world for v in vector[offset:]]


PEER_LEFT = ("Connection closed by peer", "Connection reset", "recvValue failed", "Broken pipe",
             "NCCL", "ncclRemoteError", "ncclSystemError", "timed out", "Timeout", "RankLeft", "rank left")


def classify(exc: BaseException) -> str:
    """'divergence' | 'peer-left' | 'stalled' | 'local' -- what a rank's death certificate should say."""
    if isinstance(exc, CollectiveDivergence):
        return "divergence"
    text = f"{type(exc).__name__}: {exc}"
    if type(exc).__name__ == "RankLeft" or any(mark.lower() in text.lower() for mark in PEER_LEFT):
        return "peer-left"
    return "local"


def death_note(directory, rank: int, exc: BaseException, *, phase=None, calls=None, say=print) -> "dict | None":
    """One JSON line a container removal cannot erase: what this rank died of, where, and whether the
    cause is its own or another rank's."""
    kind = classify(exc)
    note = dict(rank=int(rank), kind=kind, phase=phase, calls=calls, t=time.strftime("%F %T"), pid=os.getpid(),
                error=f"{type(exc).__name__}: {str(exc)[:1500]}",
                meaning={"divergence": "this rank and its peers reached different collectives; every rank has this note",
                         "peer-left": "a peer died or stalled first; the cause is in that rank's log and note",
                         "local": "this rank's own failure; peers will report peer-left"}[kind])
    if isinstance(exc, CollectiveDivergence) and exc.details is not None:
        # Keep the complete differing rows even when the exception summary is truncated.
        note["divergence"] = exc.details
    say(f"[serve] death rank={rank} kind={kind} phase={phase!r}: {note['error'].splitlines()[0][:300]}", flush=True)
    if directory is None:
        return note
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"death-rank{int(rank)}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(note, ensure_ascii=False, indent=1))
        note["path"] = str(path)
    except OSError as exc2:
        note["path_error"] = f"{type(exc2).__name__}: {exc2}"
    return note


class Tripwire:
    """The tagged collectives of one process. `of(comm)` keeps one per comm object, so the server's
    votes, the engine's gather votes and the step broadcast share ONE sequence."""

    def __init__(self, comm):
        self.comm = comm
        self.calls = 0                                   # tagged collectives so far: the sequence every rank must share
        self.last = None                                 # (sequence, site) of the last one, for death notes

    @classmethod
    def of(cls, comm) -> "Tripwire":
        wire = getattr(comm, "tripwire", None)
        if wire is None:
            wire = cls(comm)
            try:
                setattr(comm, "tripwire", wire)
            except (AttributeError, TypeError):
                pass                                     # a comm that cannot hold it gets a fresh one each time (world 1)
        return wire

    @property
    def world(self) -> int:
        return int(getattr(self.comm, "world_size", 1) or 1)

    @property
    def rank(self) -> int:
        return int(getattr(self.comm, "rank", 0) or 0)

    # -- the collectives -----------------------------------------------------------
    def vote(self, site: str, values) -> "list[int]":
        """How many ranks say yes to each value (the sums), the site agreed first."""
        values = [int(v) for v in values]
        if self.world <= 1 or not values:
            return values
        return self._meet(site, values, exchange=False)[: len(values)]

    def exchange(self, site: str, values) -> "list[list[int]]":
        """Every rank's values, by rank, the site and the count agreed first."""
        values = [int(v) for v in values]
        if self.world <= 1:
            return [values]
        _, _, block = layout(self.world)
        region = self._meet(site, values, exchange=True)
        return [region[block * r: block * r + len(values)] for r in range(self.world)]

    def agree(self, site: str, values) -> "list[int]":
        """The values every rank holds; raises when any rank holds different ones."""
        rows = self.exchange(site, values)
        if any(row != rows[0] for row in rows):
            raise CollectiveDivergence(
                f"the ranks disagree at {site!r}: " + ", ".join(f"rank{r}={row}" for r, row in enumerate(rows)))
        return rows[0]

    def agree_payload(self, site: str, payload) -> None:
        """Check a small host result without variable-sized collectives on the normal path.

        Sixteen 16-bit SHA256 words fit the TP4 exchange and stay exact in test
        transports that sum through FP32. Only a mismatch gathers the actual
        rows, which the death note preserves for every rank.
        """
        if self.world <= 1:
            return
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest = hashlib.sha256(raw).digest()
        words = [int.from_bytes(digest[i:i+2], "big") for i in range(0, len(digest), 2)]
        rows = self.exchange(site, words)
        if any(row != rows[0] for row in rows):
            gather = getattr(self.comm, "gather_objects", None)
            details = dict(site=site, fingerprints=rows)
            if callable(gather):
                details["ranks"] = gather(payload)
            else:
                details.update(rank=self.rank, local=payload)
            raise CollectiveDivergence(f"the ranks disagree at {site!r}; rank results are in the death note",
                                      details=details)

    def before_broadcast(self, site: str = STEP) -> None:
        """Meet the same fixed all-reduce as a peer's vote BEFORE entering a broadcast.

        A stamp checked after broadcast cannot detect broadcast-vs-gather: the
        two different collectives never complete. This preflight can.
        """
        self.exchange(site + ":broadcast", [])

    def stamp(self, site: str = STEP) -> "tuple[int, int]":
        """rank 0's side of a broadcast: the (sequence, site) it is at, sent with the payload."""
        self.calls += 1
        self.last = (self.calls, site)
        return self.calls, site_id(site)

    def expect(self, stamp, site: str = STEP) -> None:
        """A follower's side of a broadcast: the stamp it received must be the one it is at."""
        self.calls += 1
        self.last = (self.calls, site)
        seq, sid = (int(stamp[0]), int(stamp[1])) if stamp is not None else (None, None)
        if seq != self.calls or sid != site_id(site):
            raise CollectiveDivergence(
                f"rank {self.rank} is at collective #{self.calls} ({site!r}) but rank 0 broadcast #{seq} "
                f"(site id {sid}, expected {site_id(site)}): a host collective was skipped or added on one side "
                f"between the last agreed one and this step")

    # -- the machinery -------------------------------------------------------------
    def _reduce(self, vector) -> "list[int]":
        comm = self.comm
        if hasattr(comm, "all_reduce_host"):                 # the control group: no device work, no stream wait
            return [int(v) for v in comm.all_reduce_host(vector)]
        import torch                                         # LocalTP and world-1 device comms: the NCCL group
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.tensor(vector, dtype=torch.int64, device=device)
        return [int(v) for v in comm.all_reduce(t).tolist()]

    def _meet(self, site: str, values, *, exchange: bool) -> "list[int]":
        self.calls += 1
        seq, sid, world = self.calls, site_id(site), self.world
        self.last = (seq, site)
        tags, region = unpack(self._reduce(pack(world, self.rank, seq % SEQ_WRAP, sid, values, exchange)), world)
        mine = (seq % SEQ_WRAP, sid, len(values), int(bool(exchange)))
        if any(tag != mine for tag in tags):
            raise CollectiveDivergence(self._describe(site, values, tags))
        return region

    def _describe(self, site: str, values, tags) -> str:
        names = None
        gather = getattr(self.comm, "gather_objects", None)
        if callable(gather):
            try:                                             # one more collective, on the bad step only, for the names:
                names = gather((self.rank, self.calls, site, len(values)))   # every rank is here, since every rank saw the same table
            except Exception as exc:                         # noqa: BLE001 -- the ids alone still say it
                names = None
        lines = [f"the ranks met at different collectives (this rank {self.rank}: #{self.calls} {site!r}, "
                 f"{len(values)} values):"]
        for r, (seq, sid, count, exchange) in enumerate(tags):
            name = next((n[2] for n in (names or ()) if n and n[0] == r), None)
            kind = "exchange" if exchange else "vote"
            lines.append(f"  rank{r}: #{seq} {name!r} ({kind}, {count} values)" if name is not None
                         else f"  rank{r}: #{seq} site id {sid} ({kind}, {count} values)")
        lines.append("Every rank raises this. The rank whose count differs took a branch the others did not; "
                     "its own log says which. Nothing entered the collective that could not complete.")
        return "\n".join(lines)
