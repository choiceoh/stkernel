"""Who holds the four Sparks, since when, and how we know they still do (base).

The engine takes the whole fleet: one container per node, every GPU, the RoCE
fabric. Two sessions doing that at once is not a slow bench, it is two dead
boots -- 09-11 19:42 cost both of them their window and their evidence.

What the launcher had until now was `echo "$USER@$HOST" > ~/st-fleet.lock`: no
owner it could check, no expiry, and nothing that could tell a crashed boot from
a running one, so the only way out of a stale lock was a human deleting a file.
This is that lock made into a lease -- the smallest thing with an owner, a
reason, and EVIDENCE of liveness.

It is deliberately not a queue. Aging, priority, pause/resume, handoff, evidence
retention and runner pinning already exist in bench/fleet.sh and are a solved
problem; rebuilding them here would repeat, mirrored, the mistake this module
exists to end. The queue takes the lease when it grants. The engine holds it
while it serves. One record, one owner, one liveness rule.

Liveness is evidence, not a timer, because the launcher's own process exits as
soon as the containers are up:

  1. the named container is up on the head node -- the boot is real, right now;
  2. else the recorded pid is alive on THIS host -- a boot still starting;
  3. else the lease is stale once `grace` has passed since its last heartbeat,
     and only then may another owner take it.

A lease with no evidence and no grace left is reclaimable. A lease whose
evidence cannot be checked (the head node is unreachable) is NOT: unreachable is
not the same as free, and D3 says die rather than guess.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

# Every ST container bind-mounts this directory at the same path, so the engine inside one
# can read the lease -- which is what lets it notice a yield request at all.
DEFAULT_PATH = Path("/home/choiceoh/glm53-logs/st-fleet.lock")
GRACE_S = 900.0            # 15 min without evidence or heartbeat before a lease is stale
LOCK_STALE_S = 30.0        # a mutation lock older than this was left by a killed writer
FIELDS = ("owner", "host", "pid", "container", "since", "beat", "est_minutes", "note")


class LeaseHeld(RuntimeError):
    """Someone else holds the fleet. The message says who, since when and why."""


class LeaseLost(RuntimeError):
    """The lease is not ours to renew or release (reclaimed, or never taken)."""


def _now() -> float:
    return time.time()


class _Mutation:
    """A short lock around read-modify-write: the holder publishes its state while another
    session asks it to yield, and neither may drop the other's field."""

    def __init__(self, path):
        self.path = Path(str(path) + ".mut")

    def __enter__(self):
        for _ in range(300):
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
                return self
            except FileExistsError:
                try:
                    if _now() - self.path.stat().st_mtime > LOCK_STALE_S:
                        self.path.unlink()          # its writer died holding it
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.02)
        raise LeaseHeld(f"the lease at {self.path} stayed locked by another writer")

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False


def _write(path, record) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def read(path=DEFAULT_PATH) -> "dict | None":
    """The lease record, or None. A corrupt file reads as a lease we cannot judge."""
    try:
        raw = Path(path).read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LeaseHeld(f"fleet lease at {path} cannot be read: {exc}") from exc
    try:
        record = json.loads(raw)
        if not isinstance(record, dict) or not record.get("owner"):
            raise ValueError("no owner")
    except ValueError:
        # Not ours to interpret and not ours to delete: an older launcher's
        # plain-text lock reads exactly like this, and it means the same thing.
        # An older launcher's plain-text lock. We cannot parse it, but it usually NAMES a
        # pid and a host, and that is enough to judge: treating every such record as
        # permanently held made a dead session a dead hand -- 2026-09-12, a lock whose pid
        # had exited blocked three queued reservations until a human ran `stop`.
        text = raw.strip()[:200]
        named = re.search(r"pid=(\d+)", text)
        host = re.search(r"@([A-Za-z0-9_.-]+)", text)
        return {"owner": text or "unknown", "host": host.group(1).split(".")[0] if host else "",
                "pid": int(named.group(1)) if named else 0, "container": "",
                "since": 0.0, "beat": 0.0, "est_minutes": 0, "note": "opaque lease record",
                "opaque": True}
    return record


def alive(record, *, container_up=None, grace: float = GRACE_S, now=None) -> bool:
    """Evidence first, heartbeat only as the last word. See the module note.

    `container_up(name) -> bool | None` answers for the head node; None means the
    question could not be asked, which keeps the lease held rather than free.
    """
    if not record:
        return False
    if record.get("opaque"):
        # A record we cannot parse is held, never free (D3) -- unless it names a pid on a
        # host we can ask, and that pid is gone, and no ST container is running. Then it
        # is not a judgement call: its session left without clearing its lock.
        pid, host = int(record.get("pid") or 0), record.get("host") or ""
        if not pid or host != os.uname().nodename.split(".")[0]:
            return True
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except ProcessLookupError:
            pass
        return container_up("st-") if container_up is not None else True
    now = _now() if now is None else now
    container = record.get("container")
    if container and container_up is not None:
        answer = container_up(container)
        if answer is None:
            return True                       # unreachable is not free (D3)
        if answer:
            return True
    pid, host = int(record.get("pid") or 0), record.get("host") or ""
    if pid and host == os.uname().nodename.split(".")[0]:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            pass
        except PermissionError:
            return True                       # someone else's process, still running
    last = float(record.get("beat") or record.get("since") or 0.0)
    return bool(last) and now - last < grace


def describe(record) -> str:
    if not record:
        return "free"
    since = record.get("since") or 0
    when = time.strftime("%F %T", time.localtime(since)) if since else "unknown time"
    note = record.get("note") or ""
    line = f"{record.get('owner')} on {record.get('host') or '?'} since {when}" + (f" ({note})" if note else "")
    state = record.get("state") or {}
    if state:
        line += ": " + ", ".join(f"{k}={v}" for k, v in sorted(state.items()))
    asked = yield_requested(record)
    if asked:
        line += f" -- asked to yield to {asked.get('requester')}"
    return line


def acquire(owner: str, *, path=DEFAULT_PATH, container: str = "", note: str = "",
            est_minutes: int = 0, container_up=None, grace: float = GRACE_S) -> dict:
    """Take the fleet, or raise LeaseHeld naming who has it.

    The record is created with O_EXCL, so two launchers racing on the head node
    cannot both believe they won; the loser re-reads and reports the winner.
    """
    if not owner or "\n" in owner:
        raise ValueError("a lease owner is a single-line name")
    path = Path(path)
    record = {"owner": owner, "host": os.uname().nodename.split(".")[0], "pid": os.getpid(),
              "container": container, "since": _now(), "beat": _now(),
              "est_minutes": int(est_minutes), "note": note}
    payload = json.dumps(record, indent=2) + "\n"
    for attempt in (1, 2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            held = read(path)
            if held and alive(held, container_up=container_up, grace=grace):
                raise LeaseHeld(f"the fleet is held by {describe(held)}")
            if attempt == 2:
                raise LeaseHeld(f"the fleet lease at {path} could not be reclaimed")
            try:                               # stale: reclaim it, once
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        return record
    raise LeaseHeld(f"the fleet lease at {path} could not be taken")


def publish(owner: str, *, path=DEFAULT_PATH, **state) -> dict:
    """The holder says what it is doing, and beats at the same time.

    A lease that only says "held" makes every other session guess. The engine knows what
    it is doing -- layers loaded, requests in flight, when it expects to be free -- so it
    says so, and `read` turns that into one line anybody can act on.
    """
    with _Mutation(path):
        record = read(path)
        if not record or record.get("owner") != owner:
            raise LeaseLost(f"the lease is {describe(record)}, not {owner}")
        if state:
            record["state"] = {**(record.get("state") or {}), **state}
        record["beat"] = _now()
        _write(path, record)
        return record


def renew(owner: str, *, path=DEFAULT_PATH) -> dict:
    """Push the heartbeat forward. A long boot must not look stale while it works."""
    return publish(owner, path=path)


def request_yield(requester: str, *, path=DEFAULT_PATH, reason: str = "") -> "dict | None":
    """Ask the holder to hand the fleet over. Returns the lease asked, or None if free.

    This is the half a queue alone cannot do. The holder is a running engine with live
    conversations, so the answer is not "kill it" but "stop admitting, finish what you
    have, park it where it survives, and let go" -- and only the engine can carry that
    out. Asking is not taking: the requester still acquires the lease afterwards, like
    anybody else, and loses the race if a third session is quicker.
    """
    with _Mutation(path):
        record = read(path)
        if record is None:
            return None
        if record.get("opaque"):
            raise LeaseHeld("the holder predates the yield protocol; ask its session directly")
        record["yield_to"] = {"requester": requester, "reason": reason, "asked": _now()}
        _write(path, record)
        return record


def yield_requested(record) -> "dict | None":
    return (record or {}).get("yield_to") or None


def clear_yield(owner: str, *, path=DEFAULT_PATH) -> None:
    """Withdraw the request: the asker gave up, or the holder has already let go."""
    with _Mutation(path):
        record = read(path)
        if record and record.get("owner") == owner and record.pop("yield_to", None) is not None:
            _write(path, record)


def release(owner: str, *, path=DEFAULT_PATH, force: bool = False) -> "dict | None":
    """Give the fleet back. Only its owner may, unless `force` says a human decided.

    Under the same lock as `publish`: without it a holder's next heartbeat, landing between
    this read and this unlink, rewrites the file and the released lease comes back.
    """
    with _Mutation(path):
        record = read(path)
        if record is None:
            return None
        if not force and record.get("owner") != owner:
            raise LeaseLost(f"the lease is {describe(record)}, not {owner}")
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
        return record


def _selfcheck() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lease"
        assert read(path) is None and not alive(read(path))
        acquire("boot-a", path=path, container="st-glm53", note="45 layers", est_minutes=30)
        try:
            acquire("boot-b", path=path, container_up=lambda name: True)
            raise AssertionError("a live lease must refuse a second owner")
        except LeaseHeld as exc:
            assert "boot-a" in str(exc), exc
        # the container is the evidence: gone, and the pid gone, it goes stale on grace
        record = read(path)
        record["pid"] = 0
        assert alive(record, container_up=lambda name: True)
        assert alive(record, container_up=lambda name: None)          # unreachable != free
        assert alive(record, container_up=lambda name: False)         # heartbeat is fresh
        record["beat"] = record["since"] = _now() - 2 * GRACE_S
        assert not alive(record, container_up=lambda name: False)
        Path(path).write_text(json.dumps(record, indent=2) + "\n")
        taken = acquire("boot-b", path=path, container_up=lambda name: False)
        assert taken["owner"] == "boot-b" and read(path)["owner"] == "boot-b"
        try:
            release("boot-a", path=path)
            raise AssertionError("only the owner releases")
        except LeaseLost:
            pass
        assert renew("boot-b", path=path)["beat"] >= taken["beat"]
        # the holder says what it is doing, and another session asks it to go
        publish("boot-b", path=path, phase="serving", running=3, free_in_minutes=12)
        assert "running=3" in describe(read(path)) and "phase=serving" in describe(read(path))
        assert yield_requested(read(path)) is None
        request_yield("boot-c", path=path, reason="45차 §22 그래프 재생")
        asked = yield_requested(read(path))
        assert asked["requester"] == "boot-c" and "asked to yield to boot-c" in describe(read(path))
        # publishing again must not drop the request, nor the request the state
        publish("boot-b", path=path, running=0)
        assert yield_requested(read(path))["requester"] == "boot-c"
        assert read(path)["state"]["phase"] == "serving" and read(path)["state"]["running"] == 0
        clear_yield("boot-b", path=path)
        assert yield_requested(read(path)) is None
        assert release("boot-b", path=path)["owner"] == "boot-b" and read(path) is None
        # an older launcher's plain-text lock is a lease we can read and must honour
        Path(path).write_text("choiceoh@srv2 st-glm53 2026-09-12 10:00:00\n")
        opaque = read(path)
        assert opaque["owner"].startswith("choiceoh@srv2") and alive(opaque, container_up=lambda n: True)
        try:
            request_yield("boot-c", path=path)
            raise AssertionError("an opaque holder cannot be asked to yield")
        except LeaseHeld:
            pass
    print("  fleet_lease: one owner, evidence before heartbeat, unreachable is not free, "
          "stale is reclaimable once, plain-text locks are honoured, state and yield "
          "survive each other OK")


def docker_evidence(name: str) -> "bool | None":
    """Is that container up HERE? None when docker cannot answer -- the caller keeps
    the lease held rather than treating an unreachable daemon as an empty fleet."""
    import subprocess
    try:
        done = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode:
        return None
    names = done.stdout.split()
    if name.endswith("-"):                       # a prefix: "is ANY of these running?"
        return any(n.startswith(name) for n in names)
    return name in names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="the fleet lease, for the launcher")
    parser.add_argument("action", choices=("acquire", "release", "renew", "read", "publish",
                                          "yield", "clear-yield", "asked", "selfcheck"))
    parser.add_argument("--owner", default="")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--container", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--est-minutes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--requester", default="")
    parser.add_argument("--state", default="", help="k=v,k=v published with the heartbeat")
    a = parser.parse_args()
    # The lease file lives on the head node and this module runs THERE, so the container
    # question is answered by the local docker: the evidence and the record are colocated.
    if a.action == "selfcheck":
        _selfcheck()
    elif a.action == "read":
        held = read(a.path)
        if held and not alive(held, container_up=docker_evidence):
            print("free (stale: " + describe(held) + ")")
        else:
            print(describe(held) if held else "free")
    elif a.action == "acquire":
        acquire(a.owner, path=a.path, container=a.container, note=a.note,
                est_minutes=a.est_minutes, container_up=docker_evidence)
        print("held")
    elif a.action == "renew":
        renew(a.owner, path=a.path)
        print("renewed")
    elif a.action == "publish":
        pairs = dict(p.split("=", 1) for p in a.state.split(",") if "=" in p)
        publish(a.owner, path=a.path, **pairs)
        print("published")
    elif a.action == "yield":
        asked = request_yield(a.requester or a.owner, path=a.path, reason=a.note)
        print(describe(asked) if asked else "free")
    elif a.action == "clear-yield":
        clear_yield(a.owner, path=a.path)
        print("cleared")
    elif a.action == "asked":
        asked = yield_requested(read(a.path))
        print(f"{asked['requester']}: {asked.get('reason') or 'no reason given'}" if asked else "no")
    else:
        print("released" if release(a.owner, path=a.path, force=a.force) else "free")
