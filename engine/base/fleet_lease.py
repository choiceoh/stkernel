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
import time
from pathlib import Path

DEFAULT_PATH = Path.home() / "st-fleet.lock"
GRACE_S = 900.0            # 15 min without evidence or heartbeat before a lease is stale
FIELDS = ("owner", "host", "pid", "container", "since", "beat", "est_minutes", "note")


class LeaseHeld(RuntimeError):
    """Someone else holds the fleet. The message says who, since when and why."""


class LeaseLost(RuntimeError):
    """The lease is not ours to renew or release (reclaimed, or never taken)."""


def _now() -> float:
    return time.time()


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
        return {"owner": raw.strip()[:200] or "unknown", "host": "", "pid": 0, "container": "",
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
        # An older launcher's plain-text lock, or a record we cannot parse. It carries no
        # evidence and no heartbeat, so nothing here can ever call it stale -- and a lease
        # we cannot judge is held, never free (D3). A human clears it with `stop`.
        return True
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
    return f"{record.get('owner')} on {record.get('host') or '?'} since {when}" + (f" ({note})" if note else "")


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


def renew(owner: str, *, path=DEFAULT_PATH) -> dict:
    """Push the heartbeat forward. A long boot must not look stale while it works."""
    record = read(path)
    if not record or record.get("owner") != owner:
        raise LeaseLost(f"the lease is {describe(record)}, not {owner}")
    record["beat"] = _now()
    Path(path).write_text(json.dumps(record, indent=2) + "\n")
    return record


def release(owner: str, *, path=DEFAULT_PATH, force: bool = False) -> "dict | None":
    """Give the fleet back. Only its owner may, unless `force` says a human decided."""
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


def owner_for(container: str, *, path=DEFAULT_PATH) -> str:
    """Resolve an owning launcher's stop without releasing another workload."""
    record = read(path)
    if not record:
        return ""
    if record.get("container") == container:
        return record["owner"]
    if record.get("opaque") and f" {container} " in record["owner"]:
        return record["owner"]
    raise LeaseHeld(f"the fleet is held by {describe(record)}")


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
        assert release("boot-b", path=path)["owner"] == "boot-b" and read(path) is None
        # an older launcher's plain-text lock is a lease we can read and must honour
        Path(path).write_text("choiceoh@srv2 st-glm53 2026-09-12 10:00:00\n")
        opaque = read(path)
        assert opaque["owner"].startswith("choiceoh@srv2") and alive(opaque, container_up=lambda n: True)
    print("  fleet_lease: one owner, evidence before heartbeat, unreachable is not free, "
          "stale is reclaimable once, plain-text locks are honoured OK")


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
    return name in done.stdout.split()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="the fleet lease, for the launcher")
    parser.add_argument("action", choices=("acquire", "release", "renew", "read", "owner", "selfcheck"))
    parser.add_argument("--owner", default="")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--container", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--est-minutes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
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
    elif a.action == "owner":
        print(owner_for(a.container, path=a.path))
    else:
        print("released" if release(a.owner, path=a.path, force=a.force) else "free")
