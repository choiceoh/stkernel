"""Deploy main to the ST engine when main has moved, the engine is quiet, and the suite has not regressed.

    python3 launchers/st-deploy-watch.py --once --dry-run     # what would it do, and why
    python3 launchers/st-deploy-watch.py --once               # one cycle: may deploy
    python3 launchers/st-deploy-watch.py                      # the loop systemd runs

Runs on the head node, beside the supervisor (launchers/st-glm53.service), and stdlib only for the
same reason `base/fleet_lease.py` is: it has to be runnable where the engine's virtualenv is not.

**Three conditions, and all of them can say no.**

1. *main moved.* The deployed sha is a file next to the releases, not a guess. A sha the gate has
   already rejected is not retried until main moves past it -- a broken merge must not become a
   restart loop.
2. *The engine is quiet.* `vllm:num_requests_running` and `num_requests_waiting` are both zero for
   the whole of `--quiet` seconds, sampled; one arrival resets it. This is what coalesces a merge
   storm into one restart: ten merges in an afternoon become one deploy at the first gap, and no
   request is ever cut off mid-answer. It also means a busy engine is never taken down, which is
   the property worth more than freshness.
3. *The suite did not regress.* The engine tests are run over the candidate tree AND over the tree
   that is deployed, and the answer is the DIFFERENCE: files that fail on the candidate and not on
   the deployed one, or fail worse. A dozen engine test files fail on any tree right now, so an
   absolute count is not a signal and a green bar is not available; the delta is.

What it does NOT do: it does not arm itself (`--install` prints the two commands and stops), it
does not take the fleet from another stack -- nor from a ticket the queue granted or a session's
boot: a live fleet lease of any kind but `production` defers the deploy to the next cycle, without
recording anything -- and it does not restart more often than `--min-gap` seconds however fast
main moves. Its own boots hold the `production` lease (ST_LEASE_KIND=production), the fleet's
default state, which the queue asks to hand over only through the same quiet gate used here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("ST_HOME", Path.home()))
RELEASES = Path(os.environ.get("ST_RELEASES", HOME / "st-releases"))
STATE = Path(os.environ.get("ST_DEPLOY_STATE", RELEASES / "deploy-state.json"))
SOURCE = Path(os.environ.get("ST_SOURCE", HOME / "stkernel"))
BASE = os.environ.get("ST_BASE", "http://127.0.0.1:8000")
SERVICE = os.environ.get("ST_SERVICE", "st-glm53")
CARRY = ("engine", "launchers", "tests", "probes", "build")   # what a release has to hold to boot and be judged
LOCK = Path(os.environ.get("FLEET_LEASE_PATH", "/home/choiceoh/glm53-logs/st-fleet.lock"))   # the one lease file

# The quiet gate's reading of the door, and the lease, live with the lease module (stdlib, in every
# release under engine/base/): the queue applies the same gate before asking production to hand
# over, so there is one definition of "quiet" and one of "taken".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine.base import fleet_lease                                                  # noqa: E402
from engine.base.fleet_lease import LeaseHeld, door_load as busy, door_unsupported as unsupported   # noqa: E402
import st_release                                                                    # noqa: E402  the one shape of release


def run(cmd, cwd=None, timeout=1800, env=None):
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                         env={**os.environ, **(env or {})})
    return out.returncode, out.stdout, out.stderr


# -- the three conditions, as answers rather than actions -----------------------------------------
def state_of(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def wanted(head: str, held: dict) -> "str | None":
    """Why this sha should be deployed, or None. A sha the gate rejected waits for main to move on."""
    if not head:
        return None
    if head == held.get("deployed"):
        return None
    if head == held.get("rejected"):
        return None
    return f"main is {head[:12]}, deployed is {str(held.get('deployed'))[:12]}"


# `busy(metrics)` -- what the engine still has outstanding, or None when it did not answer in a way we
# understand -- and `unsupported(metrics)` are fleet_lease.door_load / door_unsupported (imported above).
# `st:quiet` is the engine's OWN answer and is the one that counts; None is not zero: an engine that
# cannot be asked is not known to be quiet, and this refuses to deploy on top of a question it could
# not get an answer to -- including one too old to publish `st:quiet`, which `unsupported` names.


def fleet_taken_by_another(log) -> bool:
    """A live fleet lease of any kind but production is a window somebody was granted.

    The queue takes the lease for a ticket's boot and a session's boot holds one of its own; deploying
    over either would be the launcher's `stop` evicting somebody else's boot (it refuses now, but this
    should not even try). Deferred, not rejected: nothing is recorded, the next cycle looks again.
    A lease this cannot read is not free (D3).
    """
    try:
        who = fleet_lease.taken(fleet_lease.read(LOCK), mine_kind="production",
                                container_up=fleet_lease.docker_evidence)
    except LeaseHeld as exc:
        log(f"  the fleet lease could not be read ({exc}); not deploying over a question")
        return True
    if who:
        log(f"  the fleet is held by {who}: deferring the deploy to the next cycle")
        return True
    return False


def failures(tree: Path, timeout: int) -> "dict[str, str]":
    """{test file: its one-line verdict} for the files that do not pass, over `tree`."""
    out = {}
    for path in sorted((tree / "tests").glob("test_engine_*.py")):
        name = path.stem
        # niced: this runs on rank 0's node, beside the door and the scheduler, while they serve
        code, stdout, stderr = run(["nice", "-n", "19", sys.executable, "-m", "unittest", f"tests.{name}"],
                                   cwd=tree, timeout=timeout)
        tail = (stdout + stderr).strip().splitlines()
        verdict = next((line for line in reversed(tail) if line.startswith(("OK", "FAILED"))), "NO VERDICT")
        if not verdict.startswith("OK"):
            out[name] = re.sub(r"id='\d+'", "id=..", verdict)
    return out


def regressed(deployed: "dict[str, str]", candidate: "dict[str, str]") -> "list[str]":
    """The files that fail on the candidate and did not, or fail differently than, on what is deployed."""
    return sorted(name for name, verdict in candidate.items() if deployed.get(name) != verdict)


# -- the actions ----------------------------------------------------------------------------------
def cut(sha: str, log) -> "Path | None":
    """A release directory for `sha`: a checkout, not a copy of a working tree that may be mid-edit.

    The cut itself lives in launchers/st_release.py, shared with bench/st_bracket.sh: one shape of
    release, so a bracket's winning arm is promoted by pointing production at the directory the
    bracket already booted.
    """
    return st_release.cut(sha, source=SOURCE, releases=RELEASES, log=log, meta=HOME / "st-engine" / "st-glm53-meta")


def deploy(release: Path, log) -> bool:
    """Stop the supervisor, relaunch from `release`, start the supervisor again.

    The supervisor owns the fleet lock while it runs, so it has to be out of the way before the
    launcher touches the nodes -- otherwise its health loop relaunches the old tree underneath this.
    """
    launcher = release / "launchers" / "start-st-glm53.sh"
    if not launcher.exists():
        log(f"  ABORT: {launcher} is missing")
        return False
    # Its boots are production's: the lease they take is kind `production`, and `stop` is judged
    # by that kind too (the supervisor's pid, or this one's, does not matter across restarts).
    env = {"REPO": str(release), "ST_LEASE_KIND": "production",
           "LEASE_OWNER_PRODUCTION": f"production/deploy/{os.getpid()}"}
    run(["systemctl", "--user", "stop", SERVICE], timeout=300)
    code, out, err = run(["bash", str(launcher), "stop"], timeout=600, env=env)
    log(f"  stop: rc={code} {out.strip().splitlines()[-1] if out.strip() else ''}")
    code, out, err = run(["bash", str(launcher), "start"], timeout=3600, env=env)
    for line in (out + err).strip().splitlines()[-6:]:
        log(f"  {line}")
    if code:
        log(f"  the launch failed (rc={code}); the supervisor takes it from here")
    run(["systemctl", "--user", "start", SERVICE], timeout=300)
    return code == 0


# -- one cycle ------------------------------------------------------------------------------------
def cycle(a, log) -> int:
    held = state_of(STATE)
    run(["git", "-C", str(SOURCE), "fetch", "origin", "--quiet"], timeout=300)
    _, head, _ = run(["git", "-C", str(SOURCE), "rev-parse", "origin/main"], timeout=60)
    head = head.strip()
    why = wanted(head, held)
    if why is None:
        log(f"nothing to deploy (main {head[:12]}, deployed {str(held.get('deployed'))[:12]})")
        ensure_probe(held.get("deployed") or "", held, a, log)      # the deployed commit keeps its warm sample
        return 0
    log(f"candidate: {why}")

    since = time.time() - held.get("deployed_at", 0)
    if since < a.min_gap:
        log(f"  too soon: {int(since)}s since the last deploy, {a.min_gap}s asked for")
        return 0

    log(f"  waiting for {a.quiet}s of quiet")
    quiet_since = None
    deadline = time.time() + a.wait
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
                body = r.read().decode()
            load = busy(body)
            if load is None and unsupported(body):
                log("  the engine answered, but it is older than st:quiet: it cannot say whether a tier")
                log("  transfer is in flight, and this will not take down an engine that cannot say.")
                log(f"  Move it forward once by hand, then --seed: the metric arrives with the tree.")
                return 0
        except Exception as exc:                                     # noqa: BLE001 -- any failure is "not known to be quiet"
            load = None
            log(f"  /metrics did not answer ({type(exc).__name__}); not treating that as quiet")
        if load:
            quiet_since = None
        elif load == 0:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= a.quiet:
                break
        if a.once and a.dry_run:
            break
        time.sleep(a.poll)
    else:
        log(f"  never went quiet within {a.wait}s: leaving it alone")
        return 0
    if a.dry_run:
        log(f"  DRY RUN: would cut and gate {head[:12]}" + ("" if quiet_since else " (and it is not quiet)"))
        return 0
    if quiet_since is None:
        log("  not quiet: leaving it alone")
        return 0

    release = cut(head, log)
    if release is None:
        return 1
    deployed_tree = Path(held["release"]) if held.get("release") and Path(held["release"]).exists() else None
    if a.gate and deployed_tree is not None:
        log("  gate: the engine suite over the candidate and over what is deployed")
        after = failures(release, a.test_timeout)
        before = failures(deployed_tree, a.test_timeout)
        worse = regressed(before, after)
        if worse:
            log(f"  REFUSED: {len(worse)} file(s) regressed against {deployed_tree.name}: {', '.join(worse)}")
            for name in worse:
                log(f"    {name}: deployed {before.get(name, 'OK')!r} -> candidate {after[name]!r}")
            STATE.write_text(json.dumps({**held, "rejected": head, "rejected_at": time.time(),
                                         "rejected_files": worse}, indent=1))
            return 1
        log(f"  gate passed: {len(after)} file(s) fail on both, none newly")
    elif a.gate:
        # The gate is a DIFFERENCE, so it needs the tree that is running. Without one there is
        # nothing to be a difference from, and skipping would make the first deploy -- the one
        # nobody has watched -- the only ungated one. Refuse and say how to seed it.
        log("  REFUSED: nothing recorded as deployed to compare against.")
        log(f"    seed it with:  python3 {Path(__file__).name} --seed <the sha that is running>")
        log("    (or --no-gate, which is a different decision)")
        return 1

    if fleet_taken_by_another(log):
        return 0                                       # deferred, not rejected: main has not moved past it
    log(f"  deploying {head[:12]}")
    ok = deploy(release, log)
    if not ok:
        # Not recorded as deployed: what is serving now is whatever the supervisor recovered, which
        # is not this release, and the next gate must not take it as the baseline. Recorded as
        # rejected so the next cycle does not walk straight back into the same launch.
        # And the baseline goes with it. `start-st-glm53.sh` rsyncs the candidate to all four nodes
        # BEFORE the step that failed, so what the supervisor recovered onto is the candidate if the
        # rsync got that far and the old tree if it did not -- nobody knows which. A gate run against
        # a guess is worse than no gate, so the next cycle refuses until a person says what is up.
        STATE.write_text(json.dumps({**held, "release": None, "rejected": head,
                                     "rejected_at": time.time(), "rejected_by": "launch"}, indent=1))
        log(f"  {head[:12]} did not launch; not recorded as deployed, and the gate's baseline is dropped")
        log(f"  (the tree on the nodes is no longer known: --seed once someone has looked)")
        return 1
    STATE.write_text(json.dumps({"deployed": head, "release": str(release), "deployed_at": time.time(),
                                 "launched_ok": True}, indent=1))
    log(f"  deployed {head[:12]} from {release}")
    after_deploy(head, a, log)
    return 0


# -- after a deploy: the queue follows ------------------------------------------------------------
CONTROLLER = Path(os.environ.get("FLEET_CONTROLLER_REPO", HOME / "fleet-controller"))   # the queue's own checkout on the head


def follow_controller(head: str, log, controller: Path = CONTROLLER) -> bool:
    """The queue's checkout moves to the deployed commit, so the queue answers by production's rules.

    A controller 639 commits behind main once told a session the fleet was FREE with four nodes
    serving (45차 §91). Waiting tickets are unaffected: they run out of pinned runner snapshots.
    """
    if not (controller / ".git").exists():
        log(f"  controller {controller} is not a checkout; the queue's rules stay where they are")
        return False
    code, _, err = run(["git", "-C", str(controller), "fetch", "--quiet", "origin", head], timeout=600)
    if code:                                           # a remote that refuses a bare sha still serves its refs
        code, _, err = run(["git", "-C", str(controller), "fetch", "--quiet", "origin"], timeout=600)
    if code:
        log(f"  controller: could not fetch {head[:12]} ({err.strip()[:80]})")
        return False
    code, _, err = run(["git", "-C", str(controller), "checkout", "--quiet", "--detach", head], timeout=120)
    if code:
        log(f"  controller: could not move to {head[:12]} ({err.strip()[:80]})")
        return False
    log(f"  controller {controller} now at {head[:12]}")
    return True


def queue_probe(head: str, log, controller: Path = CONTROLLER) -> bool:
    """One D17 probe ticket for the deployed commit: two onepass runs on the live door when it is
    idle, so the deployed commit always has a warm sample and st-pair never boots the base."""
    fleet = controller / "bench" / "fleet.sh"
    if not fleet.exists():
        log(f"  no {fleet}: no D17 probe queued")
        return False
    session = "d17-" + head[:12]
    code, out, err = run(["bash", str(fleet), "st-probe", "--detach", session, head, "10", f"D17 after deploy {head[:12]}"],
                         cwd=str(controller), timeout=300)
    tail = (out + err).strip().splitlines()[-1:] or [""]
    if code:
        log(f"  D17 probe {session} was not queued (rc={code}): {tail[0][:120]}")
        return False
    log(f"  D17 probe {session} queued: {tail[0][:120]}")
    return True


FLEET = Path(os.environ.get("FLEET_DIR", HOME / "glm53-logs" / "fleet"))                 # the queue's own files (bench/fleet.sh)
JSONL = Path(os.environ.get("ONEPASS_JSONL", HOME / "glm53-logs" / "bracket-onepass.jsonl"))


def warm_samples(sha: str, log, controller: Path = CONTROLLER, jsonl: Path = JSONL) -> "int | None":
    """How many warm, valid onepass samples the records hold for `sha` -- counted by the
    controller's own judge (bench/st_judge.py), so what a sample is gets decided once. None
    when it cannot be told, and None never queues anything."""
    judge = controller / "bench" / "st_judge.py"
    if not judge.exists():
        log(f"  no {judge}: cannot tell whether {sha[:12]} has a warm sample")
        return None
    code, out, err = run([sys.executable, str(judge), "samples", "--sha", sha, "--jsonl", str(jsonl)], timeout=120)
    if code or not out.strip().isdigit():
        log(f"  st_judge could not count samples for {sha[:12]}: {(err or out).strip()[:120]}")
        return None
    return int(out.strip())


def ticket_open(session: str, fleet_dir: Path = FLEET) -> bool:
    """A ticket of this name is queued or holding, read off the queue's own files."""
    try:
        for line in (fleet_dir / "queue").read_text().splitlines():
            if line.split("|")[1:2] == [session]:
                return True
    except OSError:
        pass
    for name in ("holder", "holder-single"):
        try:
            if (fleet_dir / name).read_text().split("|")[0].strip() == session:
                return True
        except OSError:
            pass
    return False


def remember_probe(sha: str, queued: bool, state: Path = None, tally: dict = None) -> dict:
    """The tally of probe tickets this watcher queued for the deployed sha, in the state file."""
    state = state or STATE
    held = state_of(state)
    tally = tally if tally and tally.get("sha") == sha else held.get("probe") or {}
    if tally.get("sha") != sha:
        tally = {"sha": sha, "attempts": 0}
    tally = {**tally, "attempts": tally["attempts"] + 1, "last_at": time.time(), "queued": queued}
    state.write_text(json.dumps({**held, "probe": tally}, indent=1))
    return tally


def ensure_probe(sha: str, held: dict, a, log, *, controller: Path = None, fleet_dir: Path = None,
                 jsonl: Path = None, state: Path = None) -> bool:
    """A D17 probe ticket for the deployed commit whenever it has no warm sample and none is on
    its way. The ticket after the deploy is the first; this is for the cycles after it, when
    that ticket ran into traffic (onepass --require-exclusive gives up and leaves no record) or
    was cancelled. Bounded: --probe-attempts tickets per deployed sha, --probe-gap apart."""
    if getattr(a, "dry_run", False) or not getattr(a, "probe", True) or not sha:
        return False
    controller = Path(controller or getattr(a, "controller", CONTROLLER))
    fleet_dir = Path(fleet_dir or FLEET)
    jsonl = Path(jsonl or JSONL)
    tally = held.get("probe") or {}
    if tally.get("sha") != sha:
        tally = {"sha": sha, "attempts": 0, "last_at": 0}
    limit = getattr(a, "probe_attempts", 3)
    if tally["attempts"] >= limit:
        if not tally.get("gave_up"):
            log(f"  {sha[:12]} has no warm sample after {tally['attempts']} probe tickets; queuing no more "
                f"(fleet.sh st-probe by hand, or a new deploy)")
            (state or STATE).write_text(json.dumps({**state_of(state or STATE), "probe": {**tally, "gave_up": True}}, indent=1))
        return False
    if time.time() - tally.get("last_at", 0) < getattr(a, "probe_gap", 1800):
        return False
    count = warm_samples(sha, log, controller, jsonl)
    if count is None or count > 0:
        return False
    session = "d17-" + sha[:12]
    if ticket_open(session, fleet_dir):
        return False
    log(f"  {sha[:12]} has no warm sample and no probe ticket on its way (attempt {tally['attempts'] + 1}/{limit})")
    queued = queue_probe(sha, log, controller)
    remember_probe(sha, queued, state, tally)
    return queued


def after_deploy(head: str, a, log) -> None:
    if getattr(a, "dry_run", False):
        return
    controller = Path(getattr(a, "controller", CONTROLLER))
    if getattr(a, "follow", True):
        follow_controller(head, log, controller)
    if getattr(a, "probe", True):
        remember_probe(head, queue_probe(head, log, controller))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="one cycle and exit (what the timer runs)")
    ap.add_argument("--dry-run", action="store_true", help="decide and report, change nothing")
    ap.add_argument("--install", action="store_true", help="print how to arm it; arm nothing")
    ap.add_argument("--seed", metavar="SHA", help="record SHA as what is deployed, without deploying: "
                                                  "the gate needs a tree to be a difference from")
    ap.add_argument("--quiet", type=int, default=120, help="seconds of no requests before deploying")
    ap.add_argument("--wait", type=int, default=3600, help="give up waiting for quiet after this")
    ap.add_argument("--poll", type=int, default=10)
    ap.add_argument("--min-gap", type=int, default=1800, help="never restart more often than this")
    ap.add_argument("--test-timeout", type=int, default=900)
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--interval", type=int, default=300, help="seconds between cycles in the loop")
    ap.add_argument("--controller", default=str(CONTROLLER), help="the queue's checkout: moved to the deployed commit after a deploy")
    ap.add_argument("--no-follow", dest="follow", action="store_false", help="leave the controller checkout where it is")
    ap.add_argument("--no-probe", dest="probe", action="store_false", help="queue no D17 probe ticket after a deploy")
    ap.add_argument("--probe-attempts", type=int, default=3, help="probe tickets per deployed sha before giving up")
    ap.add_argument("--probe-gap", type=int, default=1800, help="seconds between two probe tickets for the same sha")
    a = ap.parse_args(argv)

    def log(line):
        print(f"{time.strftime('%F %T')} {line}", flush=True)

    if a.install:
        unit = Path(__file__).resolve().parent / "st-deploy-watch.service"
        print(f"  # on the head node, after reading {unit.name} and its timer:\n"
              f"  cp {unit} {unit.with_suffix('.timer')} ~/.config/systemd/user/\n"
              f"  systemctl --user daemon-reload && systemctl --user enable --now st-deploy-watch.timer\n"
              f"  # and before that, once, to see what it would do:\n"
              f"  python3 {Path(__file__).resolve()} --once --dry-run")
        return 0
    if a.seed:
        _, sha, _ = run(["git", "-C", str(SOURCE), "rev-parse", a.seed], timeout=60)
        sha = sha.strip()
        if not sha:
            log(f"  {a.seed} is not a revision in {SOURCE}")
            return 1
        release = cut(sha, log)
        if release is None:
            return 1
        STATE.write_text(json.dumps({"deployed": sha, "release": str(release),
                                     "deployed_at": time.time(), "seeded": True}, indent=1))
        log(f"  recorded {sha[:12]} at {release} as deployed; nothing was launched")
        return 0
    if a.once:
        return cycle(a, log)
    while True:
        try:
            cycle(a, log)
        except Exception as exc:                                     # noqa: BLE001 -- a watcher that dies stops watching
            log(f"  cycle failed: {type(exc).__name__}: {exc}")
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
