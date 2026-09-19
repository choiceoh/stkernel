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
   absolute count is not a signal and a green bar is not available; the delta is. Each tree's files
   run inside the seed image that tree pins (engine/runtime/dependencies.json), CUDA hidden -- the
   libraries it would boot with, not the head's python, which has no torch.

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
import st_production                                                                 # noqa: E402  the model production serves


def run(cmd, cwd=None, timeout=1800, env=None, base=None):
    """`env` on top of `base` (this process's environment when None): a launch of another model's
    profile passes a base without st-glm53.env's model keys (st_production.launch_env)."""
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                         env={**(os.environ if base is None else base), **(env or {})})
    return out.returncode, out.stdout, out.stderr


def production_switching() -> "str | None":
    """Why production is between models right now, or None when it serves the selected one.

    A deploy boots the SELECTED model's launcher. While the supervisor is still moving the fleet to
    it, the other model's containers are up, that boot refuses on them, and the deploy would be
    recorded as a launch that failed -- which drops the gate's baseline and needs a person to --seed.
    No state file is a supervisor from before the selection: glm53, exactly as before.
    """
    view = st_production.show()
    selected, state = view["selected"], view["state"]
    if not state:
        return None if selected == st_production.DEFAULT else f"{selected} is selected and no supervisor says it serves it"
    if state.get("serving") != selected:
        return f"{selected} is selected, {state.get('serving') or 'nothing'} serves ({state.get('phase') or '?'})"
    return None


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


def same_engine(head: str, deployed: str) -> bool:
    """Whether `head` serves the engine already deployed: the same engine/ tree under another commit."""
    if not head or not deployed or head == deployed:
        return False
    a, b = engine_tree(head), engine_tree(deployed)
    return bool(a) and a == b


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


def pace_grace(fleet_dir: "Path | None" = None, now: "float | None" = None) -> int:
    """The queue's own grace (bench/fleet_pace.py writes restore-grace.json after every release; a
    session's `fleet.sh window` writes window.json), floor 300 s: the same number the production
    supervisor waits for, so a deploy is never quicker to take the fleet than production is."""
    d, now = Path(fleet_dir or FLEET), time.time() if now is None else now
    grace, window = 300, 0
    try:
        grace = int(json.loads((d / "restore-grace.json").read_text()).get("seconds", 300))
    except (OSError, ValueError, TypeError):
        pass
    try:
        window = max(0, int(json.loads((d / "window.json").read_text()).get("until", 0) - now))
    except (OSError, ValueError, TypeError):
        pass
    return max(grace, window, 300)


def queue_active_within(seconds: "float | None", fleet_dir: "Path | None" = None) -> "int | None":
    """Seconds since the queue's activity clock moved (bench/fleet_idle.py: enqueue, grant, release)
    when that is under `seconds`, else None. A queue that just released a ticket is likely to have
    its next one on its way; a deploy in that window takes the fleet from it, and the supervisor
    keeps the same grace before restoring production (ST_RESTORE_GRACE_S)."""
    if seconds is None:
        seconds = pace_grace(fleet_dir)
    try:
        stamp = json.loads((Path(fleet_dir or FLEET) / "idle-recovery.json").read_text()).get("updated_at")
        ago = int(time.time() - float(stamp))
    except (OSError, ValueError, TypeError):
        return None
    return ago if ago < seconds else None


def fleet_busy_with_tickets() -> bool:
    """The queue owns the fleet right now, or is about to: a lease of any kind but production, or a boot ticket waiting."""
    try:
        who = fleet_lease.taken(fleet_lease.read(LOCK), mine_kind="production", container_up=fleet_lease.docker_evidence)
    except LeaseHeld:
        return True
    return bool(who) or boot_ticket_waiting() is not None


def boot_ticket_waiting(fleet_dir: "Path | None" = None) -> "str | None":
    """The first boot ticket queued (bench/fleet.sh's queue file, kind in field 6), or None. A deploy
    is a production boot, and production comes back only when no boot ticket waits (the operator's
    rule for the queue, 2026-09-12): a deploy that takes the fleet from a waiting ticket makes it
    wait through a boot, the quiet gate and a drain -- 05:03-05:2x on 2026-09-13 would have."""
    try:
        for line in (Path(fleet_dir or FLEET) / "queue").read_text().splitlines():
            fields = line.split("|")
            if len(fields) > 5 and fields[5].strip() in ("boot", ""):
                return fields[1].strip()
    except OSError:
        pass
    return None


# The gate's container: one per tree, the files in parallel inside it. It runs on rank 0's node beside the
# door and the scheduler while they serve, so it is niced, CPU- and memory-bounded, and has no network or GPU.
GATE_WORKERS = 4
GATE_CPUS = "4"
GATE_MEMORY = "6g"
GATE_MARK = "ST_GATE "
GATE_DRIVER = r"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.nice(19)
timeout, workers = int(sys.argv[1]), int(sys.argv[2])
def verdict(name):
    try:
        done = subprocess.run([sys.executable, "-m", "unittest", "tests." + name], capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return name, "NO VERDICT (timed out)"
    tail = (done.stdout + done.stderr).strip().splitlines()
    return name, next((line for line in reversed(tail) if line.startswith(("OK", "FAILED"))), "NO VERDICT")
names = sorted(path.stem for path in Path("tests").glob("test_engine_*.py"))
with ThreadPoolExecutor(max_workers=workers) as pool:
    print("ST_GATE " + json.dumps(dict(pool.map(verdict, names))), flush=True)
"""


def gate_image(tree: Path) -> str:
    """The seed image `tree` pins: its tests run on the libraries it would boot with."""
    return json.loads((tree / "engine/runtime/dependencies.json").read_text())["seed_image_id"]


def failures(tree: Path, timeout: int) -> "dict[str, str]":
    """{test file: its one-line verdict} for the files that do not pass, over `tree`.

    The files run in the tree's own seed image with CUDA hidden, one container per tree. They used to
    run under the head's python, which has no torch: every file that imports it failed, and the gate
    counted 76 "regressions" against the 09-13 tree and refused every deploy (2026-09-15). A container
    that cannot say anything -- no pinned seed, the image not on this node, docker failing -- leaves
    every file of the tree without a verdict, and the gate refuses on that rather than passing.
    """
    names = sorted(path.stem for path in (tree / "tests").glob("test_engine_*.py"))

    def unknown(reason):
        return {name: f"NO VERDICT ({reason})" for name in names}

    try:
        image = gate_image(tree)
    except (OSError, KeyError, ValueError) as exc:
        return unknown(f"no seed image pinned: {type(exc).__name__}")
    cmd = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--cpus", GATE_CPUS,
           "--memory", GATE_MEMORY, "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
           "-e", "PYTHONPATH=/repo", "-v", f"{tree}:/repo:ro", "-w", "/repo", "--entrypoint", "python3", image,
           "-c", GATE_DRIVER, str(timeout), str(GATE_WORKERS)]
    try:
        code, stdout, stderr = run(cmd, timeout=max(3600, 4 * timeout))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return unknown(f"gate container: {type(exc).__name__}")
    line = next((line for line in reversed(stdout.splitlines()) if line.startswith(GATE_MARK)), None)
    if line is None:
        said = (stderr or stdout).strip().splitlines()
        return unknown(f"gate container rc={code}: {said[-1][:160] if said else 'no output'}")
    verdicts = json.loads(line[len(GATE_MARK):])
    out = {}
    for name in names:
        verdict = verdicts.get(name, "NO VERDICT")
        if not verdict.startswith("OK"):
            out[name] = re.sub(r"id='\d+'", "id=..", verdict)
    return out


def cached_baseline(held: dict, deployed_tree: Path, log) -> "tuple[dict | None, str]":
    """The deployed commit's verdicts from the cycle that deployed it, or None to run them again.

    Three things have to still be true, and each one that is not says so in the log rather than being
    silently ignored: the cache is for the sha that is deployed, it was taken on the seed image that
    tree pins, and it is a mapping of verdicts rather than whatever a half-written state file holds.
    """
    cache = held.get("gate")
    if not isinstance(cache, dict) or not isinstance(cache.get("verdicts"), dict):
        return None, "no baseline recorded"
    if cache.get("sha") != held.get("deployed"):
        return None, f"the baseline is for {str(cache.get('sha'))[:12]}"
    try:
        image = gate_image(deployed_tree)
    except (OSError, ValueError, KeyError) as exc:                  # noqa: BLE001 -- unreadable pin: run it
        return None, f"the deployed tree's seed image is unreadable ({type(exc).__name__})"
    if cache.get("image") != image:
        return None, "the seed image moved"
    log(f"    baseline: the {len(cache['verdicts'])} verdicts this watcher took on "
        f"{str(cache['sha'])[:12]} when it deployed it; not re-run")
    return dict(cache["verdicts"]), "cached"


def regressed(deployed: "dict[str, str]", candidate: "dict[str, str]") -> "list[str]":
    """The files that fail on the candidate and did not, or fail differently than, on what is deployed."""
    return sorted(name for name, verdict in candidate.items() if deployed.get(name) != verdict)


# -- the actions ----------------------------------------------------------------------------------
GATE_TREES = RELEASES / "gate"


def gate_tree(sha: str, log) -> "Path | None":
    """The whole commit `sha`, for the gate to judge: extracted once, next to the releases.

    A release carries only what boots (CARRY), and the engine tests also read bench/, tools/ and
    measurements/ beside it. Judged over release trees, seven files that pass on their commit failed
    for a missing file, and the gate refused main for it (2026-09-15)."""
    target = GATE_TREES / sha[:12]
    if target.is_dir():
        return target
    staging = target.with_suffix(".partial")
    run(["rm", "-rf", str(staging)])
    staging.mkdir(parents=True)
    code, _, err = run(["bash", "-c", f"set -o pipefail; git -C {SOURCE} archive {sha} | tar -x -C {staging}"])
    if code:
        log(f"  gate: cannot extract {sha[:12]} ({err.strip()[:80]})")
        run(["rm", "-rf", str(staging)])
        return None
    staging.rename(target)
    return target


def prune_gate_trees(keep) -> None:
    """Only the trees a next cycle can compare stay: the candidate's and the deployed commit's."""
    names = {sha[:12] for sha in keep if sha}
    if GATE_TREES.is_dir():
        for path in GATE_TREES.iterdir():
            if path.name not in names:
                run(["rm", "-rf", str(path)])


def cut(sha: str, log) -> "Path | None":
    """A release directory for `sha`: a checkout, not a copy of a working tree that may be mid-edit.

    The cut itself lives in launchers/st_release.py, shared with bench/st_bracket.sh: one shape of
    release, so a bracket's winning arm is promoted by pointing production at the directory the
    bracket already booted.
    """
    return st_release.cut(sha, source=SOURCE, releases=RELEASES, log=log, meta=HOME / "st-engine" / "st-glm53-meta")


PREBUILD_TIMEOUT = 1800


def prebuild(release: Path, log, profile: str = st_production.DEFAULT) -> None:
    """The release's b12x MoE kernels, compiled on every node's CPU while the deployed tree still serves.

    A release that changes any of the dispatcher's key files used to compile its kernels inside the deploy's boot,
    which is production's downtime: six of them put the door at 150 s instead of 105 s (2026-09-18,
    measurements/qwen38_boot_20260918). `launchers/b12x-prebuild.sh` replays what production's last boots asked for
    (engine/kernels/b12x_requests.py) into each node's /cache, where the boot then finds them. Best effort by design:
    a prebuild that fails, times out or skips a short node leaves that boot to compile, as it always did.
    """
    script = release / "launchers" / "b12x-prebuild.sh"
    if not script.exists():
        log("  prebuild: the release has no launchers/b12x-prebuild.sh; its boot compiles what it needs")
        return
    started = time.time()
    try:
        code, out, err = run(["bash", str(script), "--tree", str(release), "--profile", profile], timeout=PREBUILD_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"  prebuild: {type(exc).__name__}; the boot compiles what it needs")
        return
    for line in (out + err).strip().splitlines()[-8:]:
        log(f"  prebuild: {line}")
    log(f"  prebuild: rc={code} in {time.time() - started:.0f}s")


def deploy(release: Path, log, profile: str = st_production.DEFAULT) -> bool:
    """Stop the supervisor, relaunch from `release` as the model production serves, start the supervisor again.

    The supervisor owns the fleet lock while it runs, so it has to be out of the way before the
    launcher touches the nodes -- otherwise its health loop relaunches the old tree underneath this.
    """
    launcher = release / "launchers" / st_production.PROFILES[profile].launcher
    if not launcher.exists():
        log(f"  ABORT: {launcher} is missing")
        return False
    # Its boots are production's: the lease they take is kind `production`, and `stop` is judged
    # by that kind too (the supervisor's pid, or this one's, does not matter across restarts).
    env = {"REPO": str(release), "ST_LEASE_KIND": "production",
           "LEASE_OWNER_PRODUCTION": f"production/deploy/{os.getpid()}"}
    base = st_production.launch_env(profile)     # st-glm53.env's model keys never reach another model's boot
    run(["systemctl", "--user", "stop", SERVICE], timeout=300)
    code, out, err = run(["bash", str(launcher), "stop"], timeout=600, env=env, base=base)
    log(f"  stop: rc={code} {out.strip().splitlines()[-1] if out.strip() else ''}")
    code, out, err = run(["bash", str(launcher), "start"], timeout=3600, env=env, base=base)
    for line in (out + err).strip().splitlines()[-6:]:
        log(f"  {line}")
    if code:
        log(f"  the launch failed (rc={code}); the supervisor takes it from here")
    run(["systemctl", "--user", "start", SERVICE], timeout=300)
    return code == 0


# -- one cycle ------------------------------------------------------------------------------------
def cycle(a, log) -> int:
    held = state_of(STATE)
    # Which model a deploy boots (launchers/st_production.py). The D17 samples are GLM-5.3's series:
    # a commit sampled while production serves another model would file that model's speed under it.
    profile = st_production.selected()
    if profile == st_production.DEFAULT:
        ensure_probe(held.get("deployed") or "", held, a, log)  # the deployed commit keeps its warm sample, candidate or not
    run(["git", "-C", str(SOURCE), "fetch", "origin", "--quiet"], timeout=300)
    _, head, _ = run(["git", "-C", str(SOURCE), "rev-parse", "origin/main"], timeout=60)
    head = head.strip()
    why = wanted(head, held)
    if why is None:
        log(f"nothing to deploy (main {head[:12]}, deployed {str(held.get('deployed'))[:12]})")
        return 0
    log(f"candidate: {why}")
    if same_engine(head, held.get("deployed") or ""):
        # main moved but engine/ did not (a bench, launcher or docs merge): the engine that serves IS
        # this commit's. Recorded as deployed, the release cut, the controller moved -- and no boot:
        # a restart here costs the fleet three minutes and the door a drain for nothing, and the
        # deployed engine's samples carry over by tree (the operator's baseline rule, 2026-09-13).
        log(f"  {head[:12]} has the deployed engine tree ({engine_tree(head)}): recorded as deployed without a boot")
        if a.dry_run:
            return 0
        release = cut(head, log)
        if release is None:
            return 1
        STATE.write_text(json.dumps({**held, "deployed": head, "release": str(release), "deployed_at": held.get("deployed_at", time.time()),
                                     "same_engine_as": held.get("deployed"), "recorded_at": time.time()}, indent=1))
        if getattr(a, "follow", True):
            follow_controller(head, log, Path(getattr(a, "controller", CONTROLLER)))
        return 0

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
        if fleet_busy_with_tickets():
            log("  a ticket took the fleet, or one waits, while this waited for quiet: leaving it to the queue")
            return 0                                   # deferred: an hour of polling a door the queue owns helps nobody (05:12-06:12, 2026-09-13)
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
        log("  gate: the engine suite over the candidate and over what is deployed, each commit whole")
        judged = gate_tree(head, log)
        if judged is None:
            log("  REFUSED: the gate could not extract both commits; nothing recorded, the next cycle tries again")
            return 1
        after = failures(judged, a.test_timeout)
        # The baseline is a value this watcher has already computed. The deployed commit's verdicts were
        # `after` on the cycle that deployed it -- same tree, same seed image, same box, minutes before --
        # so they are carried in the state file and re-run only when something they depend on moved. That
        # halves the gate: 10m55s, 11m00s and 11m03s on 2026-09-16, of which the baseline was half.
        #
        # Drift is why this is safe to cache rather than dangerous. The baseline exists to SUBTRACT the
        # box's own failures from the candidate's, so a baseline that has gone stale can only make the
        # difference look WORSE -- a refused deploy and a line naming the files, not a bad one waved
        # through. The image id is keyed on because a new seed image changes what the tests import.
        before, why_baseline = cached_baseline(held, deployed_tree, log)
        if before is None:
            log(f"    baseline: running it -- {why_baseline}")
            baseline = gate_tree(held["deployed"], log) if held.get("deployed") else None
            if baseline is None:
                log("  REFUSED: the gate could not extract both commits; nothing recorded, the next cycle tries again")
                return 1
            before = failures(baseline, a.test_timeout)
        prune_gate_trees((head, held.get("deployed")))
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
    ago = queue_active_within(a.queue_grace)
    if ago is not None:
        log(f"  the queue was active {ago}s ago: deferring the deploy until it has been quiet for {a.queue_grace or pace_grace()}s")
        return 0
    waiting = boot_ticket_waiting()
    if waiting:
        log(f"  a boot ticket waits ({waiting}): the queue goes first, the deploy comes when none waits")
        return 0
    switching = production_switching()
    if switching:
        log(f"  production is between models ({switching}): the deploy waits for the supervisor")
        return 0                                       # deferred, not rejected
    if getattr(a, "prebuild", True):
        prebuild(release, log, profile)                # while the deployed tree still serves: no downtime spent
    log(f"  deploying {head[:12]} as {profile}")
    ok = deploy(release, log, profile)
    if not ok:
        # Not recorded as deployed: what is serving now is whatever the supervisor recovered, which
        # is not this release, and the next gate must not take it as the baseline. Recorded as
        # rejected so the next cycle does not walk straight back into the same launch.
        # And the baseline goes with it. `start-st-glm53.sh` rsyncs the candidate to all four nodes
        # BEFORE the step that failed, so what the supervisor recovered onto is the candidate if the
        # rsync got that far and the old tree if it did not -- nobody knows which. A gate run against
        # a guess is worse than no gate, so the next cycle refuses until a person says what is up.
        # The baseline's cache goes with the baseline: if nobody knows which tree is on the nodes,
        # a recorded set of verdicts for one of them is worse than none.
        STATE.write_text(json.dumps({**{k: v for k, v in held.items() if k != "gate"}, "release": None,
                                     "rejected": head, "rejected_at": time.time(),
                                     "rejected_by": "launch"}, indent=1))
        log(f"  {head[:12]} did not launch; not recorded as deployed, and the gate's baseline is dropped")
        log(f"  (the tree on the nodes is no longer known: --seed once someone has looked)")
        return 1
    # What the gate just measured on this tree IS the next cycle's baseline: same commit, same image,
    # same box. Recording it is the whole of the saving -- the next gate runs one suite, not two.
    state = {"deployed": head, "release": str(release), "deployed_at": time.time(), "launched_ok": True}
    if a.gate and deployed_tree is not None and judged is not None:
        try:
            state["gate"] = {"sha": head, "image": gate_image(judged), "verdicts": after, "at": time.time()}
        except (OSError, ValueError, KeyError) as exc:              # noqa: BLE001 -- no cache, so the next gate runs both
            log(f"  the gate's verdicts were not recorded for the next cycle ({type(exc).__name__})")
    STATE.write_text(json.dumps(state, indent=1))
    log(f"  deployed {head[:12]} from {release}")
    after_deploy(head, a, log, profile)
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


def queue_probe(head: str, log, controller: Path = CONTROLLER, attempt: int = 1) -> bool:
    """One D17 probe ticket for the deployed commit: two onepass runs on the live door when it is
    idle, so the deployed commit always has a warm sample and st-pair never boots the base."""
    fleet = controller / "bench" / "fleet.sh"
    if not fleet.exists():
        log(f"  no {fleet}: no D17 probe queued")
        return False
    session = probe_session(head, attempt)
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


def probe_session(sha: str, attempt: int = 1) -> str:
    """d17-<sha12> for the first ticket, d17-<sha12>-2 for the next: the launch layer answers a
    same-name, same-arguments detached launch with the OLD launch's record (disposition
    "existing", its exit code replayed), so a ticket that died or was cancelled can never be
    queued again under its own name -- the second armed cycle on srv2 got rc=143 back that way."""
    return f"d17-{sha[:12]}" + (f"-{attempt}" if attempt > 1 else "")


def engine_tree(sha: str) -> str:
    """The engine/ tree at `sha` (launchers/st_release.py engine_tree): what a sample identifies."""
    try:
        return st_release.engine_tree(sha, source=SOURCE)
    except Exception:                                  # noqa: BLE001 -- no tree is no identity, not a failure
        return ""


def commit_message(sha: str) -> str:
    """The deployed commit's own words, or "" when git cannot be asked (a source tree that moved)."""
    try:
        code, out, _ = run(["git", "log", "-1", "--format=%B", sha], cwd=SOURCE, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out if code == 0 else ""


PROBE_MARK = "D17-probe:"


def probe_wanted(sha: str, log) -> bool:
    """Whether a D17 probe belongs to this commit at all, said once for both callers.

    Two places queue one -- `after_deploy` right after the deploy, and `ensure_probe` on a later cycle
    when the sample is still missing -- and the gate has to be on both or it is on neither.

    A probe reserves the door for a whole C1/C2/32K bracket and answers 409 to every other request
    while it runs; on 2026-09-16 that reached a user as `API error 409` from the assistant. Paying it
    after every deploy buys a baseline for changes that never claimed to move one.
    """
    claims, why = claims_speed(commit_message(sha))
    if not claims:
        log(f"  no D17 probe for {str(sha)[:12]}: {why}")
    return claims


def claims_speed(message: str) -> "tuple[bool, str]":
    """D17's question -- "does this change claim speed?" -- asked of the commit, and why the answer is that.

    The PR template asks it outright, but a squash merge keeps the title and the commit list and drops
    the body, so the answer has to live where git can see it. Two places do:

      the type      `perf:` / `perf(engine):` -- the author saying this change is about speed
      a line        `D17-probe: yes|no` -- for the change that knows better than its type does, in
                    either direction: a `fix:` that moves the step, a `perf:` whose numbers are
                    already in a bracket and needs no second one

    Why this is the gate and not "every deployed commit": a probe reserves the door for the whole of a
    C1/C2/32K bracket, and every request that is not the recording's gets a 409 while it runs. Paying
    that per deploy buys a baseline for changes that never claimed to move one (2026-09-16 operator:
    "커밋당 한번은 너무 많아. 성능 향상을 주장하는 pr이 머지됐을때만 하면 모를까").

    What it costs, said plainly: a `fix:` that quietly regresses the step is no longer sampled, so the
    NEXT `perf:` commit's sample carries that regression and is read against an older baseline. D17's
    own answer to that is the bracket -- a speed claim is measured base->cand->base before it merges,
    not inferred from two production samples.
    """
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith(PROBE_MARK.lower()):
            said = line[len(PROBE_MARK):].strip().lower()
            if said in ("yes", "y", "true", "1"):
                return True, f"{PROBE_MARK} {said}"
            if said in ("no", "n", "false", "0"):
                return False, f"{PROBE_MARK} {said}"
    subject = lines[0] if lines else ""
    kind = subject.split(":", 1)[0].strip().lower() if ":" in subject else ""
    if kind == "perf" or kind.startswith("perf("):
        return True, f"its type is {kind!r}"
    return False, f"it claims no speed ({kind or 'no conventional type'})"


def sample_boots(sha: str, log, controller: Path = CONTROLLER, jsonl: Path = JSONL, tree: str = "") -> "list[str] | None":
    """The production boots that gave `sha` a warm, valid sample -- by the controller's own judge
    (bench/st_judge.py boots), so what a sample is gets decided once. None when it cannot be told,
    and None never queues anything."""
    judge = controller / "bench" / "st_judge.py"
    if not judge.exists():
        log(f"  no {judge}: cannot tell whether {sha[:12]} has a warm sample")
        return None
    code, out, err = run([sys.executable, str(judge), "boots", "--sha", sha, "--jsonl", str(jsonl)] + (["--tree", tree] if tree else []), timeout=120)
    if code:
        log(f"  st_judge could not list samples for {sha[:12]}: {(err or out).strip()[:120]}")
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def production_boot_id() -> "str | None":
    """The head's production container, the way onepass names a boot (docker Id|StartedAt); None when none serves."""
    code, out, _ = run(["docker", "inspect", "-f", "{{.Id}}|{{.State.StartedAt}}", "st-glm53"], timeout=20)
    return out.strip() if code == 0 and "|" in out else None


def ticket_open(session: str, fleet_dir: Path = FLEET) -> bool:
    """A ticket of this name -- or of this name with an attempt suffix -- is queued or holding,
    read off the queue's own files."""
    def mine(name: str) -> bool:
        name = name.strip()
        return name == session or name.startswith(session + "-")
    try:
        for line in (fleet_dir / "queue").read_text().splitlines():
            fields = line.split("|")
            if len(fields) > 1 and mine(fields[1]):
                return True
    except OSError:
        pass
    for name in ("holder", "holder-single", "holder-check"):
        try:
            if mine((fleet_dir / name).read_text().split("|")[0]):
                return True
        except OSError:
            pass
    return False


def remember_probe(sha: str, queued: bool, state: Path = None, tally: dict = None) -> dict:
    """The tally of probe tickets this watcher queued for the deployed sha on the boot it saw, in the state file."""
    state = state or STATE
    held = state_of(state)
    tally = tally if tally and tally.get("sha") == sha else held.get("probe") or {}
    if tally.get("sha") != sha:
        tally = {"sha": sha, "boot": None, "attempts": 0}
    tally = {**tally, "attempts": tally["attempts"] + 1, "last_at": time.time(), "queued": queued}
    state.write_text(json.dumps({**held, "probe": tally}, indent=1))
    return tally


def ensure_probe(sha: str, held: dict, a, log, *, controller: Path = None, fleet_dir: Path = None,
                 jsonl: Path = None, state: Path = None, boot: "str | None" = "?", tree: "str | None" = "?") -> bool:
    """A D17 probe ticket for a deployed commit that CLAIMS SPEED (`claims_speed`) and has fewer
    than --probe-samples (1)
    warm samples -- by commit or by engine tree, so a candidate the bracket measured and main then
    adopted needs none (the operator's rule: its own measurement is the next baseline), and a
    fleet-side merge inherits the sample of the engine it did not touch. Samples count per
    production boot; a boot that already gave its sample is not probed again. Bounded:
    --probe-attempts per boot, --probe-gap apart; nothing is queued while a probe ticket is queued
    or holding, when the judge cannot be asked, or when nothing serves."""
    if getattr(a, "dry_run", False) or not getattr(a, "probe", True) or not sha:
        return False
    if not probe_wanted(sha, log):
        return False
    controller = Path(controller or getattr(a, "controller", CONTROLLER))
    fleet_dir = Path(fleet_dir or FLEET)
    jsonl = Path(jsonl or JSONL)
    tree = engine_tree(sha) if tree == "?" else (tree or "")
    boots = sample_boots(sha, log, controller, jsonl, tree)
    if boots is None:
        return False
    target = getattr(a, "probe_samples", 1)
    if len(boots) >= target:
        return False
    boot = production_boot_id() if boot == "?" else boot
    if not boot:
        return False                                   # nothing serves: nothing to sample until production is back
    if boot in boots:
        return False                                   # this boot gave its sample; the next boot gives the next
    tally = held.get("probe") or {}
    if tally.get("sha") != sha or tally.get("boot") != boot:
        tally = {"sha": sha, "boot": boot, "attempts": 0, "last_at": 0}
    limit = getattr(a, "probe_attempts", 3)
    if tally["attempts"] >= limit:
        if not tally.get("gave_up"):
            log(f"  {sha[:12]}: {len(boots)} of {target} samples, and this boot gave none after {tally['attempts']} probe tickets; "
                f"queuing no more on it (fleet.sh st-probe by hand, or the next boot)")
            (state or STATE).write_text(json.dumps({**state_of(state or STATE), "probe": {**tally, "gave_up": True}}, indent=1))
        return False
    if time.time() - tally.get("last_at", 0) < getattr(a, "probe_gap", 1800):
        return False
    if ticket_open(probe_session(sha), fleet_dir):
        return False
    log(f"  {sha[:12]} has {len(boots)} of {target} warm samples and no probe ticket on its way (attempt {tally['attempts'] + 1}/{limit} on this boot)")
    queued = queue_probe(sha, log, controller, attempt=tally["attempts"] + 1)
    remember_probe(sha, queued, state, tally)
    return queued


def after_deploy(head: str, a, log, profile: str = st_production.DEFAULT) -> None:
    if getattr(a, "dry_run", False):
        return
    controller = Path(getattr(a, "controller", CONTROLLER))
    if getattr(a, "follow", True):
        follow_controller(head, log, controller)
    if profile != st_production.DEFAULT:
        log(f"  no D17 probe: production serves {profile}, and the samples are {st_production.DEFAULT}'s series")
        return
    if getattr(a, "probe", True) and probe_wanted(head, log):
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
    ap.add_argument("--no-prebuild", dest="prebuild", action="store_false",
                    help="deploy without compiling the release's b12x kernels on the nodes' CPUs first")
    ap.add_argument("--interval", type=int, default=300, help="seconds between cycles in the loop")
    ap.add_argument("--controller", default=str(CONTROLLER), help="the queue's checkout: moved to the deployed commit after a deploy")
    ap.add_argument("--no-follow", dest="follow", action="store_false", help="leave the controller checkout where it is")
    ap.add_argument("--no-probe", dest="probe", action="store_false", help="queue no D17 probe ticket after a deploy")
    ap.add_argument("--probe-attempts", type=int, default=3, help="probe tickets per production boot before giving up on that boot")
    ap.add_argument("--probe-samples", type=int, default=1, help="warm samples (by commit or engine tree, one per production boot) the deployed sha is kept at")
    ap.add_argument("--probe-gap", type=int, default=1800, help="seconds between two probe tickets for the same sha")
    ap.add_argument("--queue-grace", type=int, default=None, help="seconds of quiet queue before a deploy takes the fleet (default: the queue's own pace, floor 300)")
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
