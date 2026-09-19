"""Which model production serves: one selection every production path reads (launchers).

Production has always been GLM-5.3, and three scripts each said so on their own: the supervisor
(its container, its launcher, the model its health chat names), deploy-watch (the launcher it
boots a release with) and the prebuild (the profile whose kernels it replays). CHARTER D5 stopped
naming models on 2026-09-19 -- GLM-5.3 and Qwen3.8-Flash-Next are the fleet's present, not its
bound -- and the operator asked to choose the engine's model from Deneb the same day. So which
model production serves is now one file, and those paths read it instead of their constant:

    /home/choiceoh/glm53-logs/st-production.json   {"profile": "qwen38", "by": "...", "at": ..., "note": "..."}

No file is glm53: every box that never chose serves exactly what it served before. Whoever
chooses writes it -- the Deneb app, an operator with a shell (`select`) -- and the supervisor
does the switch: it waits for the door to be quiet, stops the fleet it runs, boots the chosen
profile under the production lease, and says what it is doing in a second file:

    /home/choiceoh/glm53-logs/st-production-state.json   {"serving", "wanted", "phase", "detail", "at", "profiles"}

A profile names what differs between models and nothing else. Production's tree and image stay
production's whichever model it serves: the ST image carries no model (engine/runtime), and a
release is the whole engine tree, so one deploy moves every profile. What differs is the
launcher, the container, the model id the door answers to, and the launch environment -- which
is why each profile's environment is its own file: `st-glm53.env` is what systemd hands the
supervisor, and its RANKS_DIR would boot Qwen3.8 on GLM's rank files if it leaked into the
other launch. `env <profile>` therefore clears every key any profile sets before it exports its
own.

    python3 launchers/st_production.py selected                  # the profile production serves
    python3 launchers/st_production.py profiles                  # the profiles production can serve
    python3 launchers/st_production.py select qwen38 --by deneb --note "the operator chose it"
    python3 launchers/st_production.py field qwen38 container    # container | model | launcher
    python3 launchers/st_production.py env qwen38                # shell lines for its launch
    python3 launchers/st_production.py state <serving> <wanted> <phase> [detail]
    python3 launchers/st_production.py show                      # selection + state, JSON

Stdlib only, like engine/base/fleet_lease.py: it runs on the head, outside the engine's
virtualenv, and deploy-watch imports it.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LOGS = Path("/home/choiceoh/glm53-logs")


def selection_path() -> Path:
    return Path(os.environ.get("ST_PRODUCTION_FILE", LOGS / "st-production.json"))


def state_path() -> Path:
    return Path(os.environ.get("ST_PRODUCTION_STATE", LOGS / "st-production-state.json"))


def config_dir() -> Path:
    return Path(os.environ.get("ST_PROFILE_CONFIG_DIR", Path.home() / ".config"))


@dataclass(frozen=True)
class Profile:
    model: str                     # the id the door answers to (base/serve.check_model 404s any other)
    container: str                 # one per node; the supervisor's census and forensics read it
    launcher: str                  # under launchers/
    defaults: dict = field(default_factory=dict)   # launch environment when its env file does not say

    def env_file(self, name: str) -> Path:
        return config_dir() / f"st-{name}.env"


PROFILES = {
    "glm53": Profile(model="glm-5.3-flash", container="st-glm53", launcher="start-st-glm53.sh"),
    # Its rank files are the ones the operator's windows boot (start-st-qwen38.sh's default), named
    # here because st-glm53.env's RANKS_DIR is cleared for this launch and nothing else would say.
    "qwen38": Profile(model="qwen3.8-flash-next", container="st-qwen38", launcher="start-st-qwen38.sh",
                      defaults={"RANKS_DIR": "/home/choiceoh/models/st-qwen38-tep4"}),
}
DEFAULT = "glm53"
# Production's, not a model's: the tree and the image a release pins (st-glm53.service's comment in the
# README pins all three to one release). A switch keeps them; every other key a profile sets is cleared.
SHARED = ("ST_REPO", "ST_ENGINE_DIR", "ST_IMAGE")


def read_env_file(path: Path) -> dict:
    """KEY=VALUE lines the way systemd's EnvironmentFile reads them: comments and blanks skipped,
    one pair of matching quotes stripped. Missing or unreadable is empty -- a profile with no file
    launches on its defaults."""
    out = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key.isidentifier():
            out[key] = value
    return out


def _profile_env(name: str) -> dict:
    profile = PROFILES[name]
    return {**profile.defaults, **read_env_file(profile.env_file(name))}   # the file wins


def _model_keys() -> set:
    """Every key some profile sets, less the production-wide ones: what a switch clears."""
    keys = set()
    for other in PROFILES:
        keys.update(_profile_env(other))
    return keys - set(SHARED)


def launch_env(name: str, base: "dict | None" = None) -> dict:
    """The environment `name`'s launcher runs in: `base` without any model's keys, then this
    profile's defaults and env file."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in _model_keys()}
    env.update(_profile_env(name))
    return env


def env_lines(name: str) -> str:
    """`launch_env` as shell: unset another model's keys, export this one's."""
    mine = _profile_env(name)
    lines = [f"unset {key}" for key in sorted(_model_keys() - set(mine))]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in mine.items()]
    return "\n".join(lines)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    """Whole or not at all: a reader on another box (Deneb reads these over ssh) never sees half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def selected() -> str:
    """The profile production serves. No file, an unreadable one or an unknown name is DEFAULT:
    a typo must not take production down, and `show` says what the file actually held."""
    name = _read_json(selection_path()).get("profile")
    return name if name in PROFILES else DEFAULT


def select(name: str, by: str = "", note: str = "") -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r} (known: {', '.join(PROFILES)})")
    record = {"profile": name, "by": by or f"{os.environ.get('USER', '?')}@{os.uname().nodename}",
              "at": time.time(), "note": note}
    _write_json(selection_path(), record)
    return record


def publish_state(serving: str, wanted: str, phase: str, detail: str = "") -> dict:
    """What the supervisor is doing about the selection, for whoever chose it. `profiles` rides
    along so a reader learns which models production can serve from the box that serves them."""
    record = {"serving": serving if serving in PROFILES else "", "wanted": wanted, "phase": phase,
              "detail": detail, "at": time.time(), "pid": os.getppid(),
              "profiles": {n: {"model": p.model, "container": p.container} for n, p in PROFILES.items()}}
    _write_json(state_path(), record)
    return record


def show() -> dict:
    return {"selected": selected(), "selection": _read_json(selection_path()), "state": _read_json(state_path()),
            "profiles": {n: {"model": p.model, "container": p.container} for n, p in PROFILES.items()},
            "default": DEFAULT}


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="which model production serves")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("selected")
    sub.add_parser("profiles")
    s = sub.add_parser("select")
    s.add_argument("profile")
    s.add_argument("--by", default="")
    s.add_argument("--note", default="")
    f = sub.add_parser("field")
    f.add_argument("profile")
    f.add_argument("name", choices=("container", "model", "launcher"))
    e = sub.add_parser("env")
    e.add_argument("profile")
    st = sub.add_parser("state")
    st.add_argument("serving")
    st.add_argument("wanted")
    st.add_argument("phase")
    st.add_argument("detail", nargs="?", default="")
    sub.add_parser("show")
    a = ap.parse_args(argv)

    if a.action == "selected":
        print(selected())
    elif a.action == "profiles":
        print("\n".join(PROFILES))
    elif a.action == "select":
        try:
            print(json.dumps(select(a.profile, a.by, a.note), ensure_ascii=False))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    elif a.action == "field":
        if a.profile not in PROFILES:
            print(f"unknown profile {a.profile!r}", file=sys.stderr)
            return 2
        print(getattr(PROFILES[a.profile], a.name))
    elif a.action == "env":
        if a.profile not in PROFILES:
            print(f"unknown profile {a.profile!r}", file=sys.stderr)
            return 2
        print(env_lines(a.profile))
    elif a.action == "state":
        publish_state(a.serving, a.wanted, a.phase, a.detail)
    elif a.action == "show":
        print(json.dumps(show(), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
