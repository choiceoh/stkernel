"""Facts in, knobs that expire, nothing else (base).

D11's evidence was 133 `VLLM_GLM53_*` knobs, one of which sat flipped against
the ledger's own verdict for two campaigns, costing 4.89 GiB nobody knew
about. The rule that follows: the only inputs are FACTS (model path, world
size, ports) and EXPERIMENT KNOBS that carry an expiry, a measurement they
exist for, and a rollback. A promoted knob is deleted; an expired knob kills
the boot; an environment variable nobody declared kills the boot.

Profiles declare facts and knobs. This file only enforces.
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass

PREFIX = "STK_"                        # the engine's own namespace, nothing else is read


@dataclass(frozen=True)
class Fact:
    name: str
    value: object
    source: str                        # where a reader can verify it


@dataclass(frozen=True)
class Knob:
    name: str
    default: object
    expires: _dt.date
    measures: str                      # what campaign this knob exists to measure
    rollback: str                      # what to do when it loses
    parse: type = str


class ConfigError(SystemExit):
    """Boot dies. Loudly, with the reason as the exit message (D3)."""


class Config:
    def __init__(self, facts: "list[Fact]", knobs: "list[Knob]", env: "dict | None" = None,
                 today: "_dt.date | None" = None):
        env = dict(os.environ if env is None else env)
        today = today or _dt.date.today()
        self.facts = {f.name: f for f in facts}
        self.knobs = {k.name: k for k in knobs}
        self.values = {}

        dup = set(self.facts) & set(self.knobs)
        if dup:
            raise ConfigError(f"config: {sorted(dup)} declared as both fact and knob")

        expired = [k for k in knobs if k.expires < today]
        if expired:
            raise ConfigError(
                "config: expired knob(s) still declared -- promote or delete them: "
                + ", ".join(f"{k.name} (expired {k.expires}, measures {k.measures!r})" for k in expired))

        declared = {PREFIX + k.name for k in knobs}
        unknown = sorted(v for v in env if v.startswith(PREFIX) and v not in declared)
        if unknown:
            facts_hit = [v for v in unknown if v[len(PREFIX):] in self.facts]
            why = " (facts cannot be overridden)" if facts_hit else ""
            raise ConfigError(f"config: undeclared {PREFIX}* in the environment: {unknown}{why}")

        for f in facts:
            self.values[f.name] = f.value
        for k in knobs:
            raw = env.get(PREFIX + k.name)
            self.values[k.name] = k.default if raw is None else k.parse(raw)
        self.overridden = sorted(k.name for k in knobs if PREFIX + k.name in env)

    def __getitem__(self, name: str):
        return self.values[name]

    def table(self) -> str:
        out = ["  facts"]
        for f in self.facts.values():
            out.append(f"    {f.name:<24} {str(f.value):<32} [{f.source}]")
        out.append("  knobs (every one expires)")
        for k in self.knobs.values():
            mark = " <- env" if k.name in self.overridden else ""
            out.append(f"    {k.name:<24} {str(self.values[k.name]):<32} until {k.expires}  "
                       f"measures {k.measures!r}, rollback {k.rollback!r}{mark}")
        return "\n".join(out)


def _selfcheck() -> None:
    today = _dt.date(2026, 9, 11)
    facts = [Fact("model_path", "/models/x", "profile"), Fact("world_size", 4, "fleet")]
    live = Knob("sched_max_wait_s", 20.0, _dt.date(2026, 9, 30), "DF5 starvation valve", "unset", float)
    dead = Knob("old_lever", 1, _dt.date(2026, 9, 1), "38th b12x cell", "unset", int)
    c = Config(facts, [live], env={}, today=today)
    assert c["world_size"] == 4 and c["sched_max_wait_s"] == 20.0 and c.overridden == []
    c = Config(facts, [live], env={"STK_sched_max_wait_s": "5"}, today=today)
    assert c["sched_max_wait_s"] == 5.0 and c.overridden == ["sched_max_wait_s"]
    for env, knobs, needle in (
        ({}, [live, dead], "expired"),
        ({"STK_nope": "1"}, [live], "undeclared"),
        ({"STK_world_size": "8"}, [live], "facts cannot be overridden"),
    ):
        try:
            Config(facts, knobs, env=env, today=today); raise AssertionError(f"should have died: {needle}")
        except ConfigError as e:
            assert needle in str(e), (needle, str(e))
    print("  config: facts fixed, knobs expire, undeclared env dies OK")


if __name__ == "__main__":
    _selfcheck()
