#!/usr/bin/env python3
"""What would it take to serve THIS checkpoint? -- one command, no profile required.

    python3 tools/onboard.py --ckpt ~/models/<checkpoint>
    python3 tools/onboard.py --config config.json --placement ep --json

The engine's forms are the hardware's and it names no model (CHARTER D5). So this reads a checkpoint the way the
engine would: if a profile claims the config's `model_type`, its own derivation answers; if none does,
engine/base/onboard reads the config generically and says which fields it could NOT settle, with what would settle
each. Then engine/kernels/cells judges every lane of the resulting shape and orders the work, cheapest first.

Nothing here boots, allocates or measures: it reads one config.json and the compiled cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def text_config(path: Path) -> dict:
    """A checkpoint's text config: `text_config` when the file nests it (multimodal checkpoints), else the file."""
    cfg = json.loads(path.read_text())
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if "quantization_config" in cfg and "quantization_config" not in text:
        text = dict(text, quantization_config=cfg["quantization_config"])   # DSv4.1 states it on the outer config
    return text


def run(cfg: dict, *, placement: "str | None", ckpt: "Path | None") -> dict:
    from engine.base import kernel_shape as ks
    from engine.base import onboard
    from engine.kernels import cells

    profile = ks.claims(cfg.get("model_type"))
    if profile is not None and ckpt is not None:
        shape = ks.derive_for(profile, ckpt)
        verdicts = cells.admission(shape)
        return {"door": f"profile {profile}", "model_type": cfg.get("model_type"), "shape": shape,
                "blanks": (), "unsettled": (), "admission": verdicts, "plan": cells.plan(verdicts)}
    reading = onboard.read_config(cfg, placement=placement)
    judged = onboard.judge(reading)
    door = f"profile {profile} (claims this model_type; pass --ckpt to use its derivation)" if profile else "generic"
    return {"door": door, "model_type": reading.model_type, "shape": reading.shape, "blanks": reading.blanks,
            "unsettled": reading.unsettled, "admission": judged["admission"], "plan": judged["plan"]}


def render(result: dict) -> str:
    out = [f"  door: {result['door']}", f"  model_type: {result['model_type'] or '(none declared)'}", ""]
    if result["shape"] is not None:
        out += [f"  {result['shape'].describe()}", ""]
    if result["blanks"]:
        out.append("  no shape -- the config does not settle:")
        out += [f"    - {b}" for b in result["blanks"]]
        out.append("")
        out.append("  a blank is filled by the checkpoint's reference implementation, a profile under "
                   "engine/profiles/, or --placement.")
        return "\n".join(out)
    if result["unsettled"]:
        out.append("  not established (the shape carries it, the lane refuses it by name):")
        out += [f"    - {b}" for b in result["unsettled"]]
        out.append("")
    width = max((len(v.lane) for v in result["admission"]), default=0)
    out.append("  lanes:")
    for v in result["admission"]:
        out.append(f"    {v.lane:<{width}}  {v.status:<10}  {v.why}")
    if result["plan"]:
        out += ["", "  work, cheapest first:"]
        for v in result["plan"]:
            out.append(f"    {v.recipe.cost:<6} {v.lane:<{width}}  {v.recipe.kind:<8} {v.recipe.where}")
    else:
        out += ["", "  no lane asks for work on this shape."]
    return "\n".join(out)


def as_json(result: dict) -> str:
    from engine.base import kernel_shape as ks
    from engine.kernels import cells
    return json.dumps({
        "door": result["door"], "model_type": result["model_type"],
        "shape": None if result["shape"] is None else ks.to_dict(result["shape"]),
        "describe": None if result["shape"] is None else result["shape"].describe(),
        "blanks": [{"field": b.field, "why": b.why} for b in result["blanks"]],
        "unsettled": [{"field": b.field, "why": b.why} for b in result["unsettled"]],
        "admission": cells.to_dicts(result["admission"]) if result["admission"] else [],
        "plan": [v.lane for v in result["plan"]],
    }, indent=2, ensure_ascii=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 tools/onboard.py", description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="a checkpoint directory (reads its config.json)")
    src.add_argument("--config", help="a config.json to read instead of a checkpoint")
    ap.add_argument("--placement", choices=("ep", "tp"), help="expert placement, the operator's choice: whole experts "
                                                              "a rank (ep) or every expert's intermediate sliced (tp)")
    ap.add_argument("--json", action="store_true", help="print one JSON document")
    a = ap.parse_args(argv)
    ckpt = Path(a.ckpt) if a.ckpt else None
    path = (ckpt / "config.json") if ckpt else Path(a.config)
    if not path.is_file():
        print(f"  no config at {path}", file=sys.stderr)
        return 1
    result = run(text_config(path), placement=a.placement, ckpt=ckpt)
    print(as_json(result) if a.json else render(result))
    return 0 if result["shape"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
