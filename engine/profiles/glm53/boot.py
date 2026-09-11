"""Boot GLM-5.3 on the ST engine (profile): facts -> comm -> arena -> loader
-> caches -> lanes -> runner. One function, two comms.

    python3 engine/profiles/glm53/boot.py --local --layers 0-4 --prompt 300 --max-new 8
        four ranks as threads on this box (LocalTP), reference lanes, a layer
        subset -- the whole runner path, no fleet, tokens are not language
    python3 engine/profiles/glm53/boot.py                  (on each of the four nodes, inside the glm53 image)
        the fleet: NCCL comm, served lanes, all 45 layers, then the serve loop

Nothing is discovered: the arena is weights + KV as facts.KV_GIB (the 40th
boot's measured KV), blocks = KV / block_bytes, slots = max_seqs + null.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch                                                     # noqa: E402

from engine.base import scheduler as sched                       # noqa: E402
from engine.base.arena import Arena                              # noqa: E402
from engine.base.comm import Comm, LocalTP                       # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.base.params import total_bytes                       # noqa: E402
from engine.base.record import Ring                              # noqa: E402
from engine.base.runner import STEP_RECORD, Runner               # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.caches import Glm53Caches, block_bytes, slot_bytes   # noqa: E402
from engine.profiles.glm53.engine import Glm53Engine, NullDrafter              # noqa: E402
from engine.profiles.glm53.net import Glm53Net                   # noqa: E402

GIB = 1 << 30
KV_GIB = 8.73                       # the 40th boot's KV (plan.py): what the box has left after weights, runtime floor and activations
TOKEN_BUDGET = 8192                 # MAX_BATCHED: the 6,912 chunk law follows (shapes.py)
MAX_WAIT_S = 20.0                   # D10's one starvation valve
MAX_SEQS = 4                        # launcher MAX_SEQS


def build(comm, layers, lanes, ranks_dir, kv_gib: float, max_seqs: int, drafter, recorder: Recorder,
          max_new: int = 256, temperature: float = 0.0, seed: int = 0):
    F = facts.load()
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    bb, sb = block_bytes(F, net.layers), slot_bytes(F, net.layers, net.Hk)
    ns = max_seqs + 1
    nb = int((kv_gib * GIB - ns * sb) // bb)
    if nb < 2:
        raise MemoryError(f"KV {kv_gib} GiB leaves {nb} blocks after {ns} slots of {sb / 2**20:.0f} MiB")
    with recorder.phase("arena"):
        arena = Arena(total_bytes(specs) + 256 * (len(specs) + 64) + nb * bb + ns * sb)
    with recorder.phase("load"):
        views = RankLoader(Path(ranks_dir) / f"rank{comm.rank}of{facts.TP}.safetensors").load(
            [s.name for s in specs], arena=arena, recorder=recorder)
        net.bind(views)
    caches = Glm53Caches(arena, F, net.layers, net.Hk, nb, ns, max_seqs=ns)
    engine = Glm53Engine(net, caches, F, drafter, max_new=max_new, temperature=temperature, seed=seed)
    contract = sched.Contract(chunk_align=F.block, token_budget=TOKEN_BUDGET, draft_slots=drafter.k,
                              max_wait_s=MAX_WAIT_S, max_running=max_seqs)
    runner = Runner(engine, contract, caches.blocks, caches.slots, Ring(4096, STEP_RECORD.size), recorder)
    recorder.gauge("blocks", nb); recorder.gauge("slots", ns); recorder.gauge("arena_GiB", round(arena.used / GIB, 3))
    return F, net, caches, engine, runner


def run_prompts(engine: Glm53Engine, runner: Runner, prompts: "dict[int, list[int]]"):
    """Submit every prompt, step until idle. Returns {seq: generated ids}."""
    for seq, ids in prompts.items():
        engine.add(seq, ids)
        runner.submit(seq, len(ids))
    out = {seq: None for seq in prompts}
    while runner.step() is not None:
        for seq in list(out):
            if out[seq] is None and seq not in runner.state.running and seq not in runner.state.waiting:
                out[seq] = engine.generated(seq)
    return out


def local(a) -> int:
    print(f"  box: {facts.check_box()}")
    layers = [int(x) for x in a.layers.split("-")]; layers = list(range(layers[0], layers[-1] + 1))
    torch.manual_seed(a.seed)
    prompts = {seq: torch.randint(0, 100_000, (a.prompt + 7 * seq,)).tolist() for seq in range(a.seqs)}
    tp = LocalTP(facts.TP); lane_tables.bind_tp(tp)
    lanes = lane_tables.reference()

    def rank_main(comm):
        rec = Recorder(f"rank{comm.rank}")
        F, net, caches, engine, runner = build(comm, layers, lanes, a.ranks, a.kv_gib, MAX_SEQS, NullDrafter(), rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed)
        t0 = time.perf_counter()
        with rec.phase("generate"):
            out = run_prompts(engine, runner, prompts)
            torch.cuda.synchronize()
        return {"rec": rec, "out": out, "steps": runner.steps, "ring": runner.ring.count, "secs": time.perf_counter() - t0,
                "kinds": [STEP_RECORD.unpack(r)[2] for r in runner.ring.ordered()], "blocks": caches.blocks.available, "slots": caches.slots.available}

    outs = tp.run(rank_main)
    r0 = outs[0]
    print(r0["rec"].table())
    same = all(o["out"] == r0["out"] for o in outs[1:])
    kinds = r0["kinds"]
    print(f"  layers {layers[0]}-{layers[-1]}, {a.seqs} prompts of ~{a.prompt} tokens, max_new {a.max_new}: {r0['steps']} steps "
          f"({kinds.count(1)} prefill, {kinds.count(2)} decode) in {r0['secs']:.1f} s; ring {r0['ring']} records; "
          f"blocks/slots returned: {r0['blocks']}/{r0['slots']}")
    for seq, ids in r0["out"].items():
        print(f"    seq {seq}: {len(ids)} tokens {ids[:12]}{'...' if len(ids) > 12 else ''}")
    print(f"  four ranks produced identical tokens: {same}")
    ok = same and all(len(ids) == a.max_new for ids in r0["out"].values()) and kinds.count(2) == a.max_new - 1 + 0
    print("\n  " + ("PASS: the runner drove prefill and decode through the engine on four ranks" if same and all(len(ids) == a.max_new for ids in r0['out'].values()) else "FAIL"))
    return 0 if same else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true", help="four ranks as threads on this box, reference lanes")
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--kv-gib", type=float, default=1.0)
    ap.add_argument("--prompt", type=int, default=300)
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.local:
        return local(a)
    raise SystemExit("fleet boot: comm + served lanes + serve loop -- next (see MEASUREMENTS 45차)")


if __name__ == "__main__":
    raise SystemExit(main())
