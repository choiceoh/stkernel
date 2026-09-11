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
from engine.base.record import DeathDump, Ring                   # noqa: E402
from engine.base.runner import STEP_RECORD, Runner               # noqa: E402
from engine.base.serve import Server                             # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.caches import Glm53Caches, block_bytes, slot_bytes   # noqa: E402
from engine.profiles.glm53 import drafter as drafter_mod           # noqa: E402
from engine.profiles.glm53.adapter import Glm53Engine, NullDrafter             # noqa: E402
from engine.profiles.glm53.net import Glm53Net                   # noqa: E402

GIB = 1 << 30
KV_GIB = 8.73                       # the 40th boot's KV (plan.py): what the box has left after weights, runtime floor and activations
TOKEN_BUDGET = 8192                 # MAX_BATCHED: the 6,912 chunk law follows (shapes.py)
MAX_WAIT_S = 20.0                   # D10's one starvation valve
MAX_SEQS = 4                        # launcher MAX_SEQS


def tokenizer(ckpt=facts.CKPT):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(ckpt) / "tokenizer.json"))


def eos_ids(ckpt=facts.CKPT) -> "list[int]":
    import json
    g = json.loads((Path(ckpt) / "generation_config.json").read_text())
    e = g.get("eos_token_id", [])
    return list(e) if isinstance(e, list) else [e]


def decodable_vocab(tok) -> int:
    """Rows of the head the tokenizer has a token for: ids past this are masked (as served)."""
    return max(tok.get_vocab().values()) + 1


def build(comm, layers, lanes, ranks_dir, kv_gib: float, max_seqs: int, use_drafter: bool, recorder: Recorder,
          max_new: int = 256, temperature: float = 0.0, seed: int = 0):
    F = facts.load()
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    D = drafter_mod.load() if use_drafter else None
    dspecs = drafter_mod.specs(D) if D else []
    bb, sb = block_bytes(F, net.layers), slot_bytes(F, net.layers, net.Hk)
    ns = max_seqs + 1
    ring = drafter_mod.ring_bytes(D) if D else 0
    nb = int((kv_gib * GIB - ns * (sb + ring)) // bb)
    if nb < 2:
        raise MemoryError(f"KV {kv_gib} GiB leaves {nb} blocks after {ns} slots of {(sb + ring) / 2**20:.0f} MiB")
    with recorder.phase("arena"):
        arena = Arena(total_bytes(specs) + total_bytes(dspecs) + 256 * (len(specs) + len(dspecs) + 64) + nb * bb + ns * (sb + ring))
    with recorder.phase("load"):
        views = RankLoader(Path(ranks_dir) / f"rank{comm.rank}of{facts.TP}.safetensors").load(
            [s.name for s in specs], arena=arena, recorder=recorder)
        net.bind(views)
    tok = tokenizer()
    decodable = decodable_vocab(tok)
    drafter = NullDrafter()
    draft_shape = None
    if D:
        with recorder.phase("load drafter"):
            dviews = RankLoader(drafter_mod.DRAFTER / "model.safetensors").load([s.name for s in dspecs], arena=arena, recorder=recorder)
        drafter = drafter_mod.Drafter(D, net, decodable)
        drafter.bind(dviews)
        draft_shape = (D.layers, D.window, D.kv_heads, D.head_dim)
    caches = Glm53Caches(arena, F, net.layers, net.Hk, nb, ns, max_seqs=ns, draft=draft_shape)
    # the aux layers must lie inside the chain: a layer subset (the local smoke) clips them to its last layer -- plumbing only
    aux = [min(L, net.layers[-1]) for L in drafter.aux_layers] if D else None
    engine = Glm53Engine(net, caches, F, drafter, max_new=max_new, eos_ids=eos_ids(), temperature=temperature, seed=seed,
                         decodable=decodable, aux_layers=aux)
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
        F, net, caches, engine, runner = build(comm, layers, lanes, a.ranks, a.kv_gib, MAX_SEQS, a.drafter, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed)
        t0 = time.perf_counter()
        with rec.phase("generate"):
            out = run_prompts(engine, runner, prompts)
            torch.cuda.synchronize()
        return {"rec": rec, "out": out, "steps": runner.steps, "ring": runner.ring.count, "secs": time.perf_counter() - t0,
                "kinds": [STEP_RECORD.unpack(r)[2] for r in runner.ring.ordered()], "blocks": caches.blocks.available, "slots": caches.slots.available,
                "accepted": engine.accepted_total, "drafted": engine.drafted_total, "k": engine.drafter.k}

    if a.serve:
        return local_serve(a, tp, lanes, layers, prompts)
    outs = tp.run(rank_main)
    r0 = outs[0]
    print(r0["rec"].table())
    same = all(o["out"] == r0["out"] for o in outs[1:])
    kinds = r0["kinds"]
    print(f"  layers {layers[0]}-{layers[-1]}, {a.seqs} prompts of ~{a.prompt} tokens, max_new {a.max_new}: {r0['steps']} steps "
          f"({kinds.count(1)} prefill, {kinds.count(2)} decode) in {r0['secs']:.1f} s; ring {r0['ring']} records; "
          f"blocks/slots returned: {r0['blocks']}/{r0['slots']}; drafter K={r0['k']}: {r0['accepted']}/{r0['drafted']} drafts accepted")
    for seq, ids in r0["out"].items():
        print(f"    seq {seq}: {len(ids)} tokens {ids[:12]}{'...' if len(ids) > 12 else ''}")
    print(f"  four ranks produced identical tokens: {same}")
    ok = same and all(len(ids) >= a.max_new for ids in r0["out"].values())
    print("\n  " + ("PASS: the runner drove prefill and decode through the engine on four ranks" if ok else "FAIL"))
    return 0 if ok else 1


def local_serve(a, tp, lanes, layers, prompts) -> int:
    """The serve loop itself, on four threads: rank 0 opens the door, a client
    thread posts the prompts over HTTP, the loop ends when they are answered."""
    import json
    import urllib.request
    port = a.port
    results = {}

    def rank_main(comm):
        rec = Recorder(f"rank{comm.rank}")
        F, net, caches, engine, runner = build(comm, layers, lanes, a.ranks, a.kv_gib, MAX_SEQS, a.drafter, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed)
        server = Server(engine, runner, comm, port=port)
        httpd = None
        if comm.rank == 0:
            httpd = server._serve_http()                       # the door opens before the loop
            def client():
                try:
                    for seq, ids in prompts.items():
                        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", method="POST",
                                                     data=json.dumps({"ids": ids, "max_tokens": a.max_new}).encode(),
                                                     headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=3600) as r:
                            results[seq] = json.loads(r.read())
                except Exception as e:                        # noqa: BLE001
                    results["error"] = repr(e)
                server.alive = False                          # rank 0 stops after the last answer ...
            threading.Thread(target=client, daemon=True).start()
        # ... and tells the others through the same broadcast the arrivals travel on
        while True:
            stop = comm.broadcast_object(not server.alive if comm.rank == 0 else None)
            if stop:
                break
            if not server.once():
                time.sleep(0.002)
        if httpd is not None:
            httpd.shutdown()
        if comm.rank == 0 and "error" in results:
            raise RuntimeError(f"client: {results['error']}")
        if comm.rank == 0 and server.pending:
            raise RuntimeError("rank 0 stopped with answers pending")
        return {"served": server.served, "steps": runner.steps}

    import threading
    t0 = time.perf_counter()
    outs = tp.run(rank_main)
    secs = time.perf_counter() - t0
    ok = len(results) == len(prompts) and all(o["served"] == len(prompts) for o in outs) and all(len(r["ids"]) == a.max_new for r in results.values())
    for seq, r in sorted(results.items()):
        print(f"    seq {seq}: {r['completion_tokens']} tokens in {r['seconds']} s  {r['ids'][:8]}...")
    print(f"  serve loop on four ranks: {outs[0]['steps']} steps, {outs[0]['served']} answered over HTTP in {secs:.1f} s")
    print("\n  " + ("PASS: requests in at rank 0, tokens out, every rank in lockstep" if ok else "FAIL"))
    return 0 if ok else 1


def fleet(a) -> int:
    """One rank per node, inside the glm53 image: served lanes (D3: all or nothing), every layer, then serve."""
    print(f"  box: {facts.check_box()}")
    comm = Comm.init()
    lanes = lane_tables.served(reference_for=("expert",))       # declared, printed, until b12x and the recurrent rows are bound
    rec = Recorder(f"rank{comm.rank}")
    F, net, caches, engine, runner = build(comm, None, lanes, a.ranks, a.kv_gib, MAX_SEQS, True, rec,
                                           max_new=a.max_new, temperature=a.temperature, seed=a.seed)
    dump = DeathDump(a.dump_dir, runner.ring, boot_id=f"glm53-r{comm.rank}-{int(time.time())}")
    if comm.rank == 0:
        print(rec.table())
        print(f"  ST engine: GLM-5.3, TP={facts.TP}, lanes={lanes.name}, KV {a.kv_gib} GiB, serving on :{a.port}")
    try:
        Server(engine, runner, comm, port=a.port, tokenizer=tokenizer()).loop()
    finally:
        dump.close()
        comm.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true", help="four ranks as threads on this box, reference lanes")
    ap.add_argument("--layers", default="0-4")
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--kv-gib", type=float, default=KV_GIB)
    ap.add_argument("--prompt", type=int, default=300)
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serve", action="store_true", help="with --local: through the HTTP door and the lockstep loop")
    ap.add_argument("--drafter", action="store_true", help="with --local: DFlash2 drafts (aux layers clipped to the chain: plumbing, not quality)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dump-dir", default="/home/choiceoh/glm53-logs/st-dumps")
    a = ap.parse_args(argv)
    if a.local:
        if a.kv_gib == KV_GIB:
            a.kv_gib = 1.0                                      # a layer subset on one box
        return local(a)
    return fleet(a)


if __name__ == "__main__":
    raise SystemExit(main())
