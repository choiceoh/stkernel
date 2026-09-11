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
from engine.base.arena import Arena, prepare_allocation          # noqa: E402
from engine.base.comm import Comm, LocalTP                       # noqa: E402
from engine.base.config import Config, Fact                      # noqa: E402
from engine.base.instruments import Recorder                     # noqa: E402
from engine.base.loader import RankLoader                        # noqa: E402
from engine.base.params import total_bytes                       # noqa: E402
from engine.base.record import DeathDump, Ring                   # noqa: E402
from engine.base.runner import STEP_RECORD, Runner               # noqa: E402
from engine.base.serve import Server                             # noqa: E402
from engine.base.kv_tier import NvmeTier                         # noqa: E402
from engine.base.prefix import PrefixCache                       # noqa: E402
from engine.base.shapes import chunk_for                         # noqa: E402
from engine.base.tiered_kv import TieredKV                       # noqa: E402
from engine.profiles.glm53 import facts, lanes as lane_tables    # noqa: E402
from engine.profiles.glm53.caches import Glm53Caches, layout, snapshot_layout   # noqa: E402
from engine.profiles.glm53 import drafter as drafter_mod           # noqa: E402
from engine.profiles.glm53.adapter import Glm53Engine, NullDrafter             # noqa: E402
from engine.profiles.glm53.net import Glm53Net                   # noqa: E402
from engine.profiles.glm53.weights import rank_loader            # noqa: E402

GIB = 1 << 30
KV_GIB = 8.73                       # the 40th boot's KV (plan.py): what the box has left after weights, runtime floor and activations
TOKEN_BUDGET = 8192                 # MAX_BATCHED: the 6,912 chunk law follows (shapes.py)
MAX_WAIT_S = 20.0                   # D10's one starvation valve
MAX_SEQS = 4                        # launcher MAX_SEQS
PREFIX_SNAPSHOTS = 8                # chunk-boundary checkpoints kept for prefix reuse (base/prefix.py): ~77 MiB each per rank at
                                    # 45 layers with the drafter (34 KDA states + conv taps + the drafter's context ring)


def tokenizer(ckpt=facts.CKPT):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(ckpt) / "tokenizer.json"))


CHAT_TEMPLATE = "chat_template_mm_v2.jinja"     # what production serves with (launchers/lib/glm53-chat.sh); honours the `thinking` kwarg
REASONING_END = "</think>"                       # the model closes its reasoning with this token; the door splits content there
REQUEST_TIMEOUT_S = 3600.0                       # a request older than this is cancelled (the production probe's long-ingest bound x12)


def chat_renderer(ckpt=facts.CKPT):
    """messages -> prompt text through the checkpoint's chat template (the door's OpenAI chat endpoint). transformers'
    template engine renders it (the template needs its filters); chat_template_kwargs (`thinking`, ...) pass through."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(ckpt))
    t.chat_template = (Path(ckpt) / CHAT_TEMPLATE).read_text()

    def render(messages, kwargs):
        return t.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
    return render


def eos_ids(ckpt=facts.CKPT) -> "list[int]":
    import json
    g = json.loads((Path(ckpt) / "generation_config.json").read_text())
    e = g.get("eos_token_id", [])
    return list(e) if isinstance(e, list) else [e]


def declared(a, comm_world: int) -> Config:
    """D11: the only inputs are facts and expiring knobs; an undeclared STK_*
    in the environment kills the boot. No knobs today -- every value below is
    a fact with a source, and there is nothing to tune by env."""
    facts_ = [
        Fact("model", str(a.ckpt_meta), "the checkpoint's config/tokenizer (facts.CKPT or a copy of those files)"),
        Fact("ranks", str(a.ranks), "preshard output"),
        Fact("world", comm_world, "facts.TP: four Sparks"),
        Fact("block", facts.BLOCK, "launcher --block-size"),
        Fact("spec_k", facts.SPEC_K, "launcher SPEC_K with DFlash2"),
        Fact("kv_gib", float(a.kv_gib), "40th boot's measured KV" if a.kv_gib == KV_GIB else "--kv-gib (local)"),
        Fact("port", int(a.port), "--port"),
        Fact("prefix_snapshots", PREFIX_SNAPSHOTS, "chunk-boundary checkpoints for prefix reuse (boot.PREFIX_SNAPSHOTS)"),
    ]
    cfg = Config(facts_, knobs=[])
    return cfg


def decodable_vocab(tok) -> int:
    """Rows of the head the tokenizer has a token for: ids past this are masked (as served)."""
    return max(tok.get_vocab().values()) + 1


def build(comm, layers, lanes, ranks_dir, kv_gib: float, max_seqs: int, use_drafter: bool, recorder: Recorder,
          max_new: int = 256, temperature: float = 0.0, seed: int = 0, tier_dir: "str | None" = None,
          ckpt_meta: "str | Path" = facts.CKPT, drafter_dir: "str | Path" = drafter_mod.DRAFTER):
    """`ckpt_meta`: where config.json / tokenizer.json / generation_config.json are -- the HF checkpoint dir, or a
    copy of just those files: a node needs its rank file, the drafter and this, not the 185 GB checkpoint."""
    F = facts.load(ckpt_meta)
    net = Glm53Net(F, comm, lanes, layers)
    specs = net.specs()
    drafter_dir = Path(drafter_dir)
    D = drafter_mod.load(drafter_dir) if use_drafter else None
    dspecs = drafter_mod.specs(D) if D else []
    draft_shape = (D.layers, D.window, D.kv_heads, D.head_dim) if D else None
    cache_layout = layout(F, net.layers, draft_shape)
    bb, sb = cache_layout.block_bytes, cache_layout.slot_bytes
    ns = max_seqs + 1
    # the persistent int32 block table is part of the same declared budget
    nb = int((kv_gib * GIB - ns * sb) // (bb + max_seqs * 4))
    if nb < 2:
        raise MemoryError(f"KV {kv_gib} GiB leaves {nb} blocks after {ns} slots of {sb / 2**20:.0f} MiB")
    rank = rank_loader(Path(ranks_dir) / f"rank{comm.rank}of{facts.TP}.safetensors")
    snapshot_bytes = snapshot_layout(F, net.layers, draft_shape)[0]
    arena_bytes = (total_bytes(specs) + total_bytes(dspecs) + 256 * (len(specs) + len(dspecs) + 64) + cache_layout.nbytes(nb, max_seqs)
                   + PREFIX_SNAPSHOTS * snapshot_bytes)
    if len(net.layers) == F.layers:
        # Full-model admission must not rely on the page-cache-inclusive
        # MemAvailable value. The spare 16 GiB is a conservative boot guard,
        # not a measured graph-workspace/performance budget.
        files = sorted(Path(ranks_dir).glob("rank*of4.safetensors"))
        if D:
            files.append(drafter_dir / "model.safetensors")
        failure = None
        try:
            report = prepare_allocation(arena_bytes, files, 16 * GIB,
                                        lambda: torch.cuda.mem_get_info()[0])
        except (MemoryError, OSError) as exc:
            failure = exc
        # A failed rank must prevent peers from starting their large CUDA
        # allocations; closing NCCL only after one rank fails is too late.
        failed = comm.all_reduce(torch.tensor([int(failure is not None)], device="cuda"))
        if int(failed.item()):
            raise MemoryError(f"TP arena admission failed: {failure or 'a peer has insufficient immediately free memory'}") from failure
        recorder.gauge("boot_immediately_free_GiB", round(report["immediately_free"] / GIB, 3))
    with recorder.phase("arena"):
        arena = Arena(arena_bytes)
    with recorder.phase("load"):
        views = rank.load(
            [s.name for s in specs], arena=arena, recorder=recorder)
        net.bind(views)
    tok = tokenizer(ckpt_meta)
    decodable = decodable_vocab(tok)
    drafter = NullDrafter()
    if D:
        with recorder.phase("load drafter"):
            dviews = RankLoader(drafter_dir / "model.safetensors").load([s.name for s in dspecs], arena=arena, recorder=recorder)
        drafter = drafter_mod.Drafter(D, net, decodable)
        drafter.bind(dviews)
    caches = Glm53Caches(arena, F, net.layers, nb, max_seqs, draft=draft_shape, snapshots=PREFIX_SNAPSHOTS)
    # the aux layers must lie inside the chain: a layer subset (the local smoke) clips them to its last layer -- plumbing only
    aux = [min(L, net.layers[-1]) for L in drafter.aux_layers] if D else None
    engine = Glm53Engine(net, caches, F, drafter, max_new=max_new, eos_ids=eos_ids(ckpt_meta), temperature=temperature, seed=seed,
                         decodable=decodable, aux_layers=aux)
    contract = sched.Contract(chunk_align=F.block, token_budget=TOKEN_BUDGET, draft_slots=drafter.k,
                              max_wait_s=MAX_WAIT_S, max_running=max_seqs)
    tiered = None
    if tier_dir:                                                                    # D16: idle conversations park on NVMe, per rank
        tier = NvmeTier(Path(tier_dir) / f"rank{comm.rank}", block_bytes=cache_layout.block_bytes)   # a block is one NVMe unit (block-major)
        tiered = TieredKV(caches.pool, tier)
    prefix = PrefixCache(F.block, chunk_for(F.block, TOKEN_BUDGET, drafter.k), PREFIX_SNAPSHOTS)   # boundaries = prefill chunks
    runner = Runner(engine, contract, caches.pool, caches.slots, Ring(4096, STEP_RECORD.size), recorder, tiered=tiered,
                    keep_idle=tiered is not None, prefix=prefix)                   # with a tier, conversations live on and park
    recorder.gauge("blocks", nb); recorder.gauge("slots", ns); recorder.gauge("arena_GiB", round(arena.used / GIB, 3))
    recorder.gauge("prefix_snapshots", PREFIX_SNAPSHOTS); recorder.gauge("snapshot_MiB", round(snapshot_bytes / 2**20, 1))
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
    print(declared(a, facts.TP).table())
    layers = [int(x) for x in a.layers.split("-")]; layers = list(range(layers[0], layers[-1] + 1))
    torch.manual_seed(a.seed)
    prompts = {seq: torch.randint(0, 100_000, (a.prompt + 7 * seq,)).tolist() for seq in range(a.seqs)}
    tp = LocalTP(facts.TP)
    lanes = lane_tables.reference()
    if a.park:                                            # a run-private tier: parked ids from an earlier smoke must not collide
        import tempfile
        Path(a.tier_dir).mkdir(parents=True, exist_ok=True)
        a.tier_dir = tempfile.mkdtemp(prefix="local-", dir=a.tier_dir)

    def rank_main(comm):
        rec = Recorder(f"rank{comm.rank}")
        F, net, caches, engine, runner = build(comm, layers, lanes, a.ranks, a.kv_gib, MAX_SEQS, a.drafter, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed,
                                               tier_dir=a.tier_dir if a.park else None,
                                               ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir)
        t0 = time.perf_counter()
        with rec.phase("generate"):
            out = run_prompts(engine, runner, prompts)
            torch.cuda.synchronize()
        parked = None
        if a.park:
            # D16 on the real caches: the finished conversation 0 still holds its blocks (keep_idle); park it, the arena
            # gets them back; resume into fresh blocks; wake and decode 4 more tokens -- they must equal a straight run's
            seq, straight = 0, 2
            with rec.phase("park"):
                free_before = caches.pool.available
                wrote = runner.park(seq)
                free_after = caches.pool.available
                got = runner.resume(seq)
                torch.cuda.synchronize()
            with rec.phase("continue"):
                runner.wake(seq)
                engine.limits[seq] = (engine.limits[seq][0] + 4, engine.limits[seq][1])
                while seq in runner.state.running and runner.step(now=0.0) is not None:
                    pass
                torch.cuda.synchronize()
            continued = engine.generated(seq)[a.max_new:]
            with rec.phase("straight"):                                          # the same prompt, max_new + 4 in one go
                engine.add(straight, prompts[seq], max_new=a.max_new + 4)
                runner.submit(straight, len(prompts[seq]), now=0.0)
                while straight not in runner.idle and runner.step(now=0.0) is not None:
                    pass
                torch.cuda.synchronize()
            parked = {"wrote": wrote, "got": got, "free_before": free_before, "free_after": free_after,
                      "continued": continued, "straight_tail": engine.generated(straight)[a.max_new:]}
        return {"rec": rec, "out": out, "steps": runner.steps, "ring": runner.ring.count, "secs": time.perf_counter() - t0,
                "kinds": [STEP_RECORD.unpack(r)[2] for r in runner.ring.ordered()], "blocks": caches.pool.available, "slots": caches.slots.available,
                "accepted": engine.accepted_total, "drafted": engine.drafted_total, "k": engine.drafter.k, "parked": parked}

    if a.serve:
        try:
            return local_serve(a, tp, lanes, layers, prompts)
        finally:
            if a.park:
                import shutil
                shutil.rmtree(a.tier_dir, ignore_errors=True)
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
    if r0["parked"] is not None:
        pk = r0["parked"]
        same_p = all(o["parked"]["continued"] == pk["continued"] for o in outs[1:])
        print(f"  park/resume (D16 on the real caches): wrote {pk['wrote'] / 2**20:.1f} MiB (arena free {pk['free_before']} -> {pk['free_after']} blocks), "
              f"read back {pk['got'] / 2**20:.1f} MiB, continued {pk['continued']} == straight run's {pk['straight_tail']}: "
              f"{pk['continued'] == pk['straight_tail']}; ranks agree {same_p}")
        ok = ok and same_p and pk["wrote"] == pk["got"] and pk["free_after"] > pk["free_before"] and pk["continued"] == pk["straight_tail"]
    print("\n  " + ("PASS: the runner drove prefill and decode through the engine on four ranks" if ok else "FAIL"))
    if a.park:
        import shutil
        shutil.rmtree(a.tier_dir, ignore_errors=True)
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
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed,
                                               tier_dir=a.tier_dir if a.park else None,
                                               ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir)
        tok = tokenizer(a.ckpt_meta)
        from engine.profiles.glm53.tools import parse_tool_calls
        server = Server(engine, runner, comm, port=port, tokenizer=tok, chat=chat_renderer(a.ckpt_meta) if comm.rank == 0 else None,
                        model_name="glm-5.3-flash", reasoning_end=tok.token_to_id(REASONING_END), request_timeout_s=REQUEST_TIMEOUT_S,
                        tool_parser=parse_tool_calls)
        httpd = None
        if comm.rank == 0:
            httpd = server._serve_http()                       # the door opens before the loop
            def post(body, path="/v1/completions"):
                req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST",
                                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=3600) as r:
                    return json.loads(r.read())

            def stream_chat(body):                            # the bench's dialect: SSE chunks, usage at the end
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                text, usage, finish = [], {}, None
                with urllib.request.urlopen(req, timeout=3600) as r:
                    for raw in r:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                            continue
                        obj = json.loads(line[5:])
                        usage = obj.get("usage") or usage
                        for ch in obj.get("choices") or []:
                            d = ch.get("delta") or {}
                            text.append(d.get("content") or d.get("reasoning_content") or "")
                            finish = ch.get("finish_reason") or finish
                return {"text": "".join(text), "usage": usage, "finish_reason": finish}

            def client():
                try:
                    for seq, ids in prompts.items():
                        results[seq] = post({"ids": ids, "max_tokens": a.max_new})
                    if a.park:                                # a second turn on conversation 0: parked on NVMe after its first, resumed here
                        results["turn2"] = post({"conversation": results[0]["conversation"], "ids": prompts[0][:5], "max_tokens": 4})
                    results["chat"] = stream_chat({"messages": [{"role": "user", "content": "안녕? 한 줄로 답해."}], "max_tokens": 6,
                                                   "stream": True, "stream_options": {"include_usage": True},
                                                   "chat_template_kwargs": {"thinking": True}})
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
    answers = {k: v for k, v in results.items() if k not in ("turn2", "chat")}
    ok = len(answers) == len(prompts) and all(len(r["ids"]) == a.max_new for r in answers.values())
    for seq, r in sorted(answers.items()):
        print(f"    seq {seq}: {r['completion_tokens']} tokens in {r['seconds']} s  {r['ids'][:8]}...")
    if a.park:
        t2 = results.get("turn2", {})
        print(f"    turn 2 on conversation {t2.get('conversation')}: {t2.get('completion_tokens')} tokens in {t2.get('seconds')} s (resumed from NVMe, 5 new prompt tokens)")
        ok = ok and t2.get("completion_tokens") == 4
    chat = results.get("chat", {})
    print(f"    chat (v2 template, streamed): {chat.get('usage', {}).get('completion_tokens')} tokens, finish {chat.get('finish_reason')}, text {chat.get('text', '')[:60]!r}")
    ok = ok and chat.get("usage", {}).get("completion_tokens") == 6 and chat.get("finish_reason") in ("stop", "length")
    print(f"  serve loop on four ranks: {outs[0]['steps']} steps, {outs[0]['served']} answered over HTTP in {secs:.1f} s")
    print("\n  " + ("PASS: requests in at rank 0, tokens out, every rank in lockstep" if ok else "FAIL"))
    return 0 if ok else 1


def fleet(a) -> int:
    """One rank per node, inside the glm53 image: served lanes (D3: all or nothing), every layer, then serve."""
    print(f"  box: {facts.check_box()}")
    cfg = declared(a, facts.TP)
    comm = Comm.init()
    engine = dump = None
    try:
        if comm.rank == 0:
            print(cfg.table())
        lanes = lane_tables.served()                                              # every served lane, or the boot dies (D3)
        rec = Recorder(f"rank{comm.rank}")
        F, net, caches, engine, runner = build(comm, None, lanes, a.ranks, a.kv_gib, MAX_SEQS, True, rec,
                                               max_new=a.max_new, temperature=a.temperature, seed=a.seed, tier_dir=a.tier_dir,
                                               ckpt_meta=a.ckpt_meta, drafter_dir=a.drafter_dir)
        with rec.phase("capture decode"):
            engine.capture_decode(MAX_SEQS)
        dump = DeathDump(a.dump_dir, runner.ring, boot_id=f"glm53-r{comm.rank}-{int(time.time())}")
        if comm.rank == 0:
            print(rec.table())
            print(f"  ST engine: GLM-5.3, TP={facts.TP}, lanes={lanes.name}, KV {a.kv_gib} GiB, serving on :{a.port}")
            if runner.tiered is not None:
                t = runner.tiered.tier
                print(f"  NVMe tier: {sum(1 for k in t.index if t.has(int(k)))} conversations parked from before, {len(t.stale())} under another layout (kept, not resumable)")
        tok = tokenizer(a.ckpt_meta)
        from engine.profiles.glm53.tools import parse_tool_calls
        Server(engine, runner, comm, port=a.port, tokenizer=tok, chat=chat_renderer(a.ckpt_meta) if comm.rank == 0 else None,
               model_name="glm-5.3-flash", reasoning_end=tok.token_to_id(REASONING_END), request_timeout_s=REQUEST_TIMEOUT_S,
               tool_parser=parse_tool_calls).loop()
    finally:
        if dump is not None:
            dump.close()
        if engine is not None:
            engine.close_decode()
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
    ap.add_argument("--park", action="store_true", help="with --local: park a finished conversation on NVMe, resume, continue (D16)")
    ap.add_argument("--tier-dir", default="/home/choiceoh/glm53-logs/st-tier")
    ap.add_argument("--ckpt-meta", default=str(facts.CKPT), help="dir with config.json, tokenizer.json, generation_config.json")
    ap.add_argument("--drafter-dir", default=str(drafter_mod.DRAFTER), help="DFlash2 config and model.safetensors directory")
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
