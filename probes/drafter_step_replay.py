"""The drafter's decode step as captured graphs, at production's per-rank shape, without a boot (45차 §84).

Production says the drafter is 36.4% of a decode step (observe 21.4%, propose 15.1%, ledger §81), and the only
number that means anything for it is REPLAY time: both stages run inside CUDA graphs, so the host launches are
already gone and an eager measurement mostly reports contention.

This builds a drafter whose facts are already this rank's shard -- heads 8, KV heads 2, inter 3072 at TP=4 over
hidden 4096 and five layers -- with world size 1, so `prepare_fast`'s shard is the identity and every GEMM is
the size the rank runs, plus a stub of the target's vocabulary head, which `propose_rows` puts the block's
output through. About 1.5 GiB of random weights; no checkpoint, no fleet, no collectives (the TP joins are a
no-op here, so `propose` is measured without them, and the head is BF16 where production's is FP8).

A step runs `block_rows` ONCE, not once a draft: the anchor and the K masks are the same block's t = K+1 rows
(45차 §86). The head and the selector after it cost more than the block.

    python3 probes/drafter_step_replay.py            the two stages, and a block by its parts

Read it as a difference, not an absolute, and take differences INSIDE one process: a run-to-run gap of 600 us
on this box is contention, not a change (45차 §86 read a 43.9 us fusion as 634 before it interleaved).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from types import SimpleNamespace

from engine.profiles.glm53.facts import SPEC_K   # noqa: E402  -- the profile owns k, never this probe

ROWS, K = 1, SPEC_K                  # one sequence at the profile's draft width: K + 1 rows a block
VOCAB_LOCAL = 38_720                 # 154,880 over TP=4: the head shard a rank owns
SEL_RANK, SEL_TOP_K = 256, 16        # the candidate selector's codebook width and its candidates a position


def build(device="cuda"):
    import engine.kernels.dense as dense
    if not hasattr(dense, "_deep_gemm"):                       # the prefill lane wants deep_gemm; decode never
        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            dense.FP8Linear = lambda weight, **kw: None
    import engine.profiles.glm53.drafter as drafter
    facts = drafter.DrafterFacts(layers=5, hidden=4096, heads=8, kv_heads=2, head_dim=128, inter=3072,
                                 rms_eps=1e-5, rope_theta=10000.0, window=2048, block=8, mask_id=3,
                                 conv_taps=2, conv_group=16, sel_rank=SEL_RANK, sel_top_k=SEL_TOP_K,
                                 target_layers=(1, 2, 3, 4, 5), k=K)
    embed = torch.nn.Embedding(VOCAB_LOCAL, facts.hidden, device=device, dtype=torch.bfloat16)
    comm = SimpleNamespace(world_size=1, rank=0, all_reduce=lambda x: x, all_gather=lambda x, dim: x)
    gen = torch.Generator(device=device).manual_seed(3)
    # the target's vocabulary head, this rank's shard: what `propose_rows` puts the block's output through
    head = torch.randn(VOCAB_LOCAL, facts.hidden, device=device, generator=gen, dtype=torch.float32).bfloat16() / 64
    target = SimpleNamespace(comm=comm, embed=lambda ids: embed(ids), rank=0, vp=VOCAB_LOCAL,
                             head_local=lambda h: torch.nn.functional.linear(h, head))
    d = drafter.Drafter(facts, target, VOCAB_LOCAL)
    d.p = {s.name: torch.randn(*s.shape, device=device, generator=gen, dtype=torch.float32).bfloat16() / 16
           for s in drafter.specs(facts) if not s.name.startswith("candidate_selector")}
    for name, shape in (("candidate_selector.hidden_projection.weight", (facts.sel_rank, facts.hidden)),
                        ("candidate_selector.predecessor_codebook", (VOCAB_LOCAL, facts.sel_rank)),
                        ("candidate_selector.successor_codebook", (VOCAB_LOCAL, facts.sel_rank))):
        d.p[name] = torch.randn(*shape, device=device, generator=gen, dtype=torch.float32).bfloat16() / 16
    d.prepare_fast()
    torch.cuda.empty_cache()
    return drafter, d, facts


HELD = []


def replay(call, rounds=200):
    """Capture `call` and time its replay: no host launch, only the kernels the capture recorded."""
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    HELD.append(graph)                                          # a freed graph takes its pool with it
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.capture_begin()
        call()
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rounds * 1000.0


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this probe measures captured replay")
    module, d, F = build()
    print(f"drafter on device: {torch.cuda.memory_allocated() / 2**20:.0f} MiB")
    n, t, kv = ROWS, F.k + 1, F.kv_heads
    gen = torch.Generator(device="cuda").manual_seed(4)
    field = torch.zeros(2, F.layers, 2, module.ring_cells(F), kv, F.head_dim, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([1], device="cuda")
    ids = torch.randint(0, VOCAB_LOCAL, (n * t,), device="cuda")
    positions = torch.arange(n * t, device="cuda") + 100
    ctx = torch.tensor([100], device="cuda")
    aux = torch.randn(n * t, F.hidden * len(F.target_layers), device="cuda", generator=gen).bfloat16()
    valid = torch.tensor([t], device="cuda")
    module.warm_rotary(torch.device("cuda"), F.head_dim, F.rope_theta)

    anchors = torch.randint(0, VOCAB_LOCAL, (n,), device="cuda")
    alive = torch.ones(n, dtype=torch.bool, device="cuda")
    observe = replay(lambda: d.observe_rows(field, slots, positions.view(n, t), aux, valid))
    block = replay(lambda: d.block_rows(ids, positions, slots, ctx, field, n, t))
    propose = replay(lambda: d.propose_rows(field, slots, anchors, ctx, alive=alive))
    print(f"\n  observe_rows                      {observe:9.1f} us")
    print(f"  propose_rows                      {propose:9.1f} us")
    print(f"    of which block_rows             {block:9.1f} us")
    print(f"    the head and the selector       {propose - block:9.1f} us")
    print(f"  a decode step's drafter           {observe + propose:9.1f} us")

    from engine.kernels.draft_attention import draft_attention
    x = torch.randn(n * t, F.hidden, device="cuda", generator=gen).bfloat16()
    heads, dim, group = F.heads, F.head_dim, F.hidden // F.conv_group
    qh = torch.randn(t, heads, dim, device="cuda", generator=gen).bfloat16()
    kh = torch.randn(t, kv, dim, device="cuda", generator=gen).bfloat16()
    vh = torch.randn(t, kv, dim, device="cuda", generator=gen).bfloat16()
    gate_up = torch.randn(n * t, 2 * F.inter, device="cuda", generator=gen).bfloat16()

    def gemms():
        for L in range(F.layers):
            q = f"layers.{L}."
            d.linear(x, q + "attention_conv.kernel_projection.weight")
            d.linear(x, q + "mlp_conv.kernel_projection.weight")
            d.linear(x, q + "self_attn.qkv")
            d.linear(x[:, :heads * dim].contiguous(), q + "self_attn.o_proj.weight")
            d.linear(x, q + "mlp.gate_up")
            d.linear(x[:, :F.inter].contiguous(), q + "mlp.down_proj.weight")

    def mixes():
        for L in range(F.layers):
            delta = d.linear(x, f"layers.{L}.attention_conv.kernel_projection.weight").reshape(n * t, 2, F.conv_taps, -1)
            for half in (0, 1):
                for which in ("attention_conv", "mlp_conv"):
                    module.tap_mix(x, delta[:, half], d.p[f"layers.{L}.{which}.base_kernel"][half], F.conv_group, block=t)

    def norms():
        for L in range(F.layers):
            q = f"layers.{L}."
            module.norm(x, d.p[q + "input_layernorm.weight"], F.rms_eps)
            module.norm(x, d.p[q + "post_attention_layernorm.weight"], F.rms_eps)
            module.norm_rope(x[:, :heads * dim].reshape(n * t, heads, dim), d.p[q + "self_attn.q_norm.weight"],
                             F.rms_eps, positions, F.rope_theta)
            module.norm_rope(x[:, :kv * dim].reshape(n * t, kv, dim), d.p[q + "self_attn.k_norm.weight"],
                             F.rms_eps, positions, F.rope_theta)
        module.norm(x, d.p["norm.weight"], F.rms_eps)

    def attention():
        for L in range(F.layers):
            draft_attention(qh, kh, vh, field, ctx[0], slot=slots[0:1], layer=L)

    def swiglu():
        for _ in range(F.layers):
            a, b = gate_up.chunk(2, -1)
            torch.nn.functional.silu(a) * b

    print("\n  a block by its parts:")
    parts = [("30 W4 GEMMs (6 a layer)", gemms), ("20 tap mixes (+5 projections)", mixes),
             ("15 norms + 10 norm_rope", norms), ("5 draft_attention", attention), ("5 silu*mul", swiglu)]
    total = 0.0
    for label, call in parts:
        us = replay(call)
        total += us
        print(f"    {label:34s}{us:9.1f} us")
    print(f"    {'sum of the parts':34s}{total:9.1f} us   (the block: {block:.1f})")


if __name__ == "__main__":
    main()
