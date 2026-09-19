"""Qwen3.8-Flash-Next through the engine (profile): the composition served by base/composed, driven by base/runner,
answered at base/serve's door -- the reference lane, on the CPU or one GPU, world 1.

    PYTHONPATH=. python3 -m engine.profiles.qwen38.boot --tiny --max-new 8                    # a synthetic checkpoint: the plumbing
    PYTHONPATH=. python3 -m engine.profiles.qwen38.boot --ckpt DIR --prompt "..." --max-new 16  # the checkpoint's weights and tokenizer
    PYTHONPATH=. python3 -m engine.profiles.qwen38.boot --ckpt DIR --serve --port 8000          # the door stays open

What this is not: the served lane. Every feature computes its torch form (engine/modules) and the store gathers rows
for it; the kernels, glue and captured graphs the wizard's table names for this shape bind behind the same features
later. The point of running it is that a request goes in at the door and tokens come out of the same runner,
scheduler, block pool, slot pool and prefix cache GLM-5.3 serves with -- the composition is a Model to them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

GIB = 1 << 30
EFFORT_RUNGS = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
"""The door's reasoning_effort rungs onto this template's three. chat_template.jinja accepts xhigh (its default), medium
and low and raises for anything else, so GLM-5.3's ladder (high, max) turned an ordinary `high` into a 400. OpenAI's
top rungs land on xhigh -- the deepest this template has -- and `xhigh` itself is accepted for callers that speak it."""
EFFORT_ALIASES = {"high": "xhigh", "max": "xhigh"}   # a top-level `high` and a template `xhigh` are the same request


def tokenizer(ckpt: Path):
    """The checkpoint's tokenizer without its truncation rule (engine/profiles/glm53/boot.tokenizer says why)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ckpt / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return tok


def template_kwargs(kwargs) -> dict:
    """The door's template switches in this template's words: the door speaks `thinking` (it copies a client's
    `enable_thinking` there), and Qwen3.8's template reads only `enable_thinking` -- a client that sent `thinking: false`
    alone would otherwise get a think block it asked not to."""
    kwargs = dict(kwargs or {})
    if "thinking" in kwargs and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = kwargs["thinking"]
    return kwargs


def chat_renderer(ckpt: Path):
    """messages -> prompt text through the checkpoint's chat template (transformers renders it).

    Leading system turns go in as one: this template writes a single system block and raises 'System message must be
    at the beginning' on a second, which an OpenAI client that sends `system` and `developer` instructions produces
    (base/serve.one_system). GLM-5.3's template writes a block per system turn and needs nothing."""
    from transformers import AutoTokenizer
    from engine.base.serve import one_system
    t = AutoTokenizer.from_pretrained(str(ckpt))

    def render(messages, kwargs, *, generation_prompt: bool = True, continue_final: bool = False):
        extra = {"continue_final_message": True} if continue_final else {}
        return t.apply_chat_template(one_system(messages), tokenize=False,
                                     add_generation_prompt=generation_prompt and not continue_final,
                                     **template_kwargs(kwargs), **extra)
    return render


def eos_ids(ckpt: Path, cfg: dict) -> "list[int]":
    """The generation config's end tokens, else the text config's."""
    path = ckpt / "generation_config.json"
    ids = json.loads(path.read_text()).get("eos_token_id") if path.exists() else None
    if ids is None:
        ids = cfg.get("eos_token_id")
    return [int(t) for t in (ids if isinstance(ids, list) else [ids])]


def generation_defaults(ckpt: Path) -> dict:
    path = ckpt / "generation_config.json"
    g = json.loads(path.read_text()) if path.exists() else {}
    return {k: g[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty") if k in g}   # the door fills what a request omits


def build(composition, cfg: dict, ends, *, kv_gib: float, max_seqs: int, block_tokens: int, chunk: int, token_budget: int,
          snapshots: int, max_new: int, temperature: float, device="cpu", heads=(), k: int = 0, grammars=None):
    """The engine's pieces around a composition: store, pools, model, contract, prefix cache, runner. With MTP `heads`
    and `k`, the model drafts k tokens a step through them (engine/modules/mtp) and verifies them by position; with
    `grammars` (base/grammar), it serves structured output."""
    from engine.base.composed import ComposedModel, store_for
    from engine.base.prefix import PrefixCache
    from engine.base.record import Ring
    from engine.base.runner import STEP_RECORD, Runner
    from engine.base.scheduler import Contract
    k = k if heads else 0
    store, pool, slots, plan = store_for(composition, kv_gib, max_seqs, block_tokens, snapshots=snapshots, device=device,
                                         ring=k + 1 if k else 0, also=tuple(heads) if k else ())
    drafter = None
    if k:
        from engine.modules.mtp import MTPDrafter
        drafter = MTPDrafter(heads, store, k=k, vocab=cfg["vocab_size"])
    model = ComposedModel(composition, store, vocab=cfg["vocab_size"], eos_ids=ends, max_new=max_new, temperature=temperature,
                          max_context=cfg.get("max_position_embeddings", 2 ** 31 - 1), drafter=drafter, grammars=grammars)
    contract = Contract(chunk_align=chunk, token_budget=token_budget, draft_slots=k, max_wait_s=0.0, max_running=max_seqs)
    prefix = PrefixCache(block_tokens, chunk, snapshots) if snapshots else None
    runner = Runner(model, contract, pool, slots, Ring(4096, STEP_RECORD.size), keep_idle=True, prefix=prefix)
    return model, runner, plan


def tiny(seed: int = 0, mtp: bool = False):
    """A synthetic checkpoint for the plumbing: the tiny config tests use, random weights, no tokenizer; with `mtp`, a
    random MTP head too -> (composition, cfg, heads)."""
    from engine.profiles.qwen38 import composition as qc
    from engine.profiles.qwen38.weights import random_weights
    cfg = {"hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
           "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
           "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
           "shared_expert_intermediate_size": 32, "hidden_act": "silu", "output_gate_type": "sigmoid", "norm_topk_prob": True,
           "hc_count": 4, "hc_lowrank": 16, "ple_layer_ids": [2], "ple_embed_dim": 32, "ple_conv_kernel_size": 4, "ngram_size": 3,
           "heads_per_ngram": 8, "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8, "seed": 1234,
           "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 32, "indexer_budget": 8, "indexer_compress_ratio": 4,
           "vocab_size": 512, "eos_token_id": 0, "rms_norm_eps": 1e-6, "dtype": "float32", "max_position_embeddings": 4096,
           "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                               "mrope_section": [2, 1, 1], "mrope_interleaved": True},
           "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]}
    if mtp:
        cfg["mtp_num_hidden_layers"] = 1
    weights = random_weights(cfg, seed)
    heads = qc.build_mtp(cfg, weights.__getitem__) if mtp else []
    return qc.build(cfg, weights.__getitem__), cfg, heads


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m engine.profiles.qwen38.boot", description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="the checkpoint directory (config.json, safetensors, tokenizer.json)")
    ap.add_argument("--tiny", action="store_true", help="a synthetic tiny checkpoint instead: the plumbing, no tokenizer")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"), help="the weights and activations")
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (a truncated model: timing, loading)")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--ids", default=None, help="prompt token ids, comma separated (no tokenizer needed)")
    ap.add_argument("--chat", action="store_true", help="render the prompt as a user turn through the chat template")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--kv-gib", type=float, default=1.0)
    ap.add_argument("--max-seqs", type=int, default=4)
    ap.add_argument("--block-tokens", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=256, help="prefill chunk alignment (whole blocks)")
    ap.add_argument("--token-budget", type=int, default=1024)
    ap.add_argument("--snapshots", type=int, default=8, help="prefix boundaries kept (0: no prefix cache)")
    ap.add_argument("--serve", action="store_true", help="keep the HTTP door open instead of answering one prompt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--expert-cache", type=int, default=256)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--mtp", type=int, default=0, metavar="K", help="draft K tokens a step with the checkpoint's MTP head")
    a = ap.parse_args(argv)
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    tok = chat = grammars = None
    heads = []
    if a.tiny:
        composition, cfg, heads = tiny(mtp=a.mtp > 0)
        ends = [cfg["eos_token_id"]]
    else:
        from engine.profiles.qwen38.weights import Weights
        ckpt = Path(a.ckpt)
        weights = Weights(ckpt, dtype=a.dtype, expert_cache=a.expert_cache)
        cfg = weights.config
        if a.layers is not None:
            cfg = dict(cfg, layer_types=cfg["layer_types"][:a.layers],
                       ple_layer_ids=[i for i in cfg.get("ple_layer_ids") or () if i <= a.layers])
            weights.config = cfg
        composition = weights.composition()
        if a.mtp:
            heads = weights.mtp_heads()
        ends = eos_ids(ckpt, cfg)
        tok = tokenizer(ckpt)
        if a.chat:
            chat = chat_renderer(ckpt)
        from engine.base import grammar
        grammars = grammar.for_checkpoint(ckpt, cfg["vocab_size"], "cpu", stop_token_ids=ends)   # response_format
    model, runner, plan = build(composition, cfg, ends, kv_gib=a.kv_gib, max_seqs=a.max_seqs, block_tokens=a.block_tokens,
                                chunk=a.chunk, token_budget=a.token_budget, snapshots=a.snapshots, max_new=a.max_new,
                                temperature=a.temperature, heads=heads, k=a.mtp, grammars=grammars)
    print(f"  {'tiny' if a.tiny else a.ckpt}: {len(cfg['layer_types'])} layers, vocab {cfg['vocab_size']}, {'float32' if a.tiny else a.dtype}; "
          f"kv {plan.num_blocks} blocks x {plan.block_tokens} tokens ({plan.paged_gib:.3f} GiB), {plan.num_slots - 1} slots "
          f"of {plan.slot_bytes / 2**20:.1f} MiB; built in {time.perf_counter() - t0:.1f} s", flush=True)
    from engine.base import tool_formats
    from engine.base.comm import Comm
    from engine.base.serve import Server, effort_rungs_checked, reasoning_marks
    reasoning_end, reasoning_tail = reasoning_marks(tok, chat)             # the think block, read off the template
    tools = tool_formats.detect(chat)                                      # the call layout, read off the template
    efforts = effort_rungs_checked(chat, EFFORT_RUNGS) if chat is not None else None
    print(f"  door: structured output {'on' if grammars else 'off (no xgrammar)'}; reasoning "
          + (f"split at {reasoning_end} (thinking-off tail {list(reasoning_tail)})" if reasoning_end is not None else "not split")
          + f"; tools {tools.name + ' (grammar at ' + str(tools.start_token(tok)) + ')' if tools else 'off'}", flush=True)
    server = Server(model, runner, Comm(), port=a.port, tokenizer=tok, chat=chat, model_name="qwen3.8-flash-next",
                    generation=generation_defaults(Path(a.ckpt)) if a.ckpt else None, reasoning_end=reasoning_end,
                    reasoning_tail=reasoning_tail, effort_rungs=efforts,
                    reasoning_effort_aliases=EFFORT_ALIASES if efforts is not None else None,
                    tool_parser=tools.parse if tools else None, tool_stream=tools.partial if tools else None,
                    tool_grammar=tools.grammar if tools else None, tool_call_start=tools.start_token(tok) if tools else None)
    if a.serve:
        print(f"  door open on port {a.port}", flush=True)
        server.loop()
        return 0
    if a.ids:
        ids = [int(t) for t in a.ids.split(",")]
    elif tok is None:
        ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        text = chat([{"role": "user", "content": a.prompt}], {}) if chat else a.prompt
        ids = tok.encode(text, add_special_tokens=False).ids
    request, event = server.submit(ids, a.max_new, a.temperature)
    t1 = time.perf_counter()
    steps = 0
    while not event.is_set():
        if server.once():
            steps += 1
            done = model.generated_count(next(iter(server._active))) if server._active else a.max_new
            print(f"    step {runner.steps}: {done} tokens, {time.perf_counter() - t1:.1f} s", flush=True)
        else:
            time.sleep(0.001)
    out = server.take_result(request)
    secs = time.perf_counter() - t1
    print(f"  prompt {len(ids)} tokens -> {len(out)} tokens in {secs:.1f} s ({runner.steps} steps): {out}")
    if model.k:
        rounds = max(model.drafts_total, 1)
        print(f"  mtp k={model.k}: {model.drafts_total} rounds, {model.drafted_total} drafted, {model.accepted_total} accepted "
              f"({model.accepted_total / max(model.drafted_total, 1):.0%} of drafts, {1 + model.accepted_total / rounds:.2f} tokens a round)")
    if tok is not None:
        print("  text: " + repr(tok.decode(out, skip_special_tokens=False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
