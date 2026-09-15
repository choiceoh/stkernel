"""Where `grammar.for_checkpoint` spends its time on CPU, and whether xgrammar can save its TokenizerInfo.

usage: grammar_steps.py META_DIR   (the checkpoint metadata the boot reads: tokenizer.json, config.json, ...)
"""
import json
import sys
import time

meta = sys.argv[1]
out = {}


def step(name, fn):
    t = time.perf_counter()
    value = fn()
    out[name] = round(time.perf_counter() - t, 3)
    return value


step("import torch", lambda: __import__("torch"))
xgr = step("import xgrammar", lambda: __import__("xgrammar"))
step("import transformers", lambda: __import__("transformers"))
from transformers import AutoTokenizer  # noqa: E402
tok = step("AutoTokenizer.from_pretrained", lambda: AutoTokenizer.from_pretrained(meta))
vocab = 154880
info = step("TokenizerInfo.from_huggingface", lambda: xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=[154820, 154827, 154829]))
compiler = step("GrammarCompiler", lambda: xgr.GrammarCompiler(info))
step("compile_builtin_json_grammar", lambda: compiler.compile_builtin_json_grammar())
out["xgrammar"] = getattr(xgr, "__version__", "?")
out["serialize_json"] = hasattr(info, "serialize_json")
out["deserialize_json"] = hasattr(xgr.TokenizerInfo, "deserialize_json")
if out["serialize_json"] and out["deserialize_json"]:
    blob = step("TokenizerInfo.serialize_json", lambda: info.serialize_json())
    out["serialized_bytes"] = len(blob)
    again = step("TokenizerInfo.deserialize_json", lambda: xgr.TokenizerInfo.deserialize_json(blob))
    compiler2 = step("GrammarCompiler (deserialized)", lambda: xgr.GrammarCompiler(again))
    g1 = compiler.compile_builtin_json_grammar()
    g2 = compiler2.compile_builtin_json_grammar()
    out["same vocab_size"] = again.vocab_size == info.vocab_size
    out["same stop ids"] = list(again.stop_token_ids) == list(info.stop_token_ids)
    out["same special ids"] = list(again.special_token_ids) == list(info.special_token_ids)
    out["same decoded vocab"] = list(again.decoded_vocab) == list(info.decoded_vocab)
    # the first JSON mask from both compilers, bit for bit
    import torch
    masks = []
    for g in (g1, g2):
        m = xgr.GrammarMatcher(g)
        bits = xgr.allocate_token_bitmask(1, vocab)
        m.fill_next_token_bitmask(bits)
        masks.append(bits)
    out["same first json mask"] = bool(torch.equal(masks[0], masks[1]))
print(json.dumps(out, indent=1))
