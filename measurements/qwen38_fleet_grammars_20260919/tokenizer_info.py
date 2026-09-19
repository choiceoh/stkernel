"""Qwen3.8's TokenizerInfo two ways, on the served meta (st-qwen38-tep4), in the served image (st-engine:qwen38), CPU:
transformers' AutoTokenizer + TokenizerInfo.from_huggingface (base/grammar.for_checkpoint without a tokenizer, what the
CPU boot does) against the door's own `tokenizers.Tokenizer` read by base/grammar.tokenizer_info (GLM-5.3's fleet path).
tokenizer_info is copied from engine/base/grammar.py at main e5a31d5f; boot.tokenizer and boot.eos_ids from
engine/profiles/qwen38/boot.py."""
import json
import sys
import time
from pathlib import Path

META = Path(sys.argv[1] if len(sys.argv) > 1 else "/meta")


def tokenizer_info(tokenizer, vocab_size, stop_token_ids):
    import xgrammar as xgr
    if not stop_token_ids:
        raise ValueError("a TokenizerInfo read from the backend needs the engine's stop token ids")
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    size = vocab_size or max(len(vocab), max(vocab.values()) + 1)
    encoded = [""] * size
    for token, index in vocab.items():
        if index < size:
            encoded[index] = token
    metadata = xgr.TokenizerInfo._detect_metadata_from_hf(tokenizer.to_str())
    return xgr.TokenizerInfo(encoded, vocab_type=metadata["vocab_type"], vocab_size=size,
                             stop_token_ids=sorted(stop_token_ids), add_prefix_space=metadata["add_prefix_space"])


def eos_ids(ckpt, cfg):
    path = ckpt / "generation_config.json"
    ids = json.loads(path.read_text()).get("eos_token_id") if path.exists() else None
    if ids is None:
        ids = cfg.get("eos_token_id")
    return [int(t) for t in (ids if isinstance(ids, list) else [ids])]


t0 = time.perf_counter()
import xgrammar as xgr                                                  # noqa: E402
t_import_xgr = time.perf_counter() - t0
import transformers                                                     # noqa: E402
import tokenizers                                                       # noqa: E402
from importlib.metadata import version
print(f"xgrammar {version("xgrammar")} transformers {transformers.__version__} tokenizers {tokenizers.__version__}")
print(f"import xgrammar: {t_import_xgr:.2f} s (transformers already in: {'transformers' in sys.modules})")

cfg = json.loads((META / "config.json").read_text())
text = cfg.get("text_config", cfg)
vocab = int(text["vocab_size"])
stops = eos_ids(META, text)
print(f"vocab_size {vocab}, engine stops {stops}")

t0 = time.perf_counter()
tok = tokenizers.Tokenizer.from_file(str(META / "tokenizer.json"))
tok.no_truncation()
tok.no_padding()
t_backend_load = time.perf_counter() - t0

t0 = time.perf_counter()
got = tokenizer_info(tok, vocab, stops)
t_backend_info = time.perf_counter() - t0

t0 = time.perf_counter()
hf = transformers.AutoTokenizer.from_pretrained(str(META))
t_hf_load = time.perf_counter() - t0
t0 = time.perf_counter()
want = xgr.TokenizerInfo.from_huggingface(hf, vocab_size=vocab, stop_token_ids=sorted(stops))
t_hf_info = time.perf_counter() - t0

print(f"backend: Tokenizer.from_file {t_backend_load:.2f} s + tokenizer_info {t_backend_info:.2f} s")
print(f"transformers: AutoTokenizer {t_hf_load:.2f} s ({type(hf).__name__}) + from_huggingface {t_hf_info:.2f} s")

hv, bv = hf.get_vocab(), tok.get_vocab(with_added_tokens=True)
print(f"vocab dicts: transformers {len(hv)}, backend {len(bv)}, equal {hv == bv}")
if hv != bv:
    only_hf = sorted(set(hv.items()) - set(bv.items()), key=lambda kv: kv[1])[:20]
    only_be = sorted(set(bv.items()) - set(hv.items()), key=lambda kv: kv[1])[:20]
    print("  only transformers:", only_hf)
    print("  only backend:", only_be)
dg, dw = list(got.decoded_vocab), list(want.decoded_vocab)
diff = [i for i, (a, b) in enumerate(zip(dg, dw)) if a != b]
print(f"decoded vocab: {len(dg)} vs {len(dw)}, {len(diff)} differ" + (f" (first {diff[:10]})" if diff else ""))
print(f"vocab_type {got.vocab_type} vs {want.vocab_type}; add_prefix_space {got.add_prefix_space} vs "
      f"{want.add_prefix_space}; vocab_size {got.vocab_size} vs {want.vocab_size}")
print(f"stop ids equal {list(got.stop_token_ids) == list(want.stop_token_ids)} {list(got.stop_token_ids)}")
print(f"special ids equal {list(got.special_token_ids) == list(want.special_token_ids)} "
      f"({len(list(got.special_token_ids))} of them)")
print(f"dump_metadata equal {got.dump_metadata() == want.dump_metadata()}")
marker = tok.encode("<tool_call>", add_special_tokens=False).ids
print(f"<tool_call> -> {marker}; decoded {[dg[i] for i in marker]} / {[dw[i] for i in marker]}")

# the masks: a lazy Qwen tool grammar and the builtin JSON grammar, walked token by token over a real call
t0 = time.perf_counter()
cg, cw = xgr.GrammarCompiler(got), xgr.GrammarCompiler(want)
t_compilers = time.perf_counter() - t0
ebnf = ('root ::= call ("\\n<tool_call>" call)*\n'
        'call ::= call0\n'
        'value ::= [^\\u0000]*\n'
        'call0 ::= "\\n<function=" "get_weather" ">\\n" params0 "</function>\\n</tool_call>"\n'
        'params0 ::= ("<parameter=" key0 ">\\n" value "\\n</parameter>\\n")*\n'
        'key0 ::= "city" | "unit"\n')
call = "\n<function=get_weather>\n<parameter=city>\n서울\n</parameter>\n</function>\n</tool_call>"
json_text = '{"city": "서울", "n": [1, 2.5, true, null]}'
import torch                                                             # noqa: E402
for name, grammar, text_ in (("tool (lazy)", xgr.Grammar.from_ebnf(ebnf), call),
                             ("json_object", None, json_text)):
    compiled = [c.compile_grammar(grammar) if grammar is not None else c.compile_builtin_json_grammar()
                for c in (cg, cw)]
    ms = [xgr.GrammarMatcher(c) for c in compiled]
    ids = tok.encode(text_, add_special_tokens=False).ids
    stop = [s for s in stops if s < vocab][0]
    walk = ids + [stop]
    same = True
    for j, t in enumerate(walk):
        masks = []
        for m in ms:
            b = xgr.allocate_token_bitmask(1, vocab)
            m.fill_next_token_bitmask(b)
            masks.append(b.clone())
        if not torch.equal(masks[0], masks[1]):
            same = False
            print(f"  {name}: masks differ at position {j}")
            break
        ok = [m.accept_token(t) for m in ms]
        if ok != [True, True]:
            print(f"  {name}: token {t} at {j} accepted {ok}")
            same = False
            break
    print(f"{name}: {len(walk)} positions walked, masks equal {same}, terminated "
          f"{[m.is_terminated() for m in ms]}")
print(f"GrammarCompiler x2: {t_compilers:.2f} s")
