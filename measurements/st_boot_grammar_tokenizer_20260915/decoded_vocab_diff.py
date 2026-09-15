"""Why a deserialized xgrammar TokenizerInfo's decoded_vocab compares unequal, and whether masks agree beyond JSON's first."""
import json
import sys

import torch
import xgrammar as xgr
from transformers import AutoTokenizer

meta = sys.argv[1]
tok = AutoTokenizer.from_pretrained(meta)
vocab = 154880
info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab, stop_token_ids=[154820, 154827, 154829])
again = xgr.TokenizerInfo.deserialize_json(info.serialize_json())
a, b = list(info.decoded_vocab), list(again.decoded_vocab)
out = dict(len=(len(a), len(b)), types=(type(a[0]).__name__, type(b[0]).__name__))
diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
out["differing"] = len(diff)
out["first"] = [(i, repr(a[i])[:40], repr(b[i])[:40]) for i in diff[:8]]
out["vocab_type"] = (str(info.vocab_type), str(again.vocab_type))
out["add_prefix_space"] = (info.add_prefix_space, again.add_prefix_space)
# masks along a few grammars and token walks
schema = {"type": "object", "properties": {"name": {"type": "string"}, "n": {"type": "integer"},
                                            "tags": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "n"]}
ebnf = 'root ::= "<tool_call>" [a-z_]+ "</tool_call>"'
texts = {"json": '{"a": [1, 2, {"b": "x y"}], "c": true}', "schema": '{"name": "홍길동 x", "n": 42, "tags": ["a", "b"]}',
         "ebnf": "<tool_call>get_weather</tool_call>"}
agree = {}
for label, make in (("json", lambda c: c.compile_builtin_json_grammar()),
                    ("schema", lambda c: c.compile_json_schema(json.dumps(schema))),
                    ("ebnf", lambda c: c.compile_grammar(xgr.Grammar.from_ebnf(ebnf)))):
    grammars = [make(xgr.GrammarCompiler(i)) for i in (info, again)]
    matchers = [xgr.GrammarMatcher(g) for g in grammars]
    ids = tok.encode(texts[label], add_special_tokens=False)
    same = True
    steps = 0
    for t in ids:
        bits = [xgr.allocate_token_bitmask(1, vocab) for _ in matchers]
        for m, bm in zip(matchers, bits):
            m.fill_next_token_bitmask(bm)
        same &= bool(torch.equal(bits[0], bits[1]))
        accepted = [m.accept_token(t) for m in matchers]
        same &= accepted[0] == accepted[1]
        steps += 1
        if not accepted[0]:
            break
    agree[label] = dict(steps=steps, same_masks=same)
out["masks"] = agree
print(json.dumps(out, indent=1, ensure_ascii=False))
