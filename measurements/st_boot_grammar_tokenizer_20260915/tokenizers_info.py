"""Can xgrammar's TokenizerInfo be built, bit for bit, from the `tokenizers` object the boot already loads?

`TokenizerInfo.from_huggingface` on a fast tokenizer reads three things: `get_vocab()`, `backend_tokenizer.to_str()`
(for the vocab type and prefix space) and the stop ids it is given. A transformers `AutoTokenizer` wraps the same
`tokenizers.Tokenizer` -- if both give the same vocabulary and metadata, the same constructor gets the same inputs.
"""
import json
import sys
import time

meta = sys.argv[1]
vocab = 154880
stops = [154820, 154827, 154829]
out = {}


def timed(name, fn):
    t = time.perf_counter()
    value = fn()
    out[name] = round(time.perf_counter() - t, 3)
    return value


import xgrammar as xgr  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

hf = timed("AutoTokenizer.from_pretrained", lambda: AutoTokenizer.from_pretrained(meta))
info_hf = timed("TokenizerInfo.from_huggingface", lambda: xgr.TokenizerInfo.from_huggingface(hf, vocab_size=vocab, stop_token_ids=stops))

raw = timed("tokenizers.Tokenizer.from_file", lambda: Tokenizer.from_file(f"{meta}/tokenizer.json"))
raw.no_truncation(); raw.no_padding()                       # as boot.tokenizer does


def encoded(vocab_dict):
    table = [""] * vocab
    for token, idx in vocab_dict.items():
        if idx < vocab:
            table[idx] = token
    return table


vocab_raw = timed("Tokenizer.get_vocab(with_added_tokens=True)", lambda: raw.get_vocab(with_added_tokens=True))
meta_raw = timed("detect metadata (to_str)", lambda: xgr.TokenizerInfo._detect_metadata_from_hf(raw.to_str()))
info_raw = timed("TokenizerInfo(encoded_vocab, ...)", lambda: xgr.TokenizerInfo(
    encoded(vocab_raw), vocab_type=meta_raw["vocab_type"], vocab_size=vocab, stop_token_ids=stops,
    add_prefix_space=meta_raw["add_prefix_space"]))

vocab_hf = hf.get_vocab()
meta_hf = xgr.TokenizerInfo._detect_metadata_from_hf(hf.backend_tokenizer.to_str())
out["same vocab dict"] = vocab_raw == vocab_hf
out["same metadata"] = meta_raw == meta_hf
out["metadata"] = {k: str(v) for k, v in meta_raw.items()}
out["same decoded vocab"] = list(info_raw.decoded_vocab) == list(info_hf.decoded_vocab)
out["same stop ids"] = list(info_raw.stop_token_ids) == list(info_hf.stop_token_ids)
out["same special ids"] = list(info_raw.special_token_ids) == list(info_hf.special_token_ids)
out["same vocab type / prefix"] = (info_raw.vocab_type, info_raw.add_prefix_space) == (info_hf.vocab_type, info_hf.add_prefix_space)
out["same dump_metadata"] = info_raw.dump_metadata() == info_hf.dump_metadata()
print(json.dumps(out, indent=1))
