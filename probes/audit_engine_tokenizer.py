"""Offline byte/token/stream audit of private incident records. No model or GPU."""
import argparse
import collections
import sys
import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.base.serve import _Stream

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--record', type=Path, required=True)
parser.add_argument('--tokenizer', type=Path, required=True, help='Directory containing tokenizer.json')
parser.add_argument('--alternate-tokenizer', type=Path, required=True)
parser.add_argument('--expected-text', type=Path)
parser.add_argument('--response', type=Path, action='append', default=[], help='Engine response JSON with ids and text')
parser.add_argument('--out', type=Path, required=True, help='Private directory; traces include generated text and token IDs')
args = parser.parse_args()
ROOT, META, ALT = args.out, args.tokenizer, args.alternate_tokenizer
ROOT.mkdir(parents=True, exist_ok=True)

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def tokenizer(path):
    tok = Tokenizer.from_file(str(path / 'tokenizer.json'))
    tok.no_truncation()
    tok.no_padding()
    return tok

# Invert ByteLevel's byte-to-Unicode alphabet independently of its decoder.
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
byte_of = {chr(c): b for b, c in zip(bs, cs)}

a = json.loads((META/'tokenizer.json').read_text())
b = json.loads((ALT/'tokenizer.json').read_text())
tok, alt = tokenizer(META), tokenizer(ALT)
special = {v['id'] for v in a['added_tokens'] if v['special']}
record = json.loads(args.record.read_text())
original = record['tokens'][record['prompt_len']:]
cases = [('original', original, args.expected_text.read_text() if args.expected_text else None)]
for path in args.response:
    r = json.loads(path.read_text())
    cases.append((str(path), r['ids'], r['text']))

report = dict(tokenizers_version=tokenizers.__version__, gpu_used=False,
    tokenizer_sha256=sha((META/'tokenizer.json').read_text()),
    alternate_tokenizer_sha256=sha((ALT/'tokenizer.json').read_text()),
    vocab_equal=tok.get_vocab()==alt.get_vocab(), vocab_size=tok.get_vocab_size(),
    different_json_fields=[k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)],
    specials={v['content']:v['id'] for v in a['added_tokens'] if v['special']}, cases=[])
private = {}
for name, ids, saved in cases:
    decoded = tok.decode(ids)
    raw = b''.join(bytes(byte_of[c] for c in tok.id_to_token(i)) for i in ids if i not in special)
    manual = raw.decode('utf-8', 'replace')
    errors = []
    offset = 0
    while offset < len(raw):
        try:
            raw[offset:].decode('utf-8', 'strict')
            break
        except UnicodeDecodeError as exc:
            start, end = offset+exc.start, offset+exc.end
            errors.append(dict(byte_offset=start, hex=raw[start:end].hex(), reason=exc.reason))
            offset = end
    streams = []
    for width in (1, 2, 7, 8, 64):
        repairs = collections.Counter()
        stream = _Stream(tok, repairs=repairs)
        shown = ''
        monotonic = True
        for start in range(0, len(ids), width):
            stream.extend(ids[start:start+width])
            grown = stream.decoded(False)
            monotonic &= grown.startswith(shown)
            shown = grown
        final = stream.decoded(True)
        streams.append(dict(width=width, equals_full_decode=final==decoded,
                            monotonic=monotonic, repairs=dict(repairs)))
    item = dict(name=name, output_tokens=len(ids), text_sha256=sha(decoded),
        all_ids_in_vocab=all(tok.id_to_token(i) is not None for i in ids),
        saved_text_equal=None if saved is None else saved==decoded, alternate_decode_equal=alt.decode(ids)==decoded,
        manual_byte_decode_equal=manual==decoded, utf8_errors=len(errors),
        replacement_chars=decoded.count('\ufffd'), streams=streams)
    report['cases'].append(item)
    # Private ID/byte trace stays outside git. Token-alone decoding is NOT used
    # as evidence of malformed bytes, since valid Hangul spans several tokens.
    trace=[]
    byte_offset=0
    for j, i in enumerate(ids):
        token=tok.id_to_token(i)
        raw_token=b'' if i in special else bytes(byte_of[c] for c in token)
        trace.append(dict(position=j, id=i, token=token, bytes=raw_token.hex(),
                          byte_offset=byte_offset))
        byte_offset+=len(raw_token)
    private[name]=dict(errors=errors, trace=trace, text=decoded)
    print(json.dumps(item, ensure_ascii=False))

# A decode->encode round trip detects lost/rewritten prompt bytes, but not a
# wrong prompt template or equivalence of every possible BPE segmentation.
prompt=record['tokens'][:record['prompt_len']]
prompt_text=tok.decode(prompt, skip_special_tokens=False)
encoded=tok.encode(prompt_text, add_special_tokens=False).ids
report['prompt']=dict(tokens=len(prompt), round_trip_ids_equal=encoded==prompt,
    reencoded_tokens=len(encoded), alternate_encoding_equal=alt.encode(prompt_text, add_special_tokens=False).ids==encoded,
    replacement_chars=prompt_text.count('\ufffd'), text_sha256=sha(prompt_text))
(ROOT/'tokenizer-audit.json').write_text(json.dumps(report, indent=2)+'\n')
(ROOT/'tokenizer-trace-private.json').write_text(json.dumps(private, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))
