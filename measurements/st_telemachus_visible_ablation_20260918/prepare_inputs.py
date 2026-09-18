from pathlib import Path
import hashlib
import json

from tokenizers import Tokenizer

root = Path('/work')
tokenizer = Tokenizer.from_file('/meta/tokenizer.json')
tokenizer.no_padding()
tokenizer.no_truncation()
record = json.loads((root / 'original-engine-record.json').read_text())
ids = record['tokens'][:record['prompt_len']]
digest = lambda v: hashlib.sha256(json.dumps(v, separators=(',', ':')).encode()).hexdigest()
assert len(ids) == 50005 and digest(ids) == 'e8aefde846b0c67641f9285d2646ca3977ab790534761306422ba81dcc4bb1dc'
assistant, user, reasoning_end = (tokenizer.token_to_id(s) for s in ('<|assistant|>', '<|user|>', '</think>'))
assert ids[47171] == assistant and ids[47663] == user
assert ids[50003] == assistant and tokenizer.decode(ids[50003:], skip_special_tokens=False) == '<|assistant|><think>'
start = ids.index(reasoning_end, 47172, 47663) + 1
end = 47663
special = {v['id'] for v in json.loads(Path('/meta/tokenizer.json').read_text())['added_tokens']}
assert not (set(ids[start:end]) & special), 'visible answer unexpectedly contains a structural token'
assert tokenizer.decode([220], skip_special_tokens=False) == ' '
candidate = ids[:start] + [220] * (end - start) + ids[end:]
assert len(candidate) == len(ids)
assert candidate[:start] == ids[:start] and candidate[end:] == ids[end:]
assert tokenizer.decode(candidate[start:end], skip_special_tokens=False) == ' ' * (end - start)
receipt = dict(original_tokens=len(ids), candidate_tokens=len(candidate), assistant_turn=[47171, 47663],
               replaced_visible_body=[start, end], replaced_tokens=end-start, filler_id=220, filler_text=' ',
               original_sha256=digest(ids), candidate_sha256=digest(candidate),
               unchanged_prefix_sha256=digest(ids[:start]), unchanged_suffix_sha256=digest(ids[end:]),
               body_sha256=digest(ids[start:end]),
               kept='all role markers, clean internal reasoning, all system/tool/memory records, latest user and open assistant prefix',
               changed='only visible body of the immediately preceding malformed Ithaca assistant answer',
               tokenizer_sha256=hashlib.sha256(Path('/meta/tokenizer.json').read_bytes()).hexdigest())
output = root / 'visible-assistant-ablation'
output.mkdir(exist_ok=True)
(output / 'prepared-inputs.json').write_text(json.dumps(dict(receipt=receipt, baseline=ids, neutralized=candidate)))
(output / 'preparation.json').write_text(json.dumps(receipt, indent=2))
(output / 'replaced-body.private.txt').write_text(tokenizer.decode(ids[start:end], skip_special_tokens=False))
print(json.dumps(receipt, indent=2))
