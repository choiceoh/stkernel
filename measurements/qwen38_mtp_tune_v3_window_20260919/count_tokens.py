"""Tokens of the rendered v3 texts: total, the answers' (after the last '<|im_start|>assistant\\n'), and what the
prefix cache saves (each text's longest prefix shared with an earlier text, at block granularity ignored)."""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
marker = [248045, 74455, 198]
for path in sys.argv[2:]:
    total = answers = shared = 0
    seen = []                                   # earlier texts' token ids (prefix sharing, the first 8K tokens)
    by_src = {}
    for line in open(path):
        r = json.loads(line)
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        total += len(ids)
        cut = None
        for j in range(len(ids) - 3, -1, -1):
            if ids[j:j + 3] == marker:
                cut = j + 3
                break
        a = len(ids) - cut if cut is not None else 0
        answers += a
        best = 0
        for s in seen[-64:]:
            n = 0
            for x, y in zip(ids, s):
                if x != y:
                    break
                n += 1
            best = max(best, n)
        shared += best
        seen.append(ids[:8192])
        src = by_src.setdefault(r["src"], [0, 0, 0])
        src[0] += 1
        src[1] += len(ids)
        src[2] += a
    print(json.dumps({"file": path.split("/")[-1], "tokens": total, "answer_tokens": answers,
                      "shared_prefix_tokens": shared, "computed_estimate": total - shared,
                      "by_src [n, tokens, answer_tokens]": by_src}))
