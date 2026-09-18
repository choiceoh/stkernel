"""Private incident sampling records; copied after rank agreement, never resampled."""
import hashlib
import json
from pathlib import Path

import torch


def capture(adapter, jobs, block, dists, temps, ks, ps, uniforms, picks, verdicts):
    root = getattr(adapter.net, 'incident_audit_root', None)
    if root is None or adapter.net.rank != 0 or len(jobs) != 1:
        return None
    seq, raw, _, _ = jobs[0]
    mode = adapter.incident_modes.get(seq)
    generation = adapter._generated_count(seq)
    if mode in (20, 25, 26, 27, 28) and generation >= 32:
        return None
    if mode not in (20, 21, 22, 23, 24, 25, 26, 27, 28):
        if mode not in (5, 10) or generation not in (0, 1, 64, *range(114, 129)):
            return None
    prefix = adapter.tokens[seq]
    prompt_len = adapter.prompt_len[seq]
    digest = lambda ids: hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    admission = adapter.nonces[seq]
    accepted, committed, _ = verdicts[0]
    uniform_values = torch.cat(uniforms).detach().cpu()
    root = Path(root).parent / 'incident-logits'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'admit{admission}-gen{generation}.pt'
    # Keep the existing keys and naming for the margin-analysis reader.  The
    # hashes distinguish equal positions from equal *prefixes* across arms.
    record = dict(admission=admission, seq=seq, generation=generation, mode=mode,
                  raw=raw.detach().cpu(), processed=block.detach().cpu(),
                  probabilities=dists.detach().cpu(), temperature=temps,
                  top_k=ks, top_p=ps, uniforms=uniform_values,
                  uniform=float(uniform_values.item()) if uniform_values.numel() == 1 else None,
                  picks=list(picks), input_tail=prefix[-8:],
                  accepted=accepted, committed=list(committed),
                  prompt_len=prompt_len, prefix_len=len(prefix),
                  prompt_sha256=digest(prefix[:prompt_len]),
                  prefix_sha256=digest(prefix), seed=adapter.seeds.get(seq),
                  row_key=adapter._row_key(seq))
    # Do not silently replace evidence if an admission is unexpectedly reused.
    with path.open('xb') as out:
        torch.save(record, out)
    return path
