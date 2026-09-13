"""Query work removed by coverage; no timing or memory-bandwidth prediction."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.modules.prefill_attention import covered_prefix
from engine.profiles.glm53.facts import architecture


def budget(tokens, facts, chunk=32256, select_rows=1024):
    covered = projected = scored = old_passes = new_passes = 0
    for context in range(0, tokens, chunk):
        rows = min(chunk, tokens-context)
        prefix = covered_prefix(rows, context, facts.topk, facts.kpool) if 128 <= rows <= 32768 else 0
        prefix = prefix if prefix >= 128 else 0
        remaining = rows-prefix
        projected += (max(64, remaining) if remaining else 0) if prefix else rows
        scored += remaining
        covered += prefix
        old_passes += (rows+select_rows-1)//select_rows
        new_passes += (remaining+select_rows-1)//select_rows
    return dict(tokens=tokens, bypassed_query_rows=covered, old_projected_rows=tokens,
                new_projected_rows=projected, old_scored_rows=tokens, new_scored_rows=scored,
                score_row_reduction=covered/tokens, old_selection_passes=old_passes,
                new_selection_passes=new_passes, cache_write_rows=tokens,
                dsa_layers=sum(facts.is_dsa(L) for L in range(facts.layers)))


if __name__ == '__main__':
    facts = architecture(json.loads((ROOT/'measurements/st_prefill_dense_prefix_20260913/model-config.json').read_text()))
    print(json.dumps(dict(scope='per rank per DSA layer; source work, not TTFT', chunk=32256,
                         cases=[budget(n, facts) for n in (2000, 2052, 2672, 32000, 128000)]), indent=2))
