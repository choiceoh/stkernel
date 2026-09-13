"""Source-owned query and wire counts, never a throughput estimate."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from engine.modules.prefill_indexer import QueryShard


def budget(tokens, chunk):
    rows=[]
    for ctx in range(0,tokens,chunk):
        count=min(chunk,tokens-ctx)
        shards=[QueryShard(count,ctx,rank,4,512,4) for rank in range(4)]
        # Native prefill keeps very small final steps outside this candidate.
        active=count>=128
        score=[s.score_rows if active else count for s in shards]
        project=[max(64,n) if active and n else n for n in score]
        gathered=0 if not active or shards[0].all_covered else shards[0].capacity*4*512*shards[0].wire_bits//8
        rows.append(dict(context=ctx,tokens=count,active=active,
                         score_rows_per_rank=score,projection_rows_per_rank=project,
                         gathered_id_bytes_per_rank=gathered,
                         remote_id_bytes_per_rank=gathered*3//4))
    before=4*tokens
    after=sum(sum(c['score_rows_per_rank']) for c in rows)
    return dict(tokens=tokens,chunks=rows,baseline_score_rows_all_ranks=before,
                candidate_score_rows_all_ranks=after,score_row_reduction=1-after/before,
                baseline_query_id_communication_bytes=0,
                candidate_remote_id_bytes_per_rank=sum(c['remote_id_bytes_per_rank'] for c in rows),
                scope='per DSA layer; query work count and added ID traffic, not runtime or speed')


if __name__=='__main__':
    here=Path(__file__).parent
    oracle=json.loads((here/'oracle-pr875.json').read_text())
    chunk=oracle['candidate']['prefill_chunk']
    print(json.dumps(dict(source=oracle['candidate']['source'],chunk=chunk,
                          cases=[budget(n,chunk) for n in (2000,2672,32000,128000,129775)]),indent=2))
