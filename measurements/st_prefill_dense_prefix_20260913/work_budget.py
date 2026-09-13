"""Source-level FP8 latent-row loads, not DRAM traffic or a timing estimate."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.modules.prefill_attention import covered_prefix


def budget(rows, *, chunk=32256, topk=2048, pool=4, queries=2):
    before = after = dense_queries = copy_rows = 0
    for context in range(0, rows, chunk):
        count = min(chunk, rows-context)
        copy_rows += count if count >= 128 else 0
        prefix = covered_prefix(count, context, topk, pool) if 128 <= count <= 32768 else 0
        if prefix < 128:
            prefix = 0
        lengths = [min((context+r+1)//pool, topk//pool)*pool+(context+r+1)%pool for r in range(count)]
        before += sum(lengths)
        after += sum(lengths[prefix:])
        after += sum(context+min(begin+queries, prefix) for begin in range(0, prefix, queries))
        dense_queries += prefix
    return dict(input_tokens=rows, dense_queries=dense_queries, old_kv_rows=before,
                new_kv_rows=after, requested_load_reduction=1-after/before,
                old_requested_bytes_per_rank_dsa_layer=before*512,
                new_requested_bytes_per_rank_dsa_layer=after*512,
                redundant_output_copy_read_write_bytes_removed_per_rank_dsa_layer=copy_rows*16*512*2*2,
                new_collectives=0, new_persistent_bytes=0)


if __name__ == '__main__':
    print(json.dumps(dict(scope='source loads only; not a throughput prediction', chunk=32256,
        queries_per_cta=2, cases=[budget(n) for n in (2000, 2051, 2672, 32000, 128000, 129775)]), indent=2))
