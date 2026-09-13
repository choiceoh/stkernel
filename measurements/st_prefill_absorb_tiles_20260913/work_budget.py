"""Explicit layout-copy payload only; not DRAM traffic or a speed forecast."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.profiles.glm53.facts import architecture


def budget(rows, facts, chunk=32256):
    heads = facts.heads // 4
    eligible = [min(chunk, rows-start) for start in range(0, rows, chunk)
                if 128 <= min(chunk, rows-start) <= 32768]
    covered = sum(eligible)
    query = covered*heads*facts.kv_lora*2
    output = covered*heads*facts.v_dim*2
    layers = sum(facts.is_dsa(L) for L in range(facts.layers))
    return dict(input_tokens=rows, eligible_tokens=covered, dsa_layers=layers,
                query_copy_payload_bytes=query, output_copy_payload_bytes=output,
                removed_copy_read_write_bytes_per_rank_dsa_layer=2*(query+output),
                removed_copy_read_write_gib_per_rank_request=2*(query+output)*layers/2**30,
                largest_removed_query_temporary_bytes=max(eligible, default=0)*heads*facts.kv_lora*2,
                largest_removed_output_temporary_bytes=max(eligible, default=0)*heads*facts.v_dim*2,
                new_persistent_bytes=0, new_collectives=0)


if __name__ == '__main__':
    facts = architecture(json.loads((ROOT/'measurements/st_prefill_dense_prefix_20260913/model-config.json').read_text()))
    print(json.dumps(dict(scope='source copy payload; not measured DRAM or TTFT', chunk=32256,
                         cases=[budget(n, facts) for n in (2000, 2672, 32000, 128000, 129025)]), indent=2))
