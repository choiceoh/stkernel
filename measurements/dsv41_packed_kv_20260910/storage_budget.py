#!/usr/bin/env python3
"""Static original versus packed KV bytes; no runtime/performance estimate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    config = args.reference_dir / "config.json"
    cfg = json.loads(config.read_text())
    assert cfg["head_dim"] == 512
    assert cfg["kv_source_layers"] == [2, 8, 14, 20]
    owners = cfg["kv_source_layers"]
    ratios = cfg["compress_ratios"]
    old_row, new_row = 512 * 2, 512 // 2 + 512 // 16
    rows = []
    for batch in (1, 4):
        for capacity in (2048, 32768, 131072, 1048576):
            positions = batch * sum(capacity // ratios[layer] for layer in owners)
            rows.append(dict(batch_size=batch, capacity_tokens=capacity,
                             owner_count=len(owners), stored_positions=positions,
                             bf16_allocated_bytes=positions * old_row,
                             packed_allocated_bytes=positions * new_row,
                             removed_allocated_bytes=positions * (old_row-new_row)))
    layers = sum(r > 0 for r in ratios[:cfg["n_layers"]])
    positions = layers * cfg["index_topk"]
    report = dict(schema=1,
        scope="Static tensor allocation and logical operand bytes, not measured memory reservation, HBM traffic or speed.",
        reference_sha256={name: hashlib.sha256((args.reference_dir/name).read_bytes()).hexdigest()
                          for name in ("config.json", "model.py", "kernel.py")},
        owner_layers=owners, owner_ratios=[ratios[layer] for layer in owners],
        row_bytes=dict(bf16=old_row, packed_payload=256, e4m3_scales=32, packed_total=new_row),
        reduction_percent=100*(old_row-new_row)/old_row, allocations=rows,
        max_selected_compressed_operand_per_decode_query=dict(
            layers=layers, positions_per_layer=cfg["index_topk"],
            bf16_bytes=positions*old_row, packed_bytes=positions*new_row,
            removed_bytes=positions*(old_row-new_row),
            caveat="Assumes 512 valid compressed selections per layer counted once. Head-CTA rereads, padding, caches and invalid slots change physical traffic."))
    with args.output.open("x") as out:
        out.write(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
