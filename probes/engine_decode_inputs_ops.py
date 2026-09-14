"""Inventory the retained CPU reference's tensor operations; this is not GPU timing."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from engine.base.draws import step_block
    from engine.modules.draft_inputs import build
    class Inventory(TorchDispatchMode):
        def __init__(self): self.operations, self.materializing = Counter(), Counter()
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.operations[str(func)] += 1
            if all(result.alias_info is None for result in func._schema.returns):
                self.materializing[str(func)] += 1
            return func(*args, **(kwargs or {}))
    records = []
    for rows in (1, 4):
        anchors, positions = torch.arange(rows), torch.arange(rows) + 128*1024
        for name, call in (('draft_inputs', lambda: build(anchors, positions, 7, 154879)),
                           ('step_block', lambda: step_block(19, anchors, positions, 7))):
            with Inventory() as inv: call()
            records.append(dict(rows=rows, k=7, path=name,
                                reference_materializing_tensor_ops=sum(inv.materializing.values()),
                                materializing=dict(sorted(inv.materializing.items())),
                                all_dispatches=dict(sorted(inv.operations.items())), candidate_kernel_launches=1))
    report = dict(scope='CPU reference dispatch inventory; no GPU launch count, timing or throughput measurement',
                  cases=records, source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                               ('engine/base/draws.py', 'engine/modules/draft_inputs.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
