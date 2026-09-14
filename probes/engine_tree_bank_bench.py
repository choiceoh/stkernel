"""Matched CPU tree comparison against #958; copy counts are not GPU timings."""
import argparse
import hashlib
from pathlib import Path

from probes.engine_tree_paged_bench import compare, write_report

BASELINE = '82f9aca4e034dbbe7ebf67d2f500c062f8b08734'


def run(output, iterations, baseline):
    report = compare(iterations, baseline)
    report['structural'] = dict(
        key_bank=[dict(context=context, pool=4, index_width=128,
            condition='one or more private pools completed in this tree',
            removed_prefix_temporary_bytes=(context//4)*(128+4),
            removed_prefix_concat_read_write_bytes=2*(context//4)*(128+4),
            candidate_gather_and_append_launches=1)
            for context in (32768, 131072)],
        convolution=dict(channels=6144, taps=4, dtype='bf16',
            removed_final_history_bytes=6144*3*2, removed_history_gather_and_mask_launches=2,
            candidate_history_staging_bytes=0, recurrent_state_dtype='fp32'),
        metadata='indexer starts, ends and branch columns created once per transaction, reused by DSA layers',
        scratch_admission_bound='unchanged and conservative')
    report['source_sha256'].update({p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ('engine/profiles/glm53/caches.py', 'engine/kernels/kda/tree.py', 'probes/engine_tree_bank_bench.py')})
    write_report(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    run(args.output, args.iterations, args.baseline)
