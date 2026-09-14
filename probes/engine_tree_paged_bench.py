"""Matched CPU verify+commit comparison and exact bytes removed, never GPU speed proof."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from engine.modules.speculative_tree import Tree
from engine.profiles.glm53.tree_decode import Verification
from probes.engine_tree_fastpath_bench import bracket, load_baseline
from tests.test_engine_tree_decode import TreeDecodeTests

BASELINE = 'c58a8eb534ee5bf712f09f1ab68891ce02edcb85'


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(iterations, baseline=BASELINE):
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError('CPU comparison requires hidden CUDA devices')
    torch.set_num_threads(1)
    old, baseline_hashes = load_baseline(baseline)
    records = []
    for parents in ((-1, 0, 1, 2, 3, 4, 5, 6), (-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)):
        tree = Tree(tuple(range(1, len(parents)+1)), parents)
        for context in (0, 129, 1024):
            net, cache, slot = TreeDecodeTests().prepare(context)
            state, paged = cache.state.clone(), cache.paged.clone()
            def restore():
                cache.state.copy_(state)
                cache.paged.copy_(paged)
            def step(cls):
                with cls(net, cache, tree, seq=0, slot=slot, context=context) as v:
                    hidden = v.verify(aux_layers=(0, 2))
                    result = v.commit(budget=8)
                return hidden, result
            expected = step(old['decode'].Verification)
            expected_state, expected_paged = cache.state.clone(), cache.paged.clone()
            restore()
            actual = step(Verification)
            for got, want in ((actual[0], expected[0]), (actual[1]['features'], expected[1]['features']),
                              (cache.state, expected_state), (cache.paged, expected_paged)):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
            if (actual[1]['tokens'], actual[1]['path']) != (expected[1]['tokens'], expected[1]['path']):
                raise AssertionError('greedy outputs or committed path changed')
            timing = bracket(lambda: step(old['decode'].Verification), lambda: step(Verification), iterations, restore)
            records.append(dict(nodes=len(parents), context=context, **timing,
                output_sha256=digest(actual[0]), state_sha256=digest(expected_state), paged_sha256=digest(expected_paged),
                features_sha256=digest(actual[1]['features']), tokens=actual[1]['tokens'], path=actual[1]['path'],
                output_and_commit_bit_exact=True))
    paths = ['engine/profiles/glm53/tree_decode.py', 'engine/modules/tree_attention.py',
             'engine/modules/sparse_attention.py', 'engine/kernels/indexer.py', 'engine/profiles/glm53/lanes.py',
             'engine/kernels/mla/__init__.py', 'engine/kernels/mla/decode_absorb.py',
             'engine/kernels/mla/prefill_absorb.py', 'engine/kernels/mla/glm53_megakernel.cu',
             'engine/modules/tree_kda.py', 'engine/profiles/glm53/net.py', 'probes/engine_tree_paged_bench.py']
    report = dict(scope='tiny CPU full verify+commit; no proposal or GPU kernels; not production throughput',
        baseline=baseline, gpu_used=False, torch=torch.__version__, machine=platform.machine(),
        iterations_per_block=iterations, bracket='B/A/A/B repeated twice, 3 warmups each, restore excluded',
        cases=records, baseline_source_sha256=baseline_hashes,
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        unmeasured=['real-weight GPU numerics and graph replay', 'same-build production tok/s and TTFT',
                    'reasoning length and acceptance', 'full tree production graph integration'])
    return report


def run(output, iterations):
    report = compare(iterations)
    report['structural'] = [dict(nodes=n, selected_width=2051, latent=512,
            removed_final_latent_staging_bytes=n*2051*512,
            removed_minimum_staging_read_write_bytes=2*n*2051*512,
            new_latent_staging_bytes=0, direct_slot_bytes=n*2051*4,
            removed_query_output_layout_copy_bytes=n*16*(512+256)*2,
            address_launches=1) for n in (8, 15, 32)]
    write_report(output, report)


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([{k: v for k, v in row.items() if k in ('nodes', 'context', 'median_ms', 'reduction_percent')}
                      for row in report['cases']], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    run(args.output, args.iterations)
