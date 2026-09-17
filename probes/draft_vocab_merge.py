"""Captured real-weight TP4 proposal A/B; baseline source validation stays strict."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from probes.draft_sensitivity import load_state, case_files, capture_graph, paired_time
from engine.profiles.glm53.draft_replay import agreed, sha256


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'checkpoint', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--rounds', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 100 or not torch.cuda.is_available():
        raise ValueError('native replay requires CUDA and 1..100 timing rounds')
    from engine.base.comm import Comm
    from engine.modules import vocab
    # Staged by the queue launcher, separately from capture-pinned engine files.
    from probes import vocab_merge_candidate as candidate
    comm = Comm.init()
    try:
        directory = args.capture / f'rank{comm.rank}'
        d, state, manifest = agreed(comm, lambda: load_state(directory, comm, 'cuda'))
        if comm.world_size != 4:
            raise ValueError('real capture comparison requires TP4')
        transports = comm.gather_objects(state['transport'])
        if any(t != transports[0] for t in transports):
            raise ValueError('ranks captured different collective transports')
        if state['transport'] is not None:
            comm.prepare_oneshot(**state['transport'])
        def load_cases():
            if sha256(args.checkpoint) != state['checkpoint_sha256']:
                raise ValueError('checkpoint identity mismatch')
            cases = case_files(directory, manifest)
            for c in cases:
                if (c['k'] != d.k or c['temperature'] != 0
                        or c['label_source'] != 'committed_greedy_continuation'
                        or c['ids'].tolist() != [c['anchor']] + [d.F.mask_id]*d.k):
                    raise ValueError('incompatible capture case')
            return cases
        cases = agreed(comm, load_cases)
        descriptors = [(c['case_id'], c['position'], c['baseline_drafts'], c['target']) for c in cases]
        if any(p != descriptors for p in comm.gather_objects(descriptors)):
            raise ValueError('ranks disagree on capture cases')
        root = Path(__file__).resolve().parents[1]
        report = dict(scope='Same fixed captured states, real TP4 drafter/head; no target verification or live tok/s',
            sampling_scope='T=1 proposal equality on greedy-captured contexts, not t=1 traffic acceptance',
            torch=str(torch.__version__), torch_git=torch.version.git_version,
            capability=torch.cuda.get_device_capability(), engine_booted=False,
            baseline_source_sha256=state['source_sha256'],
            candidate_source_sha256={p: sha256(root/p) for p in ('probes/vocab_merge_candidate.py',
                'engine/kernels/common/vocab_merge.py', 'probes/draft_vocab_merge.py')},
            rank_state_sha256=comm.gather_objects(manifest['state_sha256']), cases=[])
        original = vocab.topk
        dense = d.candidate_buffer
        compact = candidate.CandidateBuffer(d.k, d.target.vp*4, d.F.sel_top_k*4, 'cuda', compact=True)
        for case in cases:
            ring = case['ring'].to('cuda')
            field, slot = ring.unsqueeze(0), torch.zeros(1, device='cuda', dtype=torch.int64)
            position = torch.tensor(case['position'], device='cuda', dtype=torch.int64)
            ids, embedding = case['ids'].to('cuda'), case['embedding'].to('cuda')
            def embed(token_ids):
                if token_ids.shape != ids.shape:
                    raise ValueError('proposal embedding geometry changed')
                return embedding
            d.target.embed = embed
            anchor, before = ids[:1], ring.clone()
            record = dict(case_id=case['case_id'], request_key=case['request_key'], position=case['position'])
            for mode in ('greedy', 'sampled_t1'):
                uniforms = torch.tensor([.01,.21,.41,.61,.81,.99,.5], device='cuda')[:d.k]
                call = (lambda: d.propose_tensor(anchor, position, (field, slot))) if mode == 'greedy' else (
                    lambda: d.propose_sampled_tensor(anchor, position, ring, 1., uniforms, state['decodable']))
                graphs, outputs = [], []
                try:
                    for fn, buffer in ((original, dense), (candidate.topk, compact)):
                        d.candidate_buffer = buffer
                        with patch.object(vocab, 'topk', fn):
                            graph, output = capture_graph(call)
                        graphs.append(graph); outputs.append(output)
                    a, b = outputs
                    if mode == 'greedy':
                        exact = torch.equal(a, b) and a.tolist() == case['baseline_drafts']
                    else:
                        exact = all(torch.equal(x, y) for x, y in zip(a, b))
                    def assert_exact():
                        if not exact or not torch.equal(before, ring):
                            raise ValueError('candidate outputs drifted, baseline failed reproduction, or ring mutated')
                    agreed(comm, assert_exact)
                    timing = paired_time(graphs, comm, args.rounds)
                    # Read equality again after the complete timing replay.
                    after_exact = torch.equal(a, b) if mode == 'greedy' else all(torch.equal(x, y) for x, y in zip(a, b))
                    if not all(comm.gather_objects(after_exact and torch.equal(before, ring))):
                        raise ValueError('graph replay changed output equality or context')
                    tokens = b if mode == 'greedy' else b[0]
                    token_ids = tokens.tolist()
                    if any(peer != token_ids for peer in comm.gather_objects(token_ids)):
                        raise ValueError('ranks disagree on selected tokens')
                    record[mode] = dict(exact_all_ranks=True, tokens=token_ids, timing=timing)
                finally:
                    for graph in graphs: graph.reset()
            report['cases'].append(record)
            if comm.rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(record), flush=True)
        report.update(passed=True, peak_reserved_bytes_per_rank=comm.gather_objects(torch.cuda.max_memory_reserved()))
        if comm.rank == 0:
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    finally:
        comm.close()


if __name__ == '__main__':
    main()
