"""Admitted TP4 NCCL proposal/capture and prefill transport check; no model weights."""
import json
import os
import sys
import time

import torch
from engine.base.comm import Comm
from engine.modules.draft_agreement import agree_walk


def bounded_candidate_walk(comm):
    """Compose the real candidate selector and agreement as a retained child.

    Model GEMMs are outside this small gate. Unlike the earlier ID-only gate,
    it includes the actual top-k gather that failed inside full-model capture.
    NCCL supplies the eager reference before any bounded replay is launched.
    """
    from engine.kernels.bounded_graph import BoundedGraph, append_child
    from engine.kernels.draft_select import walk_scores
    from engine.modules.vocab import topk
    reference_comm = Comm(comm.world_size, comm.rank, comm.group)
    checks = []
    width, k, drafts, projection = 256, 16, 6, 256
    vocabulary = width * comm.world_size
    for rows in (1, 4):
        logits = torch.empty(rows * drafts, width, device='cuda', dtype=torch.bfloat16)
        banks = torch.empty(4, *logits.shape, device='cuda', dtype=logits.dtype)
        anchors = torch.arange(rows, device='cuda', dtype=torch.int64)
        projected = ((torch.arange(rows * drafts * projection, device='cuda') % 11 - 5)
                     .float().view(rows, drafts, projection) * (comm.rank + 1) / 32)
        table = torch.arange(vocabulary * projection, device='cuda').view(vocabulary, projection)
        pred, succ = ((table % 13 - 6).float() / 32).bfloat16(), ((table % 17 - 8).float() / 32).bfloat16()
        count = torch.zeros(1, device='cuda', dtype=torch.int64)
        stop = torch.zeros_like(count)
        cutoff = torch.full_like(count, 9)
        histories = (torch.empty(4, rows * drafts, k, device='cuda'),
                     torch.empty(4, rows * drafts, k, device='cuda', dtype=torch.int64),
                     torch.empty(4, rows, drafts, device='cuda', dtype=torch.int64))

        def fill(trial):
            values = torch.arange(banks.numel(), device='cuda').view_as(banks)
            # Tied scores, distinct ranks, different replays and loop iterations.
            banks.copy_(((values * (trial + 1) + comm.rank * 19) % 23 - 11).bfloat16())
            anchors.copy_(torch.arange(rows, device='cuda') + trial * 7)

        def propose(group):
            scores, ids = topk(logits, group, comm.rank * width, k, vocabulary)
            walk = walk_scores(scores.view(rows, drafts, k), ids.view(rows, drafts, k),
                               anchors, projected, pred, succ)
            return scores, ids, agree_walk(group, walk)

        fill(0)
        logits.copy_(banks[0])
        propose(comm)  # compile and initialize all kernels before capture
        torch.cuda.synchronize()
        child = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(child):
            selected = propose(comm)
        try:
            for limit in (1, 2, 4):
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph):
                    logits.copy_(banks.index_select(0, count).squeeze(0))
                    append_child(child.raw_cuda_graph())
                    for history, value in zip(histories, selected):
                        history.index_copy_(0, count, value.unsqueeze(0))
                    stop.copy_(comm.all_reduce_max(((count == cutoff) & (comm.rank == 1)).to(torch.int64)))
                loop = BoundedGraph(graph, count, stop, limit,
                                    owners=(child, logits, banks, anchors, projected, pred, succ, histories, selected, cutoff))
                try:
                    for trial in (1, 2):
                        fill(trial)
                        expected = []
                        for iteration in range(limit):
                            logits.copy_(banks[iteration])
                            expected.append(tuple(x.clone() for x in propose(reference_comm)))
                        for end in (0, 1, 9):
                            for history in histories:
                                history.fill_(-999)
                            cutoff.fill_(end); stop.zero_()
                            loop.replay()
                            torch.cuda.synchronize()
                            completed = min(limit, end + 1)
                            if count.item() != completed:
                                raise AssertionError('candidate child iteration count differs')
                            for iteration in range(completed):
                                for history, value in zip(histories, expected[iteration]):
                                    torch.testing.assert_close(history[iteration], value, rtol=0, atol=0)
                            for history in histories:
                                torch.testing.assert_close(history[completed:], torch.full_like(history[completed:], -999),
                                                           rtol=0, atol=0)
                    checks.append(dict(rows=rows, proposal_rows=rows*drafts, keys_per_rank=rows*drafts*k,
                                       limit=limit, changed_replays=2, rank1_stop_positions=[0, 1, 9],
                                       exact_topk_scores_ids_and_root_walk=True, retained_child=True))
                finally:
                    loop.close()
                    graph.reset()
        finally:
            child.reset()
    return checks


def bounded_agreement(comm):
    """The served WHILE wrapper, not merely an ordinary CUDA replay."""
    from engine.kernels.bounded_graph import BoundedGraph, build
    build()
    counter = torch.zeros(1, device='cuda', dtype=torch.int64)
    stop = torch.zeros_like(counter)
    legacy_ids = torch.arange(6, device='cuda', dtype=torch.int64) + 100 * comm.rank
    # Retain the previous NCCL-only composition as a small diagnostic control.
    # No baseline model is loaded and a rejected graph is never replayed.
    comm.broadcast_tensor(legacy_ids)
    torch.cuda.synchronize()
    legacy = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(legacy):
        comm.broadcast_tensor(legacy_ids)
    try:
        control = BoundedGraph(legacy, counter, stop, 4, owners=(legacy_ids,))
    except RuntimeError as error:
        legacy_result = str(error)
    else:
        legacy_result = 'accepted by this standalone composition'
        control.close()
    finally:
        legacy.reset()
    print(json.dumps(dict(rank=comm.rank, legacy_bounded_control=legacy_result)), flush=True)
    comm.prepare_oneshot()
    checks = []
    for rows in (1, 4):
        for limit in (1, 2, 4):
            seed = torch.arange(rows * 6, device='cuda', dtype=torch.int64).view(rows, 6)
            proposals = torch.empty_like(seed)
            history = torch.empty((4, rows, 6), device='cuda', dtype=torch.int64)
            stop_at = torch.full((1,), 9, device='cuda', dtype=torch.int64)
            def body():
                proposals.copy_(seed + counter.view(1, 1))
                shared = agree_walk(comm, proposals)
                history.index_copy_(0, counter, shared.unsqueeze(0))
                vote = ((counter == stop_at) & (comm.rank == 1)).to(torch.int64)
                stop.copy_(comm.all_reduce_max(vote))
            counter.zero_(); stop.zero_()
            body()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                body()
            loop = BoundedGraph(graph, counter, stop, limit,
                                owners=(seed, proposals, history, stop_at))
            try:
                for turn in (1, 2):
                    for cutoff in (0, 1, 9):
                        seed.copy_(torch.arange(rows * 6, device='cuda').view(rows, 6)
                                   + 100 * turn + 1000 * comm.rank)
                        history.fill_(-1); stop.zero_(); stop_at.fill_(cutoff)
                        loop.replay()
                        torch.cuda.synchronize()
                        count = min(limit, cutoff + 1)
                        if counter.item() != count:
                            raise AssertionError('bounded agreement iteration count differs')
                        base = torch.arange(rows * 6, device='cuda').view(rows, 6) + 100 * turn
                        expected = base.unsqueeze(0) + torch.arange(count, device='cuda').view(count, 1, 1)
                        torch.testing.assert_close(history[:count], expected, rtol=0, atol=0)
                        torch.testing.assert_close(history[count:], torch.full_like(history[count:], -1), rtol=0, atol=0)
                checks.append(dict(rows=rows, limit=limit, changed_replays=2,
                                   rank1_stop_positions=[0, 1, 9], exact_history=True))
            finally:
                loop.close()
                graph.reset()
    return dict(legacy_control=legacy_result, checks=checks, candidate_walk=bounded_candidate_walk(comm))


def main():
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    rank = int(sys.argv[1])
    os.environ['MASTER_PORT'] = '18132'
    comm = Comm.init(rank, 4, timeout_s=60)
    start = time.monotonic()
    checks = []
    try:
        for rows in (1, 4):
            for sampled in (False, True):
                ids = torch.empty(rows, 6, device='cuda', dtype=torch.int64)
                support = torch.empty(rows, 6, 16, device='cuda', dtype=torch.int64)
                probabilities = torch.empty_like(support, dtype=torch.float32)
                def values(turn, source):
                    tokens = torch.arange(rows * 6, device='cuda').view(rows, 6) + 100 * turn + 1000 * source
                    candidates = tokens.unsqueeze(-1) + torch.arange(16, device='cuda')
                    p = torch.softmax(torch.arange(16, device='cuda').float() * (.1 + source), -1)
                    return tokens, candidates, p.expand(rows, 6, 16).contiguous()
                def fill(turn):
                    for into, value in zip((ids, support, probabilities), values(turn, rank)):
                        into.copy_(value)
                def run():
                    return agree_walk(comm, ids, support, probabilities) if sampled else agree_walk(comm, ids)
                def check(result, turn):
                    expected = values(turn, 0)
                    for got, ref in zip(result if sampled else (result,), expected):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)
                fill(0)
                check(run(), 0)
                fill(0)
                run()  # prepare NCCL/allocation before capture
                torch.cuda.synchronize()
                comm.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = run()
                for turn in (1, 2):
                    fill(turn)
                    graph.replay()
                    torch.cuda.synchronize()
                    check(result, turn)
                checks.append(dict(rows=rows, sampled=sampled, eager=True, changed_replays=2))
                del graph

        from engine.kernels.prefill_collectives import PrefillCollectives
        from engine.modules.token_shards import TokenShards
        transport = PrefillCollectives(comm, project_tiles=True)
        for rows in (511, 512, 668, 2304, 576):
            # Exactly representable values also provide an independent sum
            # oracle after the FP8 gather; arbitrary FP8 GEMM inputs are tested
            # by test_engine_prefill_tiles_cuda in the same admitted bracket.
            x = ((torch.arange(rows * 4096, device='cuda') % 8).view(rows, 4096).float()
                 + rank).bfloat16()
            full = transport.all_gather(x)
            actual = transport.gather_project(x, lambda value: value * 2)
            torch.testing.assert_close(actual, full * 2, rtol=0, atol=0)
            reduced = transport.reduce_scatter(full)
            torch.testing.assert_close(reduced, (full * 4).chunk(4)[rank], rtol=0, atol=0)
        ragged_rows = (2047, 2048, 2121, 2122, 2123, 2124, 2128, 2671, 2672, 32002, 32003)
        for rows in ragged_rows:
            view = TokenShards(transport, rows, rank)
            full = (torch.arange(rows * 4096, device='cuda') % 8).view(rows, 4096).bfloat16()
            shard = view.shard(full)
            torch.testing.assert_close(view.all_gather(shard), full, rtol=0, atol=0)
            torch.testing.assert_close(view.gather_project(shard, lambda x: x * 2 + 3),
                                       full * 2 + 3, rtol=0, atol=0)
            reduced = view.reduce_scatter(full + rank)
            torch.testing.assert_close(view.gather_result(reduced), full * 4 + 6, rtol=0, atol=0)
            last = comm.all_gather(shard[view.last_local:view.last_local+1], dim=0)[-1:]
            torch.testing.assert_close(last, full[-1:], rtol=0, atol=0)
        torch.cuda.synchronize()
        bounded = bounded_agreement(comm)
        print(json.dumps(dict(passed=True, rank=rank, checks=checks,
                              transport_rows=[511, 512, 668, 2304, 576], bounded=bounded,
                              real_rows=ragged_rows,
                              seconds=time.monotonic() - start,
                              scope='real TP4 NCCL and one-shot, changed ordinary and bounded graph replay, FP8/BF16 transport; not consumer speed')),
              flush=True)
    finally:
        comm.close()


if __name__ == '__main__':
    main()
