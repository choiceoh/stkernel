"""Locate a TP4 captured decode stall using real weights and runner steps.

Run on four private GB10 containers. Layer subsets check execution only;
--full also checks generated language. --trace synchronizes each graph family
to locate the last completed boundary; omit it to exercise serving overlap.
"""
import argparse
import json
from pathlib import Path
import time
import weakref

import torch

from engine.base.comm import Comm
from engine.base.instruments import Recorder
from engine.profiles.glm53.boot import build
from engine.profiles.glm53.lanes import served


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--drafter-dir', required=True)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--max-seqs', type=int, default=4)
    ap.add_argument('--ceiling', type=int, default=4096)
    ap.add_argument('--kv-gib', type=float, default=.75)
    ap.add_argument('--prefill-warmup', type=int, default=128)
    ap.add_argument('--max-new', type=int, default=24)
    args = ap.parse_args()
    comm = Comm.init(world=4, timeout_s=120)
    engine = None
    started = time.monotonic()

    def emit(stage, **data):
        print(json.dumps(dict(rank=comm.rank, seconds=round(time.monotonic()-started, 3),
                              stage=stage, **data), ensure_ascii=False), flush=True)

    def trace(obj, name, label):
        original = getattr(obj, name)
        def call(*a, **kw):
            emit(label+'/begin')
            result = original(*a, **kw)
            torch.cuda.synchronize()
            emit(label+'/done')
            return result
        setattr(obj, name, call)

    try:
        from engine.kernels.b12x import moe_dispatch as md
        get_workspace = md._get_cached_workspace
        seen = {}
        def workspace(**kw):
            result = get_workspace(**kw)
            if id(result) not in seen or seen[id(result)]() is None:
                seen[id(result)] = weakref.ref(result)
                rows = getattr(result, 'routed_rows_capacity', result.max_rows)
                emit('workspace/new', owner=id(result), rows=rows, backend=kw['backend'])
                weakref.finalize(result, emit, 'workspace/freed', owner=id(result), rows=rows)
            return result
        md._get_cached_workspace = workspace
        emit('load/begin')
        F, net, caches, engine, runner = build(
            comm, None if args.full else [0, 3], served(), args.ranks, args.kv_gib,
            args.max_seqs, True, Recorder('replay'), max_new=args.max_new,
            context_ceiling=args.ceiling, ckpt_meta=args.metadata, drafter_dir=args.drafter_dir)
        emit('load/done', layers=net.layers)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.metadata)
        tok.chat_template = (Path(args.metadata)/'chat_template_mm_v2.jinja').read_text()
        prompt_text = tok.apply_chat_template(
            [dict(role='user', content='대한민국의 수도를 한 단어로 답해줘.')],
            tokenize=False, add_generation_prompt=True, thinking=False)
        prompt = tok.encode(prompt_text, add_special_tokens=False)
        runner.keep_idle = True
        # A bounded diagnostic explicitly qualifies only this prefill size.
        engine.prefill_chunk = min(engine.prefill_chunk, args.prefill_warmup)
        emit('capture/begin')
        engine.capture_decode(args.max_seqs)
        emit('capture/done', shapes=list(engine.decode_graphs.graphs.graphs),
             workspace_owners=len(engine.decode_graphs.graphs.resources))
        if engine.memory is not None:
            engine.memory.write(f'/repo/replay-memory-rank{comm.rank}.json')
        if args.trace:
            trace(engine.drafter, 'propose', 'draft')
            trace(engine.decode_graphs, 'run', 'target')
            trace(engine.sampling_graphs, 'run', 'sampler')
            trace(engine.drafter, 'observe_decode', 'observe')
        # One request first, followed by the complete declared batch.
        widths = set()
        for batch in sorted({1, args.max_seqs}):
            for seq in range(batch):
                engine.add(seq, prompt, max_new=args.max_new,
                           min_new=args.max_new if batch > 1 else 0)
                # The injected scheduler clock admits waiting requests through
                # the real starvation valve, exercising growing decode batches.
                runner.submit(seq, len(prompt), now=0.)
            step = 0
            while not all(seq in runner.idle for seq in range(batch)):
                emit('runner/begin', batch=batch, step=step)
                plan = runner.step(now=60.)
                if plan.kind == 'decode':
                    widths.add(len(plan.seqs))
                emit('runner/done', batch=batch, step=step,
                     kind=plan.kind, seqs=plan.seqs,
                     generated=[engine.generated(seq) for seq in range(batch)])
                step += 1
                assert step <= batch*(len(prompt)+args.max_new+1), 'runner made no progress'
            for seq in range(batch):
                ids = engine.generated(seq)
                text = tok.decode(ids)
                emit('result', batch=batch, seq=seq, ids=ids, text=text)
                if args.full:
                    assert '서울' in text, f'full-model answer did not contain Seoul: {text!r}'
                runner.cancel(seq)
                engine.forget(seq)
        assert widths == set(range(1, args.max_seqs+1)), widths
        emit('PASS', decode_widths=sorted(widths))
    finally:
        if engine is not None:
            engine.close_decode()
        comm.close()


if __name__ == '__main__':
    main()
