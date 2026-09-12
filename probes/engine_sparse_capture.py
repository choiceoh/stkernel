"""Bounded L3 expert-input capture from real text through all four TP ranks.

Runs the actual GLM prefix composition with native KDA chunk and explicit
reference lanes elsewhere, on one GPU using LocalTP. No live-service hooks,
full-model loading, generation, or modified checkpoints. This is a local
calibration corpus, not a language-model quality benchmark.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch

from engine.base.arena import Arena
from engine.base.comm import LocalTP
from engine.profiles.glm53 import facts, lanes
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.net import Glm53Net, Step
from engine.profiles.glm53.weights import rank_loader


def corpus(path):
    rows = json.loads(Path(path).read_text())
    if not rows or len(rows) > 128:
        raise ValueError('requires 1..128 prompts')
    seen_ids, seen_text = set(), set()
    for row in rows:
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('unknown split')
        if row['id'] in seen_ids or row['text'] in seen_text or not row['text'].strip():
            raise ValueError('duplicate or empty prompt')
        seen_ids.add(row['id']); seen_text.add(row['text'])
    return rows


def capture_lanes(tp):
    from engine.kernels.kda import chunk_kda_with_fused_gate
    ref = lanes.reference()

    def chunk(q, k, v, raw, beta, a, bias, initial, lower):
        output, state = chunk_kda_with_fused_gate(
            q=q, k=k, v=v, raw_g=raw, beta=torch.sigmoid(beta.float()),
            A_log=a.view(1, 1, -1, 1), g_bias=bias,
            initial_state=initial.transpose(-1, -2).contiguous() if initial is not None else None,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32),
            safe_gate=True, lower_bound=lower,
            out=torch.empty_like(v, memory_format=torch.contiguous_format))
        return output, state.transpose(-1, -2).contiguous()

    def mla(q, kv, slots, valid, scale, ckv_scale):
        # Bound the reference's [T, selected_keys, 512] FP32 temporary.
        return torch.cat([ref.mla_sparse(q[i:i+16], kv, slots[i:i+16],
                         valid[i:i+16], scale, ckv_scale) for i in range(0, len(q), 16)])

    return replace(ref, name='native KDA chunk; other lanes reference',
                   kda_chunk=lambda *a: tp.on_main(chunk, *a), mla_sparse=mla)


@torch.inference_mode()
def capture_rank(comm, table, F, ranks, sequences):
    net = Glm53Net(F, comm, table, layers=range(4))
    loader = rank_loader(ranks / f'rank{comm.rank}of4.safetensors')
    count = len(sequences)
    arena = Arena(layout(F, net.layers).nbytes(count * 2, count) + (1 << 20))
    caches = Glm53Caches(arena, F, net.layers, count * 2, count)
    caches.reset()
    chunks = []
    for i, ids in enumerate(sequences):
        caches.pool.reserve(i, len(ids))
        slot = caches.slots.take(i)
        chunks.append((torch.tensor(ids, device='cuda', dtype=torch.int64), 0, i, slot))
    step = Step.decode(chunks)
    caches.prepare(step)
    net.p = loader.load(['embed'], device='cuda', max_run=32 << 20)
    x = net.embed(step.ids)
    res = x[:, None, :].expand(-1, F.hc, -1).contiguous()
    post = comb = None
    for layer in range(4):
        if post is not None:
            res = table.mhc_post(x, res, post, comb)
        net.p = None
        keys = [s.name for s in net.specs() if s.name.startswith(f'L{layer}.')
                and (layer != 3 or '.moe.' not in s.name
                     or s.name.endswith(('.moe.gate', '.moe.bias')))]
        net.p = loader.load(keys, device='cuda', max_run=32 << 20)
        post, comb, x = net._hc_pre(layer, res, 'attn')
        x = net._dsa(layer, x, step, caches) if F.is_dsa(layer) else net._kda(layer, x, step, caches)
        res = table.mhc_post(x, res, post, comb)
        post, comb, x = net._hc_pre(layer, res, 'ffn')
        if layer == 3:
            selected, coefficient = net.route(layer, x)
            assert torch.isfinite(x).all() and torch.isfinite(coefficient).all()
            return x.cpu(), selected.cpu(), coefficient.cpu()
        x = net._dense(layer, x)


def main():
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ranks', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if not 1 <= args.batch <= 4 or not 32 <= args.max_tokens <= 512:
        ap.error('bounded capture: batch 1..4, max tokens 32..512')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((6 << 30) / torch.cuda.get_device_properties(0).total_memory)
    prompts = corpus(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    F = facts.load(args.metadata)
    tokenizer = AutoTokenizer.from_pretrained(args.metadata, local_files_only=True)
    template_path = Path(__file__).resolve().parents[1] / 'launchers/chat_template_mm_v2.jinja'
    template = template_path.read_text()
    raw_tokenizer = Tokenizer.from_file(str(args.metadata / 'tokenizer.json'))
    rendered = [tokenizer.apply_chat_template([dict(role='user', content=p['text'])],
                chat_template=template, add_generation_prompt=True, tokenize=False,
                thinking=False) for p in prompts]
    sequences = [raw_tokenizer.encode(text, add_special_tokens=False).ids[:args.max_tokens]
                 for text in rendered]
    tp = LocalTP(4, timeout_s=600)
    table = capture_lanes(tp)
    report = dict(scope=__doc__, layer=3, rank_of_expert_weights=0,
                  lanes=table.name, collective='LocalTP; FP32 rank-order sum then BF16',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                  template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                  max_tokens=args.max_tokens, discard_prefix_tokens=8, batches=[], prompts=[])
    all_x, all_sel, all_c, all_p = [], [], [], []
    started = time.monotonic()
    for start in range(0, len(prompts), args.batch):
        subset = sequences[start:start+args.batch]
        parts = tp.run(lambda comm: capture_rank(comm, table, F, args.ranks, subset))
        x, selected, coefficient = parts[0]
        for other in parts[1:]:
            for a, b in zip(parts[0], other):
                if not torch.equal(a, b):
                    raise RuntimeError('replicated L3 inputs/router differ across TP ranks')
        at = 0
        for i, ids in enumerate(subset):
            p = prompts[start+i]
            keep = slice(at+8, at+len(ids))
            all_x.append(x[keep].clone()); all_sel.append(selected[keep].clone())
            all_c.append(coefficient[keep].clone())
            all_p.append(torch.full((len(ids)-8,), start+i, dtype=torch.int32))
            report['prompts'].append(dict(id=p['id'], split=p['split'], token_ids=ids,
                                           kept_tokens=len(ids)-8))
            at += len(ids)
        row = dict(start=start, tokens=sum(map(len, subset)),
                   seconds=time.monotonic()-started, tp_replicas_exact=True)
        report['batches'].append(row)
        print(json.dumps(row), flush=True)
    payload = dict(x=torch.cat(all_x), selected=torch.cat(all_sel),
                   coefficient=torch.cat(all_c), prompt_index=torch.cat(all_p))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report.update(tensor_file_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                  peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                  total_kept_tokens=len(payload['x']), gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__)
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
