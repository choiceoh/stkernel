"""Bounded, explicit measurement capture for standalone draft sensitivity replay.

Disabled unless attach() is called after warmup. Forces host decode while armed;
capture traffic is NOT a performance baseline. Saves actual prepared readers and
pre-proposal rings, then labels them from committed greedy continuation, never
from target logits after a rejected draft. No target model is needed for replay.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cpu(t):
    # Packed views can share a huge serving arena; serialize only owned bytes.
    return t.detach().to('cpu', copy=True).contiguous()


def agreed(comm, call):
    """A local file/preparation failure must be heard before the next collective."""
    result, error = None, None
    try:
        result = call()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = comm.gather_objects(error)
    if any(errors):
        raise ValueError(f'draft replay preparation failed: {errors}')
    return result


def fp8_state(layer):
    if layer is None:
        return None
    return dict(rows=layer.rows, cols=layer.cols, q=cpu(layer.weight[0]), scale=cpu(layer.weight[1]))


def dense_state(layer):
    return dict(rows=layer.rows, cols=layer.cols, decode_precision=layer.decode_precision,
        decode_input_rows=list(layer.decode_input_rows), smooth=None if layer.smooth is None else cpu(layer.smooth),
        packs=[dict(data=cpu(p.data), scale=cpu(p.scale), rowscale=cpu(p.rowscale),
                    rows=p.rows, cols=p.cols, calibrated=p.calibrated) for p in layer.packs],
        fp8=fp8_state(layer.fp8), decode_fp8=fp8_state(layer.decode_fp8))


def save_prepared(d, directory, checkpoint):
    """Only proposal readers: FC/context precision cannot be changed with a fixed ring."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'state.pt').exists():
        raise ValueError('capture directory already contains a state; choose a new directory')
    names = [k for k in d.p if k == 'norm.weight' or k.startswith('candidate_selector.')
             or k.endswith(('layernorm.weight', '.q_norm.weight', '.k_norm.weight', '.base_kernel'))]
    from engine.base.kernel_shape import bound
    transport = d.target.comm.transport
    if transport is not None and type(transport).__name__ != 'OneShot':
        raise ValueError('unsupported capture transport')
    state = dict(version=1, facts=asdict(d.F), kernel_shape=asdict(bound()), world=d.target.comm.world_size, rank=d.target.comm.rank,
        transport=None if transport is None else dict(rails=transport.rails, inline_flags=transport.inline_flags,
                                                      consumer_max_elements=transport.consumer_max_elements),
        decodable=d.decodable, vp=d.target.vp, local_heads=d.local_heads, local_kv_heads=d.local_kv_heads,
        p={k: cpu(d.p[k]) for k in names},
        dense={k: dense_state(v) for k, v in d.dense.items() if k.startswith('layers.')},
        head=fp8_state(d.target.dense['head']), selector_alpha=list(d.selector_alpha),
        selector_projection_fp32=d.tuning.selector_projection_fp32,
        checkpoint_sha256=sha256(checkpoint), torch=str(torch.__version__),
        source_sha256={name: sha256(Path(__file__).parents[3] / name) for name in (
            'engine/profiles/glm53/drafter.py', 'engine/profiles/glm53/draft_replay.py',
            'engine/kernels/dense/__init__.py', 'engine/kernels/dense/packing.py',
            'engine/kernels/dense/kernels.cu', 'engine/kernels/dense/fp8.py',
            'engine/kernels/draft_attention.py', 'engine/kernels/draft_conv.py',
            'engine/kernels/draft_select.py', 'engine/modules/draft_projection.py',
            'engine/kernels/common/norm_rope.py', 'engine/kernels/common/swiglu.py',
            'engine/kernels/common/vocab_candidates.py', 'engine/modules/vocab.py',
            'engine/base/comm.py', 'engine/kernels/oneshot/__init__.py')})
    torch.save(state, directory / 'state.pt')
    identity = sha256(directory / 'state.pt')
    (directory / 'manifest.json').write_text(json.dumps(dict(version=1, state_sha256=identity,
        rank=state['rank'], world=state['world'], checkpoint_sha256=state['checkpoint_sha256'],
        scope='C1 greedy proposal-only; fixed baseline context; no capture timing'), indent=2) + '\n')
    return identity


class Capture:
    def __init__(self, engine, directory, identity, *, max_cases=16, every=16, per_request=2):
        if (type(max_cases) is not int or not 1 <= max_cases <= 128 or type(every) is not int or every < 1
                or type(per_request) is not int or not 1 <= per_request <= 128):
            raise ValueError('capture needs 1..128 cases, a positive interval and a bounded request quota')
        self.engine, self.directory, self.identity = engine, Path(directory), identity
        self.max_cases, self.every, self.per_request = max_cases, every, per_request
        self.pending, self.captured, self.seen, self.counts = [], 0, {}, {}
        self.current = None
        self.original_propose = engine.drafter.propose
        self.original_decode = engine.decode
        self.original_close = engine.close
        self.original_async_ready = engine.async_ready

    def eligible(self, seq):
        e = self.engine
        generated = e._generated_count(seq)
        return (e.limits[seq][1] == 0 and not e._rich(seq)
                and not e._reasoning_boundary(seq)
                and e.min_new.get(seq, 0) <= generated
                and e.limits[seq][0] - generated >= e.drafter.k + 1)

    def install(self):
        self.engine.drafter.propose = self.propose
        self.engine.decode = self.decode
        self.engine.close = self.close
        self.engine.async_ready = lambda seqs: False
        return self

    def propose(self, anchor, position, ring, **kwargs):
        drafts = self.original_propose(anchor, position, ring, **kwargs)
        seq = self.current
        if seq is None or self.captured >= self.max_cases or kwargs.get('boundary') is not None or not self.eligible(seq):
            return drafts
        if self.counts.get(seq, 0) >= self.per_request:
            return drafts
        self.seen[seq] = self.seen.get(seq, 0) + 1
        if (self.seen[seq] - 1) % self.every:
            return drafts
        d, e = self.engine.drafter, self.engine
        case_id = f'case-{self.captured:05d}'
        ids = torch.tensor([anchor] + [d.F.mask_id] * d.k, device=ring.device, dtype=torch.int64)
        # Same collective on every rank, outside capture/timing; avoids exporting the whole target embedding.
        embedding = d.target.embed(ids)
        blob = dict(version=1, state_sha256=self.identity, case_id=case_id, seq=int(seq),
            request_key=f'{seq}:{getattr(e, "nonces", {}).get(seq, 0)}',
            start=len(e.tokens[seq]), anchor=int(anchor), position=int(position),
            baseline_drafts=list(map(int, drafts)), ring=cpu(ring), embedding=cpu(embedding),
            ids=cpu(ids), temperature=0., k=d.k)
        def write():
            path = self.directory / (case_id + '.pt')
            torch.save(blob, path)
            return sha256(path)
        digest = agreed(d.target.comm, write)
        self.pending.append(dict({k: blob[k] for k in ('case_id', 'seq', 'start', 'k')}, snapshot_sha256=digest))
        self.captured += 1
        self.counts[seq] = self.counts.get(seq, 0) + 1
        return drafts

    def labels(self, finished=()):
        e, keep = self.engine, []
        for case in self.pending:
            seq, start, k = case['seq'], case['start'], case['k']
            tokens = list(map(int, e.tokens.get(seq, [])[start:start + k]))
            terminal = seq in finished or any(t in e.ends.get(seq, e.eos) for t in tokens)
            if len(tokens) < k and not terminal:
                keep.append(case)
                continue
            meta = dict(case_id=case['case_id'], snapshot_sha256=case['snapshot_sha256'],
                        target=tokens, complete=len(tokens) == k,
                        terminal=terminal, label_source='committed_greedy_continuation')
            agreed(e.drafter.target.comm, lambda: (self.directory / (case['case_id'] + '.json')).write_text(json.dumps(meta) + '\n'))
        self.pending = keep
        if self.captured >= self.max_cases and not self.pending:
            self.engine.async_ready = self.original_async_ready

    def decode(self, seqs, blocks, slots):
        # C>1 cannot silently become the C1 cost/benefit dataset.
        self.current = seqs[0] if len(seqs) == 1 else None
        try:
            result = self.original_decode(seqs, blocks, slots)
        finally:
            self.current = None
        self.labels([s for s, done in zip(seqs, result) if done])
        return result

    def close(self, seq):
        self.labels([seq])
        self.seen.pop(seq, None)
        self.counts.pop(seq, None)
        return self.original_close(seq)


def attach(engine, directory, checkpoint, *, max_cases=16, every=16):
    """Call on ALL ranks after warmup, before requests; never hot-attach to an active pipeline."""
    d = engine.drafter
    if not d.k or not d.fast_attention or not d.dense or engine.pipeline is not None and engine.pipeline.pending:
        raise ValueError('capture needs a prepared drafter and an idle pipeline')
    directory = Path(directory) / f'rank{d.target.comm.rank}'
    error, identity = None, None
    try:
        identity = save_prepared(d, directory, checkpoint)
        recorder = Capture(engine, directory, identity, max_cases=max_cases, every=every)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = d.target.comm.gather_objects(error)
    if any(errors):
        raise ValueError(f'draft replay capture preparation failed: {errors}')
    return recorder.install()
