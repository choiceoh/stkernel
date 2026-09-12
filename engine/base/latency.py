"""Bounded, request-scoped onepass recording, on each serving rank.

Normal requests keep host spans and existing device-stage samples. Diagnostic
requests additionally profile one prefill chunk and four decode steps. Traces
are filed immediately; a lost client cannot make the next run overwrite them.
"""
from contextlib import contextmanager
from functools import wraps
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from engine.base import graph_labels
from engine.base.latency_trace import attribute

SCHEMA = 1
_PREPARATIONS = 0
_HOOKS = []
_OBSERVER = None


def install_preparation_observers():
    """Observe new specializations, including disk-cache loads, without changing results.

    Only already-imported compiler modules are touched. Missing coverage is
    reported and cannot silently be interpreted as a warmed compiler.
    """
    targets = [('triton.runtime.jit', 'JITFunction', '_do_compile'),
               ('cutlass.cute', None, 'compile'), ('torch.utils.cpp_extension', None, '_jit_compile')]
    for module, cls, name in targets:
        obj = sys.modules.get(module)
        obj = getattr(obj, cls, None) if cls else obj
        original = getattr(obj, name, None)
        if original is None or getattr(original, '_onepass_observer', False):
            continue
        def wrap(fn, label):
            @wraps(fn)
            def observed(*args, **kwargs):
                global _PREPARATIONS
                _PREPARATIONS += 1
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    if _OBSERVER is not None:
                        _OBSERVER.row(kind='preparation', operation=label, phase='compile_or_load',
                                      duration_us=(time.perf_counter() - start) * 1e6)
            observed._onepass_observer = True
            return observed
        setattr(obj, name, wrap(original, module + '.' + name))
        _HOOKS.append(module + '.' + name)


def preparation():
    return dict(specializations=_PREPARATIONS, graph_captures=graph_labels.CAPTURES,
                observers=list(_HOOKS), scope='Triton/CuTe/C++ JIT entry and ST graph capture; not arbitrary external compilers')


def _write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


class Recorder:
    def __init__(self, rank, root):
        self.rank, self.root = rank, Path(root)
        self.active = None
        self.last = None
        self.lock = threading.Lock()
        self.step_index = 0

    def begin(self, token, diagnostic=False, concurrency=1):
        global _OBSERVER
        if type(concurrency) is not int or concurrency not in (1, 4):
            raise ValueError('latency concurrency must be 1 or 4')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', token):
            raise ValueError('invalid latency request token')
        if self.active is not None:
            raise ValueError('another latency request is active')
        install_preparation_observers()
        directory = self.root / token / f'rank-{self.rank}'
        directory.mkdir(parents=True, exist_ok=False)
        self.active = dict(token=token, diagnostic=bool(diagnostic), directory=directory,
                           started=time.monotonic(), before=preparation(), profiles=[], rows=[], errors=[],
                           kinds={'prefill': 0, 'decode': 0}, selected={'prefill': 0, 'decode': 0},
                           concurrency=concurrency, row_count=0)
        self.file = (directory / 'latency.jsonl').open('x')
        _write(directory / 'manifest.json', dict(schema=SCHEMA, token=token, rank=self.rank,
               status='running', diagnostic=bool(diagnostic), preparation_before=self.active['before']))
        self.step_index = 0
        _OBSERVER = self
        return dict(rank=self.rank, token=token, status='recording', preparation=self.active['before'])

    def row(self, **value):
        run = self.active
        if run is None:
            return
        if run['row_count'] >= 100000:
            if not run['errors'] or run['errors'][-1] != '100000-row recording limit reached':
                run['errors'].append('100000-row recording limit reached')
            return
        row = dict(rank=self.rank, request_token=run['token'], **value)
        with self.lock:
            self.file.write(json.dumps(row, ensure_ascii=False) + '\n')
            run['row_count'] += 1
        # Basic rows remain available in the control reply; kernel rows live in
        # the trace artifact and are summarized offline, not copied twice.
        if row.get('kind') != 'gpu_activity':
            run['rows'].append(row)

    @contextmanager
    def step(self, kind, seqs, positions, tokens):
        run = self.active
        if run is None:
            yield
            return
        self.step_index += 1
        index = self.step_index
        run['kinds'][kind] += 1
        selected = (run['diagnostic'] and run['selected'][kind] < (1 if kind == 'prefill' else 4)
                    and (kind == 'prefill' or len(seqs) == run['concurrency']))
        if selected:
            run['selected'][kind] += 1
        prof = None
        if selected:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA activity profiling unavailable')
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
                prof.__enter__()
                graph_labels.profiling(True)
            except Exception as exc:
                run['errors'].append('profile start: ' + str(exc))
                prof = None
        before, start = preparation(), time.perf_counter()
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            duration = (time.perf_counter() - start) * 1e6
            self.row(kind='host_step', operation='runner', phase=kind, step=index, rows=list(seqs),
                     positions=positions, tokens=tokens, duration_us=duration,
                     timing_scope='host launch/runner work; asynchronous GPU completion is separate',
                     profiled=prof is not None, preparation_before=before, preparation_after=preparation(), error=error)
            graph_labels.profiling(False)
            if prof is not None:
                try:
                    prof.__exit__(None, None, None)
                    path = run['directory'] / f'{kind}-{index}.trace.json'
                    prof.export_chrome_trace(str(path))
                    # Compression and parsing occur AFTER the selected step.
                    raw = path.read_bytes()
                    trace = json.loads(raw)
                    rows = attribute(trace, graph_labels.NODE_LABELS)
                    trace['st_graph_labels'] = {str(r['graph_node_id']): r['operation'] for r in rows
                                               if r['attribution'] == 'graph_node'}
                    compressed = gzip.compress(json.dumps(trace).encode(), compresslevel=1, mtime=0)
                    gz = path.with_suffix('.json.gz')
                    gz.write_bytes(compressed)
                    path.unlink()
                    for r in rows:
                        self.row(**r, phase=kind, step=index, profiled=True)
                    run['profiles'].append(dict(file=gz.name, bytes=len(compressed),
                        sha256=hashlib.sha256(compressed).hexdigest(), phase=kind, step=index,
                        activities=len(rows), unmapped=sum(r['attribution'] == 'unmapped' for r in rows)))
                except Exception as exc:
                    run['errors'].append('profile finish: ' + str(exc))
            self.file.flush()

    def finish(self, token, device_clock=None):
        global _OBSERVER
        run = self.active
        if run is None or token != run['token']:
            raise ValueError('latency token does not own the active recording')
        after = preparation()
        if device_clock is not None:
            device_clock._drain()                  # query-only; pending samples stay marked pending
            self.row(kind='device_stage_totals', operation='decode', phase='decode',
                     totals_seconds=dict(device_clock.totals), samples=device_clock.samples,
                     pending=bool(device_clock._pending), timing_scope='lifetime counters; not request deltas')
        if run['diagnostic'] and not run['selected']['decode']:
            run['errors'].append('no decode step at requested concurrency was profiled')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        result = dict(schema=SCHEMA, token=token, rank=self.rank, diagnostic=run['diagnostic'],
                      preparation_before=run['before'], preparation_after=after,
                      preparation_changed=any(run['before'][k] != after[k] for k in ('specializations', 'graph_captures')),
                      rows=run['rows'], row_count=run['row_count'], steps=run['kinds'], traces=run['profiles'],
                      concurrency=run['concurrency'],
                      graph_attribution_errors=list(graph_labels.ERRORS), errors=run['errors'],
                      server_directory=str(run['directory']), complete=not run['errors'])
        _write(run['directory'] / 'manifest.json', {k: v for k, v in result.items() if k != 'traces'} |
               {'status': 'complete' if result['complete'] else 'incomplete', 'traces': run['profiles']})
        self.active = None
        _OBSERVER = None
        self.last = result
        return result

    def artifact(self, token, name, offset):
        if self.last is None or self.last['token'] != token or name not in {r['file'] for r in self.last['traces']}:
            raise ValueError('artifact is not part of this latency recording')
        if type(offset) is not int or offset < 0:
            raise ValueError('invalid artifact offset')
        path = Path(self.last['server_directory']) / name
        with path.open('rb') as f:
            f.seek(offset)
            block = f.read(1 << 20)
        return dict(rank=self.rank, token=token, file=name, offset=offset,
                    base64=base64.b64encode(block).decode(), bytes=path.stat().st_size)


def record_step(fn):
    @wraps(fn)
    def measured(runner, step):
        recorder = getattr(runner, 'latency', None)
        if recorder is None or recorder.active is None:
            return fn(runner, step)
        positions = [runner.state.computed.get(s) for s in step.seqs]
        clock = getattr(getattr(runner.model, 'pipeline', None), 'clock', None)
        if clock is not None:
            token = recorder.active['token']
            def sink(name, seconds, context):
                if recorder.active and recorder.active['token'] == token:
                    recorder.row(kind='device_stage', operation=name, duration_us=seconds * 1e6, **context)
            clock.sink = sink
            clock.context = dict(phase=step.kind, step=recorder.step_index + 1, rows=list(step.seqs))
        with recorder.step(step.kind, step.seqs, positions, step.tokens):
            return fn(runner, step)
    return measured
