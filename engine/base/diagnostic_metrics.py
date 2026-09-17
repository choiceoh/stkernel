"""Bounded host-side serving diagnostics; never synchronizes a device.

Completion cohorts use actual rows and context lengths. Async residency and
device-burst times are distinct labels, since they are not interchangeable.
Response lengths count completed chat choices, not raw tokenizer guesses.
"""
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import threading


def context_band(n):
    return next((str(limit) for limit in (8192, 32768, 131072) if n <= limit), 'over128k')


def build_identity():
    explicit = os.environ.get('ST_BUILD_SHA')
    if explicit:
        return explicit[:64]
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                                       stderr=subprocess.DEVNULL, timeout=1, text=True).strip()[:64]
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


class DiagnosticMetrics:
    LENGTH_BOUNDS = (0, 128, 512, 2048, 8192, 32768, 131072, math.inf)

    def __init__(self, server):
        self.server = server
        self.build = build_identity()
        self.values = collections.Counter()
        self.lock = threading.Lock()  # HTTP completions and runner updates; no callbacks under it.

    def begin(self, seqs):
        model = self.server.engine
        if not hasattr(model, 'generated_count'):
            return None
        state = getattr(getattr(self.server, "runner", None), "state", None)
        running = getattr(state, "running", self.server._active)
        rows = [s for s in seqs if s in self.server._active and s in running]
        if not rows or len(rows) != len(seqs):
            # A graph with already-finished ghost rows is not a clean C=n cohort.
            return None
        contexts = [model.context(s) for s in rows]
        cache = [self.server._cached.get(self.server._active[s][0]) for s in rows]
        label = ('unknown' if any(v is None for v in cache) else
                 'cold' if all(v == 0 for v in cache) else
                 'hit' if all(v > 0 for v in cache) else 'mixed')
        return (rows, max(contexts), label, sum(model.generated_count(s) for s in rows),
                getattr(model, 'drafted_total', 0), getattr(model, 'accepted_total', 0))

    def end(self, before, seconds, iterations=1, timing='sync_wall'):
        if before is None or seconds <= 0 or iterations <= 0:
            return
        rows, context, cache, generated, drafted, accepted = before
        model = self.server.engine
        emitted = sum(model.generated_count(s) for s in rows)-generated
        labels = (('cache', cache), ('context', context_band(context)),
                  ('sequences', str(len(rows)) if len(rows) <= 4 else 'over4'), ('timing', timing))
        with self.lock:
            for name, value in (
                    ('steps', iterations), ('seconds', seconds), ('tokens', max(0, emitted)),
                    ('drafted', max(0, getattr(model, 'drafted_total', 0)-drafted)),
                    ('accepted', max(0, getattr(model, 'accepted_total', 0)-accepted))):
                self.values[(f'st:condition_{name}_total', labels)] += value

    def prefill(self, tokens, seconds):
        with self.lock:
            self.values[('st:prefill_computed_tokens_total', ())] += tokens
            self.values[('st:prefill_compute_seconds_total', ())] += seconds

    def response(self, choices):
        with self.lock:
            for c in choices:
                self.values[('st:response_finished_total', (('reason', c.finish_reason()),))] += 1
                for part, count in (('reasoning', len(c.streams['reasoning_content'].ids)),
                                    ('answer', len(c.streams['content'].ids))):
                    labels = (('part', part),)
                    self.values[('st:response_tokens_sum', labels)] += count
                    self.values[('st:response_tokens_count', labels)] += 1
                    for bound in self.LENGTH_BOUNDS:
                        le = '+Inf' if math.isinf(bound) else str(bound)
                        self.values[('st:response_tokens_bucket', (('le', le), *labels))] += int(count <= bound)

    def render(self):
        server = self.server
        draft = getattr(server.engine, 'drafter', None)
        info = dict(build=self.build, boot=server.latency_boot_id,
                    k=str(getattr(draft, 'k', 'unknown')), precision=str(getattr(draft, 'decode_precision', 'unknown')))
        with self.lock:
            values = list(self.values.items())
        out = ['st:runtime_info{' + ','.join(k+'='+json.dumps(v) for k,v in info.items()) + '} 1\n']
        for (name, labels), value in sorted(values):
            out.append(name+'{'+','.join(k+'='+json.dumps(v) for k,v in labels)+'} '+str(value)+'\n')
        return ''.join(out)
