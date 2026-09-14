"""M2 layer scheduler: agreed bounded dispatch and fence-owned retirement.

All ranks must call the same methods, including after local preparation
failure. Host control votes precede device work and output collectives.
This does not change the request scheduler's homogeneous Step contract.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

from engine.base.comm import Comm


@dataclass(frozen=True)
class LayerTicket:
    serial: int
    request: str
    slot: int
    generation: int
    layer: int


@dataclass
class _Entry:
    key: LayerTicket
    owner: object
    signature: str
    state: str = 'queued'
    decode: object = None
    prefill: object = None
    decode_ready: object = None
    prefill_ready: object = None
    reader: object = None
    consumer: object = None
    borrowed: bool = False
    owner_phase: object = None


def signature(owner):
    plan, cold = owner.plan, owner.cold
    data = (asdict(plan.identity), plan.decode, plan.prefill, plan.sources,
            plan.experts, plan.tile_m, plan.quota, cold.task_quota,
            cold.task_expert, cold.task_valid_rows, cold.windows)
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


class MixedLayerScheduler:
    def __init__(self, layer, comm, *, capacity=4):
        if type(layer) is not int or not 0 <= layer < 64:
            raise ValueError('mixed scheduler needs a valid layer')
        if (type(capacity) is not int or not 1 <= capacity <= 4
                or not isinstance(comm, Comm) or comm.world_size not in (1, 4)):
            raise ValueError('mixed scheduler supports one-rank components or TP4 and at most four slots')
        self.layer, self.comm, self.capacity = layer, comm, capacity
        self._entries, self._latest = {}, {}
        self._serial = 0
        self._stream = None

    def _vote(self, command, *, error=None, ready=True):
        reply = dict(command=command, error=None if error is None else str(error)[:240], ready=bool(ready))
        replies = self.comm.gather_objects(reply)
        if len(replies) != self.comm.world_size:
            raise RuntimeError('mixed control vote lost a rank')
        errors = [r['error'] for r in replies if r['error'] is not None]
        if errors:
            raise RuntimeError('mixed rank refused work: ' + '; '.join(errors))
        if any(r['command'] != command for r in replies):
            raise RuntimeError('mixed ranks disagreed on ticket, descriptor or collective order')
        return all(r['ready'] for r in replies)

    def admit(self, owner, *, request, slot, preparation_error=None):
        """Every rank votes even when its local prepare returned an error."""
        key, digest, error = None, None, preparation_error
        try:
            if error is not None:
                raise RuntimeError(str(error))
            if not isinstance(request, str) or not 1 <= len(request) <= 128:
                raise ValueError('request identity must be a bounded nonempty string')
            if type(slot) is not int or not 0 <= slot < self.capacity or slot in self._entries:
                raise ValueError('mixed slot is occupied or outside capacity')
            if any(e.owner is owner for e in self._entries.values()):
                raise ValueError('mixed owner is already leased by another ticket')
            if self._stream is not None and owner.stream != self._stream:
                raise ValueError('mixed tickets must share one eager dispatch stream')
            identity = owner.plan.identity
            if identity.layer != self.layer or owner.state not in ('new', 'complete'):
                raise ValueError('mixed owner is not a ready invocation for this layer')
            owner.validate(identity)
            prior = self._latest.get(slot)
            if prior is not None and (identity.slot_generation < prior.generation or
                    (identity.slot_generation == prior.generation and request != prior.request)):
                raise ValueError('slot reuse requires a new request generation')
            key = LayerTicket(self._serial, request, slot, identity.slot_generation, self.layer)
            digest = signature(owner)
            phase = (owner.state, owner.next_window)
            stream = owner.stream
        except Exception as exc:
            error = exc
        self._vote(('admit', self.layer, self._serial, request, slot), error=error)
        self._vote(('descriptor', key, digest, phase))
        self._entries[slot] = _Entry(key, owner, digest, owner_phase=phase)
        self._latest[slot] = key
        self._stream = stream
        self._serial += 1
        return key

    def _guard(self, action, key, states, *, validate=True):
        entry, phase, error = None, None, None
        try:
            entry = self._entries[key.slot]
            if entry.key != key or entry.state not in states:
                raise ValueError('stale ticket or invalid mixed completion phase')
            if validate:
                entry.owner.validate(entry.owner.plan.identity)
                current = (entry.owner.state, entry.owner.next_window)
                if entry.owner_phase != current:
                    raise ValueError('mixed owner advanced outside its ticket')
                phase = (entry.state, current)
            else:
                # A failed launch can leave different local owner cursors.
                # Cancellation must still let every rank drain its readers.
                phase = entry.state
        except Exception as exc:
            error = exc
        self._vote((action, key, phase), error=error)
        return entry

    def _run(self, action, key, states, fn, state, *, reduce=False):
        entry = self._guard(action, key, states)
        entry.state = 'dispatching'
        value, phase, next_state, reduction, error = None, None, None, None, None
        try:
            value = fn(entry.owner)
            next_state = state(entry.owner) if callable(state) else state
            phase = (entry.owner.state, entry.owner.next_window)
            if phase[0] != next_state:
                raise RuntimeError('mixed owner returned an unexpected completion phase')
            if reduce:
                # Do local refusal checks BEFORE peers enter a device
                # collective. M2 uses one fixed process-group sum, not the
                # ordinary per-tensor one-shot/NCCL reader selection.
                self.comm._check_packets()
                if not value.is_contiguous() or not value.numel():
                    raise ValueError('mixed output collective requires a nonempty contiguous tensor')
                reduction = (tuple(value.shape), str(value.dtype), value.device.type)
        except Exception as exc:
            error = exc
        try:
            self._vote((action+'-submitted', key, phase, reduction), error=error)
            # Every surviving rank enters the publication vote even if its
            # local fence recording fails. Device/process failure recovery is
            # the communicator owner's responsibility, not a ticket retry.
            error = None
            try:
                if reduce and self.comm.world_size != 1:
                    import torch.distributed as dist
                    dist.all_reduce(value, group=self.comm.group)
                fence = entry.owner.reader_fence()
            except Exception as exc:
                error = exc
            self._vote((action+'-published', key), error=error)
        except Exception:
            entry.state = 'failed'  # retain sources; cancel/reap must drain them
            raise
        entry.reader = fence
        entry.state, entry.owner_phase = next_state, phase
        return entry, value, fence

    def begin(self, key):
        entry, value, event = self._run('begin', key, ('queued',),
            lambda o: o.begin(o.plan.identity), 'decode', reduce=True)
        entry.decode, entry.decode_ready = value, event

    def advance(self, key):
        entry, _, _ = self._run('advance', key, ('decode', 'cold'),
            lambda o: o.advance(o.plan.identity), lambda o: 'routed' if o.state == 'routed' else 'cold')
        return entry.state == 'routed'

    def finish(self, key):
        # Never call the owner's draining finish while cold windows remain:
        # each scheduler advance must remain one bounded work quantum.
        entry, value, event = self._run('finish', key, ('routed',),
            lambda o: o.finish(o.plan.identity), 'complete', reduce=True)
        entry.prefill, entry.prefill_ready = value, event

    def result(self, key, *, prefill):
        entry = self._guard('prefill-result' if prefill else 'decode-result', key,
            ('complete',) if prefill else ('decode', 'cold', 'routed', 'complete'))
        entry.borrowed = True
        return (entry.prefill, entry.prefill_ready) if prefill else (entry.decode, entry.decode_ready)

    def _retire(self, key, consumer_fence, *, cancel):
        entry = self._guard('cancel' if cancel else 'release', key,
            ('queued', 'decode', 'cold', 'routed', 'complete', 'failed') if cancel else ('complete',), validate=False)
        error = None
        try:
            if entry.borrowed and consumer_fence is None:
                raise ValueError('borrowed outputs need a fence after their last consumer')
            if consumer_fence is not None and not callable(getattr(consumer_fence, 'query', None)):
                raise ValueError('consumer fence must report completion')
            reader = entry.owner.reader_fence()
        except Exception as exc:
            error = exc
        self._vote(('retire-fence', key, cancel), error=error)
        entry.reader, entry.consumer, entry.state = reader, consumer_fence, 'retiring'

    def cancel(self, key, *, consumer_fence=None):
        """Stop future dispatch; retain all already-submitted readers."""
        self._retire(key, consumer_fence, cancel=True)

    def release(self, key, *, consumer_fence=None):
        self._retire(key, consumer_fence, cancel=False)

    def reap(self, key):
        """Reuse is legal only when every rank's readers and consumers ended."""
        entry = self._guard('reap', key, ('retiring',), validate=False)
        ready, error = False, None
        try:
            ready = entry.reader.query() and (entry.consumer is None or entry.consumer.query())
        except Exception as exc:
            error = exc
        if not self._vote(('retired', key), error=error, ready=ready):
            return False
        del self._entries[key.slot]
        return True
