"""Private same-boot control: upstream top-k over every complete kpool.

Transformers glm5_next blob 8efbef9839129b5db0164653c1d9e978ab215666
and vLLM sparse_indexer blob 1fc42c9e4114bf9ff997dcf49cd88015127b5b3e
select complete pools by score, then append only the incomplete tail. ST's
extra pin changes that selection whenever the token count is divisible by 4.
This diagnostic changes only that pin, on an exclusive eager target request.
"""
from contextlib import contextmanager


@contextmanager
def unpinned_complete_pools(net):
    from engine.profiles.glm53 import net as module

    previous = module.pin_pools_in_logits
    calls = 0
    rows = 0

    def keep_scores(logits, pin, *, k=None):
        nonlocal calls, rows
        calls += 1
        rows += logits.shape[0]

    module.pin_pools_in_logits = keep_scores
    try:
        yield
    finally:
        module.pin_pools_in_logits = previous
        identity = getattr(net, 'incident_step_identity', {})
        if identity.get('prefill') or identity.get('generation') == 1:
            print(f'[incident-tail-reference] rank={net.rank} '
                  f'prefill={identity.get("prefill")} '
                  f'generation={identity.get("generation")} '
                  f'unpinned_calls={calls} rows={rows}', flush=True)
