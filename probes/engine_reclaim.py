"""Prepare immediately free UMA pages before restarting the pinned ST release.

Run only with the ST model stopped. Preserves at least 16 GiB MemAvailable
during the transient allocation, then releases it before model admission.
"""
import json
from engine.base.arena import _meminfo, touch_pages, GIB

before = _meminfo()
touched = 0
if before['MemFree'] < 80*GIB:
    amount = 78*GIB
    if amount > before['MemAvailable']-16*GIB:
        raise MemoryError('insufficient headroom for bounded ST cache reclaim')
    touched = touch_pages(amount)
after = _meminfo()
print(json.dumps(dict(before_free=before['MemFree'], after_free=after['MemFree'],
                      available=after['MemAvailable'], touched=touched)), flush=True)
