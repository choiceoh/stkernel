"""What a parked record costs to hold, to read back, and to summarize -- and what the loop's new work costs.

From the repo root:  python3 measurements/parked_record_bound_20260919/record_cost.py
"""
import json
import random
import sys
import time
import tracemalloc

sys.path.insert(0, ".")
from engine.base.runner import Runner, history_head  # noqa: E402
from engine.base.serve import Server  # noqa: E402


def best_ms(fn, reps=20):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def glm_record(n, seed=0):
    """The shape engine/profiles/glm53/adapter.py `park` returns; fresh int objects, as a sampler's .tolist() makes."""
    rng = random.Random(seed)
    tokens = [int(str(rng.randrange(0, 151_000))) for _ in range(n)]
    return {"context": n - 1, "pending": 1, "tokens": tokens, "prompt_len": max(n - 200, 1),
            "limits": [4096, 1.0], "min_new": 0, "options": {"top_p": 0.95}, "media": []}


print(f"python {sys.version.split()[0]}")
for n in (233, 43_318, 100_000):
    tracemalloc.start()
    record = glm_record(n)
    held, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    text = json.dumps(record, separators=(",", ":"))
    ids = record["tokens"] + [7]
    digest = Runner._history(Runner._summary(record))
    offer = (n, False)
    assert Server._offer(ids, [], record["tokens"], [], set()) == offer
    assert Server._offer_digest(ids, [], digest, set()) == offer
    print(f"{n:>7,} tokens: held {held / 2**20:7.3f} MiB ({held / n:4.1f} B/token), JSON {len(text) / 2**20:6.3f} MiB, "
          f"json.loads {best_ms(lambda: json.loads(text)):7.3f} ms | "
          f"history_head {best_ms(lambda: history_head(record['tokens'])):6.3f} ms, "
          f"_summary (park) {best_ms(lambda: Runner._summary(record)):6.3f} ms, "
          f"_offer_digest {best_ms(lambda: Server._offer_digest(ids, [], digest, set())):6.3f} ms "
          f"vs _offer {best_ms(lambda: Server._offer(ids, [], record['tokens'], [], set())):6.3f} ms (hint re-check)")
