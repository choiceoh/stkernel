"""srv2: C=1 vs C=2 decode kernel profile on a held door (:8001), after measurements/st_decode_profile_20260914.

Differences from decode_profile.py: the window plan is an argument (default C=2, C=1, C=2 -- the third profile in one
process has come back empty before, so the two that matter go first), the client waits for the door, and each
window's iteration count is per *step* (drafted / 7 / concurrency) as well as per row-iteration.

    python3 c2_profile.py --out ~/c2opt/profile-main963.json --label main963 --plan 2,1,2 --wait-minutes 40
"""
import argparse
import importlib.util
import json
import sys
import threading
import time

spec = importlib.util.spec_from_file_location("dp", sys.argv[sys.argv.index("--base") + 1])
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)


def wait_door(minutes):
    deadline = time.time() + 60 * minutes
    while time.time() < deadline:
        try:
            if "glm" in json.dumps(dp.http("/v1/models", timeout=5)):
                return True
        except Exception:  # noqa: BLE001 -- the door is not up yet
            pass
        time.sleep(10)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--plan", default="2,1,2")
    ap.add_argument("--wait-minutes", type=int, default=40)
    ap.add_argument("--base", required=True, help="path to decode_profile.py (helpers)")
    a = ap.parse_args()
    if not wait_door(a.wait_minutes):
        print("door never came up", flush=True)
        sys.exit(2)
    print("door up", time.strftime("%T"), flush=True)
    dp.http("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": 16, "chat_template_kwargs": {"thinking": False}}, timeout=300)
    result = {"label": a.label, "plan": a.plan, "runs": []}
    repeat = {}
    for c in (int(x) for x in a.plan.split(",")):
        stop = threading.Event()
        threads = dp.decode(c, stop)
        time.sleep(20)                                   # past prefill, into steady bursts
        table, prof = dp.window(True, 0)
        _, plain = dp.window(False, 8)
        r = repeat.get(c, 0)
        repeat[c] = r + 1
        for d in (prof, plain):
            if d.get("iterations"):
                d["steps"] = d["iterations"] / c
                d["step_ms"] = 1000 * d.get("st:step_seconds_sum", 0.0) / d["steps"]
        run = {"concurrency": c, "repeat": r, "profiled": prof, "unprofiled": plain,
               "device_us_per_step": (table or {}).get("device_us_per_step"),
               "kernels": (table or {}).get("kernels", [])}
        result["runs"].append(run)
        print(f"C={c} #{r}: kernels {len(run['kernels'])}, device {run['device_us_per_step']} us/burst, "
              f"step ms profiled {prof.get('step_ms')} plain {plain.get('step_ms')}, "
              f"burst ms {prof.get('burst_ms')} / {plain.get('burst_ms')}, accept {prof.get('acceptance')}", flush=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
        stop.set()
        for t in threads:
            t.join(timeout=900)
    print("PROFILE done", time.strftime("%T"), flush=True)


if __name__ == "__main__":
    main()
