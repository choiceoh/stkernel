"""CPU measurement of the fleet boot's native builds: one after another (as first uses did) or all at once (natives).

Runs the real build entry points of profiles/glm53/natives in the ST image with CUDA hidden, against build roots
under /work (never the fleet's /cache). `serial` builds in the order a cold boot first used them; `parallel` runs
natives.NativeBuilds. Each sample is a fresh process; a root that already holds the keys measures the kept case.

usage: native_builds_cpu.py serial|parallel
"""
import json
import os
import resource
import sys
import time


def peak_bytes():
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return int(open(path).read().strip())
        except (OSError, ValueError):
            continue
    return None


def main():
    mode = sys.argv[1]
    start = time.perf_counter()
    from engine.profiles.glm53 import natives
    builds = natives.builds(2, True)                      # the served one-shot: two rails, inline flags
    imported = time.perf_counter() - start
    start = time.perf_counter()
    if mode == "parallel":
        seconds = natives.NativeBuilds(builds).wait()
    else:
        seconds = {}
        order = ("one-shot", "dense", "prefill-topk", "mla", "bounded-graph", "mapped-staging", "decode-queue")
        table = dict(builds)
        for name in order:
            t = time.perf_counter()
            table[name]()
            seconds[name] = round(time.perf_counter() - t, 3)
    wall = time.perf_counter() - start
    print(json.dumps(dict(mode=mode, wall_s=round(wall, 3), import_s=round(imported, 3), seconds=seconds,
                          children_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                          cgroup_peak_bytes=peak_bytes(), cpus=os.cpu_count(),
                          roots={k: v for k, v in os.environ.items() if k.endswith("_BUILD_ROOT")})))


if __name__ == "__main__":
    main()
