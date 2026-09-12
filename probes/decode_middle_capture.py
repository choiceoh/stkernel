"""Capture the decode chain's middle and see what it buys. The answer is 1.1x, and why is the point.

The eager stretch between the captured graphs in pipeline.launch -- sampler, verify, commit -- was 180 device
operations before block verification was folded (ledger 45차 §68) and is 23 after (§69). Twenty-three is few
enough to capture, and capturing it was the last piece of "one continuous decode chain". So it was captured.

    docker run --rm --gpus all -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/decode_middle_capture.py

It captures cleanly, generator and Triton kernel included, and the generator advances across replays rather than
freezing -- which is the part that had to be checked. But the replay is only 1.1x the eager stretch, because the
host was never on the critical path: the eager wall (847 us) is the sum of the two kernels' own GPU time (sampler
555, verify 288). The host issues the next launch while the device is still working, so there is nothing for a
graph to hide. What is left is occupancy -- both kernels use one program per row, which at the production shape
of one sequence is six of this box's 48 SMs.
"""
import sys

import torch

sys.path.insert(0, "/w")
from engine.base.sampler import block_verify_batch   # noqa: E402
from engine.profiles.glm53.pipeline import distribution_batch   # noqa: E402

N, T, V, C = 1, 6, 154880, 16
K = T - 1
DEV = torch.device("cuda")


def timed(fn, iters=300, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000)
    out.sort()
    return out[len(out) // 2], out[0]


def main():
    gen = torch.Generator(device=DEV).manual_seed(5)
    g = torch.Generator(device=DEV).manual_seed(3)
    full = torch.randn(N * T, V, generator=g, device=DEV, dtype=torch.bfloat16)
    temps = torch.full((N * T,), 0.7, device=DEV)
    topk = torch.zeros(N * T, dtype=torch.int32, device=DEV)
    topp = torch.full((N * T,), 0.95, device=DEV)
    into = torch.empty(N * T, V, dtype=torch.float32, device=DEV)
    cand = torch.randint(0, V, (N, K, C), generator=g, device=DEV)
    qp = torch.softmax(torch.randn(N, K, C, generator=g, device=DEV), -1)
    drafts = cand[:, :, 0].contiguous()

    def middle():
        probs = distribution_batch(full, temps, topk, topp, None, into).view(N, T, V)
        return block_verify_batch(probs, drafts, cand, qp, gen)

    side, rec = torch.cuda.Stream(), torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        middle()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(gen)
    try:
        torch.cuda.synchronize()
        with torch.cuda.stream(rec):
            graph.capture_begin(torch.cuda.graph_pool_handle(), capture_error_mode="global")
            try:
                out = middle()
            finally:
                graph.capture_end()
    except BaseException:
        graph.reset()
        raise
    print("  captured: sampler + verify + the generator, in one graph")

    seen = set()
    for _ in range(4):
        graph.replay(); torch.cuda.synchronize()
        seen.add(str((out[0].tolist(), out[1].tolist())))
    print(f"  the generator advances across replays rather than freezing: {len(seen) > 1}")

    eager, eager_low = timed(middle)
    replay, replay_low = timed(graph.replay)
    print(f"\n  {'':<20}{'median':>11}{'min':>11}")
    print(f"  {'eager':<20}{eager:>9.1f}us{eager_low:>9.1f}us")
    print(f"  {'one replay':<20}{replay:>9.1f}us{replay_low:>9.1f}us   {eager / replay:.1f}x")
    print(f"\n  {torch.cuda.get_device_properties(0).multi_processor_count} SMs; both kernels run one program a "
          f"row, so at n={N} they use {N * T} of them")


if __name__ == "__main__":
    main()
