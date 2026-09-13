"""Bounded CUDA WHILE with the served commit kernel; no synthetic speed verdict."""
import gc
import unittest
import weakref
import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires admitted GB10")
class BoundedLoopCudaTests(unittest.TestCase):
    def test_changing_inputs_limits_and_every_exit_preserve_each_commit(self):
        from engine.kernels.bounded_graph import BoundedGraph
        from engine.kernels.common.decode_commit import advance
        from engine.profiles.glm53.bounded_loop import stop_at_boundary
        for rows in (1, 4):
            for limit in (1, 2, 4):
                counter = torch.zeros(1, dtype=torch.int64, device="cuda")
                stop = torch.zeros_like(counter)
                program = torch.empty(4, rows, 7, dtype=torch.int64, device="cuda")
                drafts = torch.empty(4, rows, 6, dtype=torch.int64, device="cuda")
                interrupts = torch.zeros(4, 1, dtype=torch.int64, device="cuda")
                reserved = torch.full((rows,), 32768, dtype=torch.int64, device="cuda")
                state = {key: torch.zeros(rows, dtype=torch.int64, device="cuda") for key in
                         ("generated", "limit", "ctx", "anchor", "real_slot", "slot")}
                state.update(drafts=torch.zeros(rows, 6, dtype=torch.int64, device="cuda"),
                             ends=torch.full((rows, 1), -7, dtype=torch.int64, device="cuda"),
                             alive=torch.ones(rows, dtype=torch.bool, device="cuda"))
                logs = {key: torch.empty(4, rows, dtype=torch.int64, device="cuda")
                        for key in ("count", "kept", "before", "ctx")}
                logs["tokens"] = torch.empty_like(program)
                logs["done"] = torch.empty(4, rows, dtype=torch.bool, device="cuda")

                def reset(case, trial):
                    values = torch.arange(4 * rows * 7, device="cuda").view(4, rows, 7) + 100 + 500 * trial
                    program.copy_(values)
                    if case == "eos":
                        program[1, 0, 2] = -7
                    drafts.copy_(program[:, :, :6])
                    if rows == 4:
                        drafts[:, 1, 2].add_(1)  # shorter accepted prefix on another row
                    interrupts.zero_()
                    if case == "interrupt":
                        interrupts[1, 0] = 1
                    reserved.fill_(32768)
                    state["alive"].fill_(True)
                    state["generated"].zero_()
                    state["limit"].fill_(100)
                    state["ctx"].fill_(100 + trial)
                    state["anchor"].zero_()
                    state["real_slot"].copy_(torch.arange(1, rows + 1, device="cuda"))
                    state["slot"].copy_(state["real_slot"])
                    state["drafts"].zero_()
                    if case == "prefix":
                        state["ctx"][0] = 2304 - 14
                    elif case == "reservation":
                        reserved[0] = state["ctx"][0] + 14
                    elif case == "bucket":
                        state["ctx"].fill_(32768 - 14)
                    elif case == "generation_limit":
                        state["limit"][0] = 10
                    counter.zero_()
                    stop.zero_()
                    for key, log in logs.items():
                        log.fill_(False if key == "done" else -1)

                def body():
                    picks = program.index_select(0, counter).squeeze(0)
                    state["drafts"].copy_(drafts.index_select(0, counter).squeeze(0))
                    count, done, kept, tokens, before = advance(picks, state)
                    values = dict(count=count, done=done, kept=kept, tokens=tokens,
                                  before=before, ctx=state["ctx"])
                    for key, value in values.items():
                        logs[key].index_copy_(0, counter, value.unsqueeze(0))
                    vote = stop_at_boundary(before, state["ctx"], state["alive"], reserved, 32768,
                                            interrupts.index_select(0, counter).reshape(1),
                                            step_tokens=7, block=2304)
                    # This single-GPU numerical gate has no TP4 transport.
                    # The serving protocol must agree_stop() before this copy.
                    stop.copy_(vote)

                reset("steady", 0)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    body()
                torch.cuda.current_stream().wait_stream(side)
                reset("steady", 0)
                captured = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(captured):
                    body()
                loop = BoundedGraph(captured, counter, stop, limit,
                                    owners=(state, logs, program, drafts, interrupts, reserved))
                try:
                    for case in ("steady", "eos", "prefix", "reservation", "bucket", "generation_limit", "interrupt"):
                        for trial in range(3):
                            with self.subTest(rows=rows, limit=limit, case=case, trial=trial):
                                reset(case, trial)
                                reference = {key: value.clone() for key, value in state.items()}
                                expected = {key: value.clone() for key, value in logs.items()}
                                completed = 0
                                for iteration in range(limit):
                                    reference["drafts"].copy_(drafts[iteration])
                                    count, done, kept, tokens, before = advance(program[iteration], reference)
                                    values = dict(count=count, done=done, kept=kept, tokens=tokens,
                                                  before=before, ctx=reference["ctx"])
                                    for key, value in values.items():
                                        expected[key][iteration].copy_(value)
                                    completed += 1
                                    vote = stop_at_boundary(before, reference["ctx"], reference["alive"],
                                                            reserved, 32768, interrupts[iteration],
                                                            step_tokens=7, block=2304)
                                    if vote.item():
                                        break
                                loop.replay()
                                self.assertEqual(counter.item(), completed)
                                for key in state:
                                    torch.testing.assert_close(state[key], reference[key], rtol=0, atol=0)
                                for key in logs:
                                    torch.testing.assert_close(logs[key], expected[key], rtol=0, atol=0)
                    with torch.cuda.stream(side), self.assertRaisesRegex(RuntimeError, "construction stream"):
                        loop.replay()
                finally:
                    loop.close()
                    captured.reset()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    loop.replay()

    def test_native_owner_survives_wrapper_reference_release(self):
        from engine.kernels.bounded_graph import BoundedGraph
        class Owner:
            pass
        owner = Owner()
        owner.value = torch.zeros(1, device="cuda")
        counter = torch.zeros(1, dtype=torch.int64, device="cuda")
        stop = torch.zeros_like(counter)
        captured = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(captured):
            owner.value.add_(1)
            stop.zero_()
        loop = BoundedGraph(captured, counter, stop, 4, owners=(owner,))
        ref = weakref.ref(owner)
        del owner
        loop.owners, loop.body = (), None  # native owner must cover Python teardown order
        gc.collect()
        self.assertIsNotNone(ref())
        loop.replay()
        loop.close()
        self.assertEqual(counter.item(), 4)
        gc.collect()
        self.assertIsNone(ref())
        captured.reset()


if __name__ == "__main__":
    unittest.main()
