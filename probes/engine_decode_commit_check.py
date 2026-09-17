"""Check the fused commit body with Triton's CPU interpreter, without a GPU.

CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python -m probes.engine_decode_commit_check
This checks kernel arithmetic, not native CUDA compilation or graph replay.
"""
import os


def main():
    if os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('use CUDA_VISIBLE_DEVICES= and TRITON_INTERPRET=1')
    import torch
    from engine.kernels.common.decode_commit import advance
    from engine.base.sampler import commit_batch

    cases = 0
    for k in (1, 3, 7):
        for sampled in (False, True):
            for kept in range(k + 1):
                for room in range(k + 2):
                    for stop_at in (0, kept, k + 1):
                        drafts = torch.arange(10, 10 + k).view(1, k)
                        picks = torch.cat((drafts[:, :kept], torch.tensor([[50]]),
                                           torch.full((1, k - kept), 60)), 1)
                        end = int(picks[0, stop_at]) if stop_at < k + 1 else 99
                        state = dict(drafts=drafts, ends=torch.tensor([[end]]), alive=torch.tensor([True]),
                                     generated=torch.tensor([0]), limit=torch.tensor([room]),
                                     ctx=torch.tensor([17]), anchor=torch.tensor([9]),
                                     real_slot=torch.tensor([1]), slot=torch.tensor([1]))
                        accepted = torch.tensor([kept]) if sampled else None
                        expected = commit_batch(picks, drafts, state['alive'], state['generated'],
                                                state['limit'], state['ends'], accepted)
                        ctx0 = int(state['ctx'].item())
                        gen0 = int(state['generated'].item())
                        alive0 = bool(state['alive'].item())
                        anchor0 = int(state['anchor'].item())
                        slot0 = int(state['real_slot'].item())
                        got = advance(picks, state, accepted)
                        # count/done/kept against the shared commit reference.
                        for actual, reference in zip(got[:3], expected[:3]):
                            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                        count, done = int(got[0].item()), bool(got[1].item())
                        # `picks` is the same object in got and expected, so the
                        # old picks comparison was a no-op and no state store was
                        # read. Check the readback and every store the kernel makes.
                        assert int(got[4].item()) == ctx0, 'readback context'
                        assert int(state['ctx'].item()) == ctx0 + count, 'context advance'
                        assert int(state['generated'].item()) == gen0 + count, 'generated advance'
                        assert bool(state['alive'].item()) == (alive0 and not done), 'alive clear'
                        assert int(state['slot'].item()) == (slot0 if (alive0 and not done) else 0), 'slot release'
                        want_anchor = int(picks[0, max(count - 1, 0)].item()) if count > 0 else anchor0
                        assert int(state['anchor'].item()) == want_anchor, 'anchor follow'
                        cases += 1
    assert not torch.cuda.is_initialized()
    print(f'PASS Triton CPU interpreter: {cases} greedy/sampled K1/K3/K7 clipping cases; '
          'counts, tokens, done, kept, context, generated, alive, slot and anchor agree')


if __name__ == '__main__':
    main()
