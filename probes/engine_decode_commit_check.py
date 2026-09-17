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
                        got = advance(picks, state, accepted)
                        for actual, reference in zip(got[:4], expected):
                            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                        assert state['ctx'].item() == 17 + got[0].item()
                        cases += 1
    assert not torch.cuda.is_initialized()
    print(f'PASS Triton CPU interpreter: {cases} greedy/sampled K1/K3/K7 clipping cases; '
          'counts, tokens, done, kept and context agree')


if __name__ == '__main__':
    main()
