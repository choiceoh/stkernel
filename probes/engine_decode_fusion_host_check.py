"""Execute the decode integer kernels with Triton's CPU interpreter.

This checks the actual Triton bodies against independent PyTorch operations;
it does not execute CUDA or establish device timing/graph safety.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['TRITON_INTERPRET'] = '1'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from engine.base.sampler import commit_batch
    from engine.kernels.decode_commit import advance
    from engine.kernels.vocab_candidates import argmax_key
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(914)
    cases = 0
    for n in (1, 3, 8):
        for k in (0, 1, 5, 8):
            for e in (1, 3, 7):
                for sampled in (False, True):
                    for trial in range(8):
                        picks = torch.randint(0, 32, (n, 2*(k+1)), generator=gen)[:, ::2].contiguous()
                        drafts = picks[:, :k].clone()
                        if k:
                            drafts[::2, trial % k] = (drafts[::2, trial % k] + 1) % 32
                        accepted = torch.randint(0, k+1, (n,), generator=gen) if sampled else None
                        state = dict(drafts=drafts, ends=torch.randint(-1, 32, (n, e), generator=gen),
                                     alive=torch.randint(0, 2, (n,), generator=gen).bool(),
                                     generated=torch.randint(0, 16, (n,), generator=gen),
                                     limit=torch.randint(0, 16, (n,), generator=gen),
                                     ctx=torch.randint(0, 4096, (n,), generator=gen),
                                     anchor=torch.randint(0, 32, (n,), generator=gen),
                                     real_slot=torch.arange(1, n+1), slot=torch.arange(1, n+1))
                        if trial == 0:
                            state['ends'][:, 0] = picks[:, 0]       # EOS on the first token
                        if trial == 1:
                            state['limit'] = state['generated']    # no remaining room
                        count, done, kept, _ = commit_batch(picks, drafts, state['alive'], state['generated'],
                                                           state['limit'], state['ends'], accepted)
                        before = {name: value.clone() for name, value in state.items()}
                        actual = advance(picks, state, accepted)
                        for got, want in zip(actual, (count, done, kept, picks, before['ctx'])):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
                        expected = dict(ctx=before['ctx']+count, generated=before['generated']+count,
                                        alive=before['alive'] & ~done,
                                        slot=torch.where(before['alive'] & ~done, before['real_slot'], 0),
                                        anchor=torch.where(count > 0,
                                            picks.gather(1, (count-1).clamp_min(0)[:, None]).squeeze(1), before['anchor']))
                        for name, want in expected.items():
                            torch.testing.assert_close(state[name], want, rtol=0, atol=0)
                        cases += 1
    vocab_cases = 0
    for dtype in (torch.float32, torch.float16):
        for width in (1, 31, 1024, 1025, 38720):
            logits = torch.randn(7, width*2, generator=gen).to(dtype)[:, ::2]
            logits[0].fill_(float('-inf'))
            logits[1].fill_(-0.); logits[1, -1] = 0.
            logits[2, -1] = float('nan')
            logits[3, 0] = float('-nan')
            logits[4, -1] = float('inf')
            logits[5].fill_(3.)
            for valid in sorted({1, width, max(1, width//2)}):
                for start in (0, 3*38720):
                    key = argmax_key(logits, start, valid)
                    ids = 0xffffffff - (key & 0xffffffff)
                    torch.testing.assert_close(ids, logits[:, :valid].argmax(-1)+start, rtol=0, atol=0)
                    value, index = logits[:, :valid].float().max(-1)
                    value = torch.where(value == 0, 0., value)
                    value = torch.where(torch.isnan(value), float('nan'), value)
                    bits = value.contiguous().view(torch.int32).long()
                    ordered = torch.where(bits < 0, bits ^ 0x7fffffff, bits)
                    reference = (ordered << 32) | (0xffffffff - (index+start))
                    torch.testing.assert_close(key, reference, rtol=0, atol=0)
                    vocab_cases += 1
            torch.testing.assert_close(argmax_key(logits, 0, 0), torch.full((7,), -(2**63)), rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
    paths = ('engine/kernels/decode_commit.py', 'engine/kernels/vocab_candidates.py')
    report = dict(status='PASS', evidence='Triton CPU interpreter, not GPU execution',
                  commit_cases=cases, vocab_cases=vocab_cases, cuda_initialized=False,
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
