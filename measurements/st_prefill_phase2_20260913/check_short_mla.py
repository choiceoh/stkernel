"""Short MLA numerical, changed-input and stream gate; requires owned GPU lease."""
import json
import time
from unittest.mock import patch
import torch
from engine.kernels import mla


def elapsed(fn):
    for _ in range(2): fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / 5


def main():
    started = time.monotonic()
    mla.configure_prefill('tile32')
    mla._build()
    torch.manual_seed(213)
    cache = (torch.randn(8192, 512, device='cuda') * .4).to(torch.float8_e4m3fn)
    report = []
    for rows, width in ((128,33), (129,2176), (289,33), (1024,2048),
                        (2121,2176), (2128,2176), (2304,2048), (4095,2176)):
        q = torch.randn(rows,16,512,device='cuda',dtype=torch.bfloat16)
        slots = torch.randint(8192,(rows,width),device='cuda',dtype=torch.int32)
        lens = (torch.arange(rows,device='cuda',dtype=torch.int32)+1).clamp(max=width)
        lens[0] = 0; slots[0] = -1
        lens[1] = min(width,7); slots[1] = 8191
        slots[2] = 0
        def candidate(): return mla.mla_decode(q,cache,slots,lens,512**-.5,.7)
        def previous():
            with patch.object(mla,'ENABLE_MLA_PREFILL32',False):
                return mla.mla_decode(q,cache,slots,lens,512**-.5,.7)
        for changed in (False,True):
            if changed:
                q.mul_(-.5); lens[-1] = 7; slots[-1] = 8191
            saved = (q.clone(),slots.clone(),lens.clone())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = candidate()
            torch.cuda.current_stream().wait_stream(stream)
            control = previous()
            ix = torch.tensor([0,1,2,rows//2,rows-1],device='cuda')
            oracle = mla.mla_decode_ref(q[ix],cache,slots[ix],lens[ix],512**-.5,.7)
            denominator = oracle.float().flatten(1).norm(dim=1).clamp_min(1e-6)
            error = ((actual[ix].float()-oracle.float()).flatten(1).norm(dim=1)/denominator).max().item()
            old_error = ((control[ix].float()-oracle.float()).flatten(1).norm(dim=1)/denominator).max().item()
            assert error <= .02 and old_error <= .02, (rows,width,error,old_error)
            assert torch.isfinite(actual).all() and not actual[0].count_nonzero()
            for a,b in zip((q,slots,lens),saved): torch.testing.assert_close(a,b,atol=0,rtol=0)
            row = dict(rows=rows,width=width,changed_input=changed,reference_error=error,
                       control_error=old_error,nondefault_stream=True,input_unchanged=True)
            if changed and rows in (2121,2128):
                row.update(candidate_ms=elapsed(candidate),control_ms=elapsed(previous))
            report.append(row); print(json.dumps(row),flush=True)
    print(json.dumps(dict(passed=True,seconds=time.monotonic()-started,cases=report,
                          scope='MLA kernel only; consumer TTFT and quality pending')),flush=True)


if __name__ == '__main__': main()
