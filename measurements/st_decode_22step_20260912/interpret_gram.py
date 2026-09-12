"""Exercise staged statistics/flush semantics without a GPU (not GPU proof)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from engine.kernels.dense.calibration_gram import observe, flush

torch.manual_seed(491)
width = 95
buffer = torch.empty(283, width)
h = torch.zeros(width, width)
count, armed = torch.zeros(()), torch.zeros(())
peaks, cursor = torch.zeros(width), torch.zeros((), dtype=torch.int32)
kept = []
for step in range(37):
    rows = (1, 7, 28)[step % 3]
    x = torch.randn(rows, 128)[:, 3:98]
    mask = torch.arange(rows) % 3 != 0
    observe(x, mask, buffer, h, cursor, armed, count, peaks)
    if step == 0:
        assert count == 0 and cursor == 0
        armed.fill_(1)
    else:
        kept.append(x[mask])
flush(buffer, h, cursor, armed)
expected = torch.cat(kept)
torch.testing.assert_close(h.double(), expected.double().T @ expected.double(), rtol=3e-5, atol=2e-4)
torch.testing.assert_close(peaks, expected.abs().amax(0), rtol=0, atol=0)
assert int(count) == len(expected) and int(cursor) == 0
observe(x, None, None, None, None, armed, count, peaks)
assert int(count) == len(expected) + len(x)
assert not torch.cuda.is_initialized()
print('PASS: masked strided rows, disarmed warmup, threshold flush, peaks-only, no CUDA device')
