"""Judge ST's served KDA state orientation and every verify-token snapshot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.profiles.glm53 import lanes


def relative(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def main():
    torch.manual_seed(29)
    ref, served = lanes.reference(), lanes.served(reference_for=("expert",))
    H, D = 16, 128
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    rows = []
    for T in (1, 6, 64):
        for seeded in (False, True):
            q, k, v, raw = (rand(1, T, H, D) for _ in range(4))
            beta = rand(1, T, H)
            A, bias = rand(H).float() * .2, rand(H * D).float() * .1
            initial = torch.randn(1, H, D, D, device="cuda") * .1 if seeded else None
            saved = initial.clone() if seeded else None
            args = (q, k, v, raw, beta, A, bias, initial, -5.)
            expected, states = ref.kda_recurrent(*args)
            out, actual = served.kda_recurrent(*args)
            chunk, last = served.kda_chunk(*args)
            result = {"tokens": T, "seeded": seeded, "recurrent_output": relative(out, expected),
                      "every_state": relative(actual, states), "chunk_output": relative(chunk, expected),
                      "chunk_final_state": relative(last, states[-1:])}
            assert all(v < .02 for k, v in result.items() if k not in ("tokens", "seeded")), result
            if seeded:
                assert torch.equal(saved, initial), "the lane overwrote the rollback starting state"
            rows.append(result)
    print(json.dumps({"passed": True, "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
