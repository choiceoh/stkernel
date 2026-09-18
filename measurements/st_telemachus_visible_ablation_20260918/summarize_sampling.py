"""Recheck the incident's sampling claims without emitting private output text."""
import argparse
import json
from pathlib import Path
import statistics

import torch


def summarize(directory, admission=1):
    paths = sorted(directory.glob(f'admit{admission}-gen*.pt'),
                   key=lambda path: int(path.stem.split('-gen')[1]))
    assert paths, 'no incident captures'
    rows, selected = [], []
    non_argmax = 0
    expected_non_argmax = 0.0
    reasoning_end = None
    for generation, path in enumerate(paths):
        data = torch.load(path, map_location='cpu', weights_only=True)
        assert data['generation'] == generation and data['admission'] == admission
        probabilities = data['probabilities'][0].double()
        probabilities /= probabilities.sum()
        picked = data['committed'][0]
        argmax = int(probabilities.argmax())
        selected.append(float(probabilities[picked]))
        non_argmax += picked != argmax
        expected_non_argmax += 1.0 - float(probabilities[argmax])
        if picked == 154842:  # verified GLM-5.3 tokenizer's </think>
            assert reasoning_end is None
            reasoning_end = generation
        if generation in (0, 2, 3):
            cdf = probabilities.cumsum(0)
            lower = float(cdf[picked - 1]) if picked else 0.0
            upper = float(cdf[picked])
            uniform = data['uniform']
            rows.append(dict(generation=generation, sampled_id=picked, argmax_id=argmax,
                             sampled_probability=float(probabilities[picked]),
                             argmax_probability=float(probabilities[argmax]),
                             uniform=uniform, cdf_lower=lower, cdf_upper=upper,
                             cdf_nearest_margin=min(uniform - lower, upper - uniform),
                             position_in_selected_interval=(uniform - lower) / (upper - lower)))
    assert reasoning_end is not None
    return dict(capture_count=len(paths), admission=admission,
                reasoning_end_generation=reasoning_end, non_argmax=non_argmax,
                expected_non_argmax_mass=expected_non_argmax,
                median_sampled_probability_gen4_to_end=statistics.median(selected[4:]),
                median_sampled_probability_visible=statistics.median(selected[reasoning_end + 1:]),
                selected_positions=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('--admission', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(summarize(args.captures, args.admission), indent=2))
