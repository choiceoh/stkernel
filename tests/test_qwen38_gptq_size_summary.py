import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock


path = Path(__file__).resolve().parents[1] / 'measurements/qwen38_gptq_20260919/summarize_expanded.py'
spec = importlib.util.spec_from_file_location('qwen_size_summary', path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SizeSummaryTests(unittest.TestCase):
    def source(self, path, *, bad_hessian=False, bad_control=False):
        name = path.name
        rank = int(name.rsplit('rank', 1)[1].split('.')[0])
        arm = name.split('-')[0]
        n = dict(B131pack=131184, B240pack=240490, B330pack=330234, validation=55441)[arm]
        records = [dict(name=f'site{i}', ntok=n, hessian_sha256=f'{arm}-{rank}-{i}') for i in range(193)]
        value = dict(rank=rank, weights_id=f'weights-{rank}', records=records,
                     serving_gptq_verified=True, minimum_rows=n)
        if 'projection' in name:
            rmse = dict(B131pack=.8, B240pack=.6, B330pack=.5)[arm]
            value['cases'] = [dict(name=f'site{i}', lane=lane, heldout_rows=55441, fit_rows=n,
                heldout_hessian_sha256=('wrong' if bad_hessian and arm=='B330pack' else f'validation-{rank}-{i}'),
                fit_hessian_sha256=f'{arm}-{rank}-{i}',
                rtn=dict(relative_rmse=1., reference_energy=1., error_energy=(2. if bad_control and arm=='B330pack' else 1.)),
                gptq=dict(relative_rmse=rmse, reference_energy=1., error_energy=rmse**2))
                for i in range(193) for lane in (('w4','fp8') if i < 192 else ('fp8',))]
        return json.dumps(value).encode()

    def test_pairs_all_sites_and_compares_sizes_on_fixed_validation(self):
        with mock.patch.object(Path, 'read_bytes', autospec=True, side_effect=self.source):
            result = summary.summarize(Path('/evidence'))
        w4 = result['arms']['B330pack']['w4']
        self.assertEqual(w4['versus_131k']['sites'], 768)
        self.assertEqual(w4['versus_131k']['improved'], 768)
        self.assertAlmostEqual(w4['versus_131k']['median_ratio'], .625)
        self.assertAlmostEqual(result['ratio_330k_to_240k']['fp8']['median_ratio'], 5/6)

    def test_changed_validation_or_rtn_control_invalidates_the_comparison(self):
        for option in ('bad_hessian', 'bad_control'):
            with self.subTest(option=option), mock.patch.object(Path, 'read_bytes', autospec=True,
                    side_effect=lambda p: self.source(p, **{option: True})):
                with self.assertRaises(AssertionError):
                    summary.summarize(Path('/evidence'))


if __name__ == '__main__':
    unittest.main()
