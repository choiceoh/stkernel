"""One real error must remain visible; detector failures must block acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'probes'))
from glm53_sanitizer_report import error_count,validate_canaries


def control_fixture(source):
    digest=lambda name:hashlib.sha256((source/name).read_bytes()).hexdigest()
    return [dict(mode='driver-torch-invalid',exit_code=99,error_count=1,
        api_reports={'========= Program hit CUDA_ERROR_INVALID_DEVICE (error 101) due to "invalid device ordinal" on CUDA API call to cuDeviceGet.':1},
        program=dict(source_sha256=digest('glm53_cuda_driver_lookup_check.py'),
            positive_control=dict(call='cuDeviceGet',code=101,intentional=True)))]+[
        dict(tool=tool,exit_code=99,deliberate_error_detected=True,source_sha256=digest('glm53_sanitizer_order_canary.py'))
        for tool in ('memcheck','racecheck')]


class ReportTests(unittest.TestCase):
    def test_singular_error_is_counted_and_missing_or_duplicate_summary_fails(self):
        for count in (0,1,34):
            line=f'========= ERROR SUMMARY: {count} '+('error' if count==1 else 'errors')
            self.assertEqual(error_count([line]),count)
            with self.assertRaises(ValueError):error_count([line,line])
        with self.assertRaises(ValueError):error_count([])

    def test_missing_or_broken_detector_cannot_approve_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('glm53_cuda_driver_lookup_check.py','glm53_sanitizer_order_canary.py'):(root/name).write_text(name)
            original=control_fixture(root)
            def validate(rows):return validate_canaries('\n'.join(json.dumps(r) for r in rows),root)
            self.assertEqual(validate(original)['verdict'],'SANITIZER_DETECTOR_CONTROLS_PASS')
            for case in ('missing','api_count','api_exit','memory','race','source'):
                rows=copy.deepcopy(original)
                if case=='missing':rows.pop()
                elif case=='api_count':rows[0]['error_count']=0
                elif case=='api_exit':rows[0]['exit_code']=0
                elif case=='memory':rows[1]['deliberate_error_detected']=False
                elif case=='race':rows[2]['deliberate_error_detected']=False
                else:rows[2]['source_sha256']='0'*64
                with self.subTest(case=case),self.assertRaises(ValueError):validate(rows)


if __name__=='__main__':unittest.main()
