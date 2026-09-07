"""Missing destinations, changed-input failures and output lifetime block INT8 acceptance."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'probes'))
from glm53_prefill_int8_sanitize import ROWS,validate


class SanitizeTests(unittest.TestCase):
    def fixture(self):
        return dict(cases=[dict(rows=n,changed=c,destination=d,packet_equal=True,output_equal=True,
            source_unchanged=True,finite=True,retained_unchanged=True)
            for n in ROWS for c in (False,True) for d in range(4)])

    def test_missing_changed_input_destination_or_repeated_case_blocks_acceptance(self):
        original=self.fixture();self.assertIs(validate(original),original)
        for mutation in ('missing','duplicate','old_input'):
            report=copy.deepcopy(original)
            if mutation=='missing':report['cases'].pop()
            elif mutation=='duplicate':report['cases'][-1]=report['cases'][-2]
            else:report['cases'][-1]['changed']=False
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'coverage'):validate(report)

    def test_any_packet_output_finiteness_source_or_lifetime_failure_is_fatal(self):
        for field in ('packet_equal','output_equal','source_unchanged','finite','retained_unchanged'):
            report=self.fixture();report['cases'][-1][field]=False
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'fidelity or lifetime'):validate(report)


if __name__=='__main__':unittest.main()
