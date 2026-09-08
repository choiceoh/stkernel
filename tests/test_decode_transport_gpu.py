"""CPU-only tests of the new four-rank transport admission boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
import decode_transport_gpu_probe as probe
import run_decode_transport_gpu as runner
import tests.test_ar_consumer_reuse as fixtures


def transport_proof():
    from ar_consumer_probe import AR_OWNERSHIP_SIZES
    return dict(requested_flags=probe.FLAGS, modes_before=[1,1,0,0,0],
        modes_initialized=[1,1,16,16,16], captures=[
            dict(elements=n, consumer=consumer, ctas=12 if consumer and n <= 32768 else 48)
            for n in AR_OWNERSHIP_SIZES for consumer in (False, True)],
        mixed_graph_cases=[dict(small_elements=small,large_elements=large,seed=seed,
            ctas=[12,48,48,12],exact_outputs=4,passed=True)
            for small,large in ((24576,32769),(32768,65536)) for seed in (17,0,29)])


class ProbeProofTests(unittest.TestCase):
    def test_compiled_flags_and_inline_capability_are_required(self):
        for actual, initialized in (([0,0,0,0,0],False),([1,0,0,0,0],False),
                                    ([1,1,0,0,0],True),([1,1,8,7,8],True),
                                    ([1,1,8,8],True)):
            with self.assertRaises(ValueError):
                probe.require_modes(actual, initialized=initialized)
        probe.require_modes([1,1,0,0,0], initialized=False)
        probe.require_modes([1,1,8,8,8], initialized=True)

    def test_native_calls_are_forwarded_and_capture_modes_observed(self):
        extension = SimpleNamespace(transport_modes=Mock(return_value=[1,1,0,0,0]),
            init=Mock(),oneshot_ar=Mock(return_value='ordinary'),
            oneshot_ar_consumer=Mock(return_value='consumer'))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:True))
        observed = probe.AttestedTransport(extension,torch)
        extension.transport_modes.return_value = [1,1,16,16,16]
        observed.init(0,4,'192.0.2.1')
        for n in (24576,32768,32769):
            value = SimpleNamespace(numel=lambda:n)
            self.assertEqual(observed.oneshot_ar_consumer(value),'consumer')
            self.assertEqual(observed.oneshot_ar(value),'ordinary')
        self.assertEqual(extension.oneshot_ar.call_count,3)
        self.assertEqual(extension.oneshot_ar_consumer.call_count,3)
        self.assertIn((32768,True,12),observed.captures)
        self.assertIn((32769,True,48),observed.captures)
        self.assertIn((24576,False,48),observed.captures)

    def test_old_proof_missing_boundary_or_timing_is_rejected(self):
        correct = dict(schema=probe.SCHEMA, samples=[], transport=transport_proof())
        probe.validate_transport_proof(correct)
        variants=[]
        old=deepcopy(correct);old.pop('schema');variants.append(old)
        timed=deepcopy(correct);timed['samples']=[{'us':1}];variants.append(timed)
        missing=deepcopy(correct);missing['transport']['captures'].pop();variants.append(missing)
        mixed=deepcopy(correct);mixed['transport']['mixed_graph_cases'].pop();variants.append(mixed)
        mode=deepcopy(correct);mode['transport']['modes_initialized'][0]=0;variants.append(mode)
        wrong=deepcopy(correct);wrong['transport']['captures'][0]['ctas']=12;variants.append(wrong)
        for invalid in variants:
            with self.assertRaises(ValueError):
                probe.validate_transport_proof(invalid)


class CohortAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.StageReuse()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.out=self.fixture.base/'new-transport'
        self.out.mkdir()
        self.source=runner.evidence.source()
        self.source['harness_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                      for name in runner.HARNESS}
        self.completed=[]
        for rank in range(4):
            name='probe-rank'+str(rank)
            self.completed.append(self.fixture.rank_evidence(self.out,name))
            path=self.out/(name+'.json')
            report=json.loads(path.read_text())
            report.update(schema=probe.SCHEMA,samples=[],transport=transport_proof(),
                          wrapper_sha256=self.source['harness_sha256'][runner.HARNESS[0]])
            self.fixture.write_json(path,report)

    def validate(self,completed=None):
        return runner.validate_cohort(self.out,'probe',self.completed if completed is None else completed,
                                      self.fixture.runtime,self.source)

    def test_plain_cohort_requires_exactly_all_four_ranks(self):
        entries,hashes=self.validate()
        self.assertEqual(len(entries),4)
        self.assertEqual(len(hashes),12)
        with self.assertRaises(ValueError):
            self.validate(self.completed[:3])
        with self.assertRaises(ValueError):
            self.validate([*self.completed,self.completed[0]])

    def test_old_same_source_report_cannot_admit_new_flags(self):
        # Even complete old numerics/container proof from this source fails.
        self.fixture.rank_evidence(self.out,'probe-rank2')
        with self.assertRaises(ValueError):
            self.validate()

    def test_bad_qp_or_source_or_failed_container_is_rejected(self):
        path=self.out/'probe-rank1.json'
        initial=json.loads(path.read_text())
        for change in ('cap','source'):
            report=deepcopy(initial)
            if change=='cap':report['transport']['modes_initialized'][-1]=4
            else:report['wrapper_sha256']='0'*64
            self.fixture.write_json(path,report)
            with self.assertRaises(ValueError):self.validate()
        self.fixture.write_json(path,initial)
        container=self.out/'probe-rank3.container.json'
        data=json.loads(container.read_text());data['state']['OOMKilled']=True
        self.fixture.write_json(container,data)
        with self.assertRaises(ValueError):self.validate()

    def test_onepass_admission_checks_artifact_hashes_and_required_group(self):
        _,hashes=self.validate()
        for name,value in (('source.json',self.source),('runtime.json',self.fixture.runtime)):
            self.fixture.write_json(self.out/name,value)
            hashes[name]=hashlib.sha256((self.out/name).read_bytes()).hexdigest()
        admission=dict(schema=probe.SCHEMA,status='PASS',image=runner.base.IMAGE,flags=probe.FLAGS,
            selected_stages=['probe'],completed=self.completed,artifacts_sha256=hashes)
        self.fixture.write_json(self.out/'admission.json',admission)
        with patch.object(runner,'source_identity',return_value=self.source):
            runner.verify_admission(self.out)
            with self.assertRaises(ValueError):
                runner.verify_admission(self.out,required_stages=('racecheck',))
            with (self.out/'probe-rank0.log').open('a') as log:log.write('changed\n')
            with self.assertRaises(ValueError):runner.verify_admission(self.out)


if __name__=='__main__':
    unittest.main()
