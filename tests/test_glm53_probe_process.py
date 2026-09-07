"""Process failure evidence survives a nonzero exit without false resource proof."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

spec=importlib.util.spec_from_file_location('probe_process',Path(__file__).resolve().parents[1]/'probes/glm53_probe_process.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ProcessTests(unittest.TestCase):
    def test_nonzero_exit_retains_peak_and_final_oom_state(self):
        active=dict(state=dict(Pid=11,Running=True,OOMKilled=False),
            memory={'memory.peak':'1500','memory.events':'oom 0\noom_kill 0'})
        final=dict(state=dict(Pid=0,Running=False,OOMKilled=True,ExitCode=15))
        process=Mock(returncode=15);process.poll.side_effect=[None,15]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'report.json'
            with patch.object(m.subprocess,'Popen',return_value=process), \
                 patch.object(m,'snapshot',side_effect=[active,final]),patch.object(m.time,'sleep'):
                self.assertEqual(m.run(['timeout','test'],'owned',out),15)
            report=json.loads(out.read_text())
            self.assertEqual(report['observed_memory_peak_bytes'],1500)
            self.assertEqual(report['last_resource'],active)
            self.assertEqual(report['final_container'],final)
            self.assertEqual(report['exit_code'],15)
            self.assertFalse(report['serving_gate']);self.assertFalse(report['numerical_acceptance'])

    def test_unavailable_metrics_remain_unknown(self):
        process=Mock(returncode=1);process.poll.side_effect=[None,1]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'report.json'
            with patch.object(m.subprocess,'Popen',return_value=process), \
                 patch.object(m,'snapshot',side_effect=[OSError('sample unavailable'),None]),patch.object(m.time,'sleep'):
                self.assertEqual(m.run(['timeout','test'],'owned',out),1)
            report=json.loads(out.read_text())
            self.assertIsNone(report['observed_memory_peak_bytes'])
            self.assertIsNone(report['final_container'])
            self.assertEqual(report['sampler_errors'],['OSError'])


if __name__=='__main__':unittest.main()
