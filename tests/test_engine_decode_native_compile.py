"""Resource evidence follows the CUDA tool's actual function header formats."""
import unittest

from probes.engine_decode_native_compile import mhc_resources


class MhcResourceTests(unittest.TestCase):
    def test_cuda13_resource_dump_retains_candidate_and_packed_consumer(self):
        text = '''
Fatbin elf code:
arch = sm_121a
 Function _ZN43_anonymous16mk_mhc_ar_kernelILb1ELi4096EEEvArgs:
  REG:128 STACK:16 SHARED:28736 LOCAL:0
 Function _ZN43_anonymous23mk_mhc_ar_single_kernelEArgs:
  REG:80 STACK:16 SHARED:28736 LOCAL:0
 Function _ZN43_anonymous20mk_gemm_input_kernelEArgs:
  REG:80 STACK:0 SHARED:1040 LOCAL:0
'''
        sections = mhc_resources(text)
        self.assertEqual(len(sections), 2)
        self.assertIn('mk_mhc_ar_single_kernel', sections[1]['kernel'])
        self.assertEqual(sections[1]['usage'], 'REG:80 STACK:16 SHARED:28736 LOCAL:0')

    def test_older_header_and_absent_symbol(self):
        self.assertEqual(mhc_resources(' Function : mk_mhc_ar_single_kernel\n  REG:80\n'),
                         [dict(kernel='mk_mhc_ar_single_kernel', usage='REG:80')])
        self.assertEqual(mhc_resources(' Function unrelated:\n  REG:80\n'), [])


if __name__ == '__main__':
    unittest.main()
