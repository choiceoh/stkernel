"""Known API discovery diagnostics must not hide a real sanitizer fault."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('san',Path(__file__).resolve().parents[1]/'probes/moe_reform_sanitizer.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
API='========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument" on CUDA API call to cuGetProcAddress_v2.\n'
GOOD='========= COMPUTE-SANITIZER\n'+API+'Compiling CuTe-DSL kernel x\n========= ERROR SUMMARY: 1 error\n'


class SanitizerDiagnostics(unittest.TestCase):
    def test_pre_kernel_lookup_retained(self):
        self.assertEqual(module.sanitizer_result(GOOD,77)['pre_kernel_api_lookup_errors'],1)

    def test_memory_fault_is_fatal(self):
        with self.assertRaises(AssertionError):
            module.sanitizer_result(GOOD.replace('========= ERROR SUMMARY:',
                '========= Invalid __shared__ write of size 1 bytes\n========= ERROR SUMMARY:'),77)

    def test_unknown_api_is_fatal(self):
        with self.assertRaises(AssertionError):
            module.sanitizer_result(GOOD.replace('cuGetProcAddress_v2','cuLaunchKernel'),77)

    def test_wrong_count_is_fatal(self):
        with self.assertRaises(AssertionError):
            module.sanitizer_result(GOOD.replace('1 error','2 errors'),77)

    def test_post_compile_error_is_fatal(self):
        with self.assertRaises(AssertionError):
            module.sanitizer_result('Compiling CuTe-DSL kernel earlier\n'+GOOD,77)

    def test_unexpected_exit_is_fatal(self):
        with self.assertRaises(AssertionError):
            module.sanitizer_result(GOOD,1)

    def test_zero_races_with_api_discovery(self):
        log=GOOD.replace('ERROR SUMMARY: 1 error','RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')
        self.assertEqual(module.sanitizer_result(log,77)['pre_kernel_api_lookup_errors'],1)
        with self.assertRaises(AssertionError):
            module.sanitizer_result(log.replace('0 hazards','1 hazards'),77)


if __name__=='__main__':unittest.main()
