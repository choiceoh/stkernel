"""CPU failures for a mislabeled/mixed CUDA migration and corrupt inputs."""
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.runtime.fetch_cuda132 import fetch
from engine.runtime.verify import check_versions, compiler_report

ROOT = Path(__file__).resolve().parents[1]


class RuntimePinTests(unittest.TestCase):
    def test_a_cu132_torch_label_does_not_hide_old_runtime_packages(self):
        expected = {'torch': '2.13.0+cu132', 'nvidia-cuda-runtime': '13.2.75'}
        actual = expected | {'nvidia-cuda-runtime': '13.0.96'}
        with patch('importlib.metadata.version', side_effect=actual.__getitem__):
            with self.assertRaisesRegex(RuntimeError, 'nvidia-cuda-runtime'):
                check_versions(expected)
        with patch('importlib.metadata.version', side_effect=expected.__getitem__):
            self.assertEqual(check_versions(expected), expected)

    def test_old_nvcc_or_triton_bundled_assembler_is_refused_before_loading_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            toolkit = Path(tmp) / 'nvidia/cu13'
            (toolkit / 'bin').mkdir(parents=True)
            for name in ('nvcc', 'ptxas'):
                (toolkit / 'bin' / name).write_text('fixture')
            old = [(str(toolkit / 'bin' / name), 'release 13.0, V13.0.88') for name in ('nvcc', 'ptxas')]
            new = [(path, 'release 13.2, V13.2.78') for path, _ in old]
            with patch.dict('sys.modules', {
                    'torch.utils.cpp_extension': SimpleNamespace(CUDA_HOME=str(toolkit)),
                    'triton': SimpleNamespace(knobs=SimpleNamespace(nvidia=SimpleNamespace(
                        ptxas=SimpleNamespace(path='/triton/old-ptxas'),
                        ptxas_blackwell=SimpleNamespace(path='/triton/old-ptxas'))))}), \
                    patch('sysconfig.get_path', return_value=tmp), \
                    patch('ctypes.CDLL', side_effect=AssertionError('must fail before loading a library')):
                with patch('engine.kernels.common.native_cache.cuda_toolchain_identity', return_value=old):
                    with self.assertRaisesRegex(RuntimeError, 'compiler mismatch'):
                        compiler_report({'cuda_compiler': '13.2.78'})
                with patch('engine.kernels.common.native_cache.cuda_toolchain_identity', return_value=new):
                    with self.assertRaisesRegex(RuntimeError, 'unpinned assembler'):
                        compiler_report({'cuda_compiler': '13.2.78'})

    def test_lock_pins_runtime_compilers_and_matching_torch_vision_video(self):
        lock = json.loads((ROOT / 'engine/runtime/cuda132.lock.json').read_text())
        deps = json.loads((ROOT / 'engine/runtime/dependencies.json').read_text())
        wheels = {entry['name']: entry for entry in lock['wheels']}
        self.assertEqual(len(wheels), len(lock['wheels']))
        for name in ('torch', 'torchvision', 'torchcodec'):
            self.assertEqual(wheels[name]['version'], deps[name])
        for name in ('nvidia-cuda-nvcc', 'nvidia-cuda-nvrtc', 'nvidia-nvvm', 'nvidia-nvjitlink'):
            self.assertEqual(wheels[name]['version'], deps['cuda_compiler'])
        self.assertEqual(wheels['cuda-toolkit']['version'], deps['cuda_toolkit'])


class LockedFetchTests(unittest.TestCase):
    def test_verified_download_is_reused_and_corruption_is_replaced(self):
        data = b'wheel fixture'
        entry = dict(filename='fixture.whl', url='https://example.invalid/fixture.whl',
                     sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(data)):
                fetch(entry, directory)
            with patch('urllib.request.urlopen', side_effect=AssertionError('valid cache must be reused')):
                fetch(entry, directory)
            (directory / entry['filename']).write_bytes(b'corrupt')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(data)):
                fetch(entry, directory)
            self.assertEqual((directory / entry['filename']).read_bytes(), data)

    def test_bad_hash_never_publishes_a_wheel(self):
        entry = dict(filename='fixture.whl', url='https://example.invalid/fixture.whl', sha256='0' * 64)
        with tempfile.TemporaryDirectory() as tmp, \
                patch('urllib.request.urlopen', return_value=io.BytesIO(b'wrong artifact')):
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                fetch(entry, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
