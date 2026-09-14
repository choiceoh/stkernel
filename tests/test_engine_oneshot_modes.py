"""Inline selection must change native identity and fail before connecting on disagreement."""
import importlib.util
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires PyTorch')
class ModeTests(unittest.TestCase):
    def test_build_defaults_to_inline_and_all_four_variants_have_distinct_keys(self):
        from engine.kernels.oneshot import build
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'ST_ONESHOT_BUILD_ROOT': directory}), \
             patch('torch.utils.cpp_extension.CUDA_HOME', '/fixture/cuda'), \
             patch('engine.kernels.common.native_cache.cuda_toolchain_identity', return_value=[
                 ('/fixture/cuda/bin/nvcc', '13.2.78'), ('/fixture/cuda/bin/ptxas', '13.2.78')]), \
             patch('torch.utils.cpp_extension.load', side_effect=lambda **kwargs: kwargs):
            variants = [build(rails, inline_flags=inline) for rails in (1, 2) for inline in (False, True)]
            self.assertEqual(len({item['name'] for item in variants}), 4)
            self.assertEqual(build()['name'], variants[1]['name'])
            for item, (rails, inline) in zip(variants, ((1, 0), (1, 1), (2, 0), (2, 1))):
                self.assertIn(f'-DOSAR_RAILS={rails}', item['extra_cuda_cflags'])
                self.assertIn(f'-DOSAR_PROXY_INLINE={inline}', item['extra_cuda_cflags'])
            for invalid in (None, '0', 0, 1):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    build(inline_flags=invalid)

    def test_local_mode_or_remote_preparation_failure_releases_before_connect(self):
        from engine.kernels import oneshot
        comm = SimpleNamespace(rank=0, world_size=4)
        for remote_failure in (False, True):
            ext = SimpleNamespace(__name__='native', init=Mock(), shutdown=Mock(), connect=Mock(),
                                  transport_modes=lambda: [0, int(remote_failure)])
            def vote(items, error, group):
                items[:] = [error, 'remote preparation failed' if remote_failure else None, None, None]
            with self.subTest(remote_failure=remote_failure), \
                 patch.object(oneshot, '_cell', return_value=SimpleNamespace(world=4, hidden=4096)), \
                 patch.object(oneshot, 'build', return_value=ext) as build, \
                 patch.object(oneshot.dist, 'new_group', return_value='control'), \
                 patch.object(oneshot.dist, 'all_gather_object', side_effect=vote), \
                 patch.object(oneshot.dist, 'destroy_process_group') as destroy:
                with self.assertRaisesRegex(RuntimeError, 'local preparation failed'):
                    oneshot.OneShot(comm, ('a', 'b', 'c', 'd'))
                build.assert_called_once_with(1, inline_flags=True)
                ext.connect.assert_not_called()
                ext.shutdown.assert_called_once_with()
                destroy.assert_called_once_with('control')


if __name__ == '__main__':
    unittest.main()
