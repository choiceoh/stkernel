"""Pinned import and capsule admission without accelerator imports or calls."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probes'))
import glm53_ep_capsule_runtime as runtime


class CapsuleRuntimeTests(unittest.TestCase):
    def test_expected_identity_matches_preserved_cpu2_imports_and_v6_gpu_candidate(self):
        base = ROOT/'measurements/glm53_ep_local_20260908'
        cpu = json.loads((base/'bindings_capsule_cpu/attempt2/result.json').read_text())
        gpu = json.loads((base/'binding_gpu_submission/v6completed/capture/candidate.json').read_text())
        expected = runtime.expected_runtime_receipt()
        self.assertEqual(expected['binding_identity'], gpu['binding_identity'])
        self.assertEqual(expected['pathfinder'], cpu['imports']['base_pathfinder'])
        self.assertEqual(expected['capsule_manifest_sha256'], cpu['capsule_manifest_sha256'])
        runtime.validate_runtime_receipt(expected)

    def test_receipt_rejects_missing_old_version_hash_origin_and_extra_fields(self):
        expected = runtime.expected_runtime_receipt()
        for path,value in [(['capsule_manifest_sha256'],'0'*64),
                           (['binding_identity','version'],'13.3.1'),
                           (['binding_identity','modules','cuda.bindings.driver','path'],'/stock/driver.so'),
                           (['binding_identity','distributions','cuda-python','sha256'],'0'*64),
                           (['pathfinder','sha256'],'0'*64)]:
            altered=copy.deepcopy(expected);node=altered
            for key in path[:-1]:node=node[key]
            node[path[-1]]=value
            with self.assertRaises(ValueError):runtime.validate_runtime_receipt(altered)
        for altered in (None,{},dict(expected,extra=True)):
            with self.assertRaises(ValueError):runtime.validate_runtime_receipt(altered)
        expected['binding_identity']['version']='poison'
        self.assertEqual(runtime.expected_runtime_receipt()['binding_identity']['version'],'13.0.3')

    def test_host_requires_fixed_manifest_before_touching_files(self):
        with patch.object(runtime,'validate_capsule') as validate:
            for path,sha in [('/missing','0'*64),('/tmp/a,b',runtime.CAPSULE_SHA256)]:
                with self.assertRaises(ValueError):runtime.validate_capsule_input(path,sha)
            validate.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runtime,'validate_capsule',side_effect=ValueError('mutated capsule')):
                with self.assertRaisesRegex(ValueError,'mutated'):runtime.validate_capsule_input(directory,runtime.CAPSULE_SHA256)

    def test_mount_has_exact_read_only_origin_and_python_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime,'validate_capsule') as validate:
            path=Path(directory).resolve()
            argv=runtime.docker_capsule_args(directory,runtime.CAPSULE_SHA256)
            self.assertEqual(argv,['--mount',f'type=bind,source={path},target={runtime.CAPSULE_MOUNT},readonly',
                                   '-e','PYTHONPATH='+runtime.CAPSULE_MOUNT,'-e','PYTHONNOUSERSITE=1',
                                   '-e','PYTHONDONTWRITEBYTECODE=1'])
            validate.assert_called_once_with(path,runtime.CAPSULE_SHA256)

    def test_runtime_rejects_environment_before_imports_and_requires_exact_identities(self):
        expected=runtime.expected_runtime_receipt()
        env={'PYTHONPATH':runtime.CAPSULE_MOUNT,'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
        with (patch.object(runtime,'validate_capsule_input',return_value=Path(runtime.CAPSULE_MOUNT)),
              patch.object(runtime.sys,'dont_write_bytecode',True),
              patch.object(runtime,'verify_import_identity',return_value=expected['binding_identity']) as imports,
              patch.object(runtime,'pathfinder_identity',return_value=expected['pathfinder'])):
            for key in env:
                bad=dict(env);bad[key]='wrong'
                with patch.dict(os.environ,bad), self.assertRaises(RuntimeError):
                    runtime.verify_runtime(runtime.CAPSULE_MOUNT,runtime.CAPSULE_SHA256)
            imports.assert_not_called()
            with patch.dict(os.environ,env):
                self.assertEqual(runtime.verify_runtime(runtime.CAPSULE_MOUNT,runtime.CAPSULE_SHA256),expected)
                imports.return_value=copy.deepcopy(expected['binding_identity'])
                imports.return_value['version']='13.3.1'
                with self.assertRaises(ValueError):runtime.verify_runtime(runtime.CAPSULE_MOUNT,runtime.CAPSULE_SHA256)

    def test_runtime_rejects_other_mount_and_bytecode_before_imports(self):
        with patch.object(runtime,'verify_import_identity') as imports:
            with patch.object(runtime,'validate_capsule_input',return_value=Path('/other')):
                with self.assertRaises(RuntimeError):runtime.verify_runtime('/other',runtime.CAPSULE_SHA256)
            with (patch.object(runtime,'validate_capsule_input',return_value=Path(runtime.CAPSULE_MOUNT)),
                  patch.object(runtime.sys,'dont_write_bytecode',False)):
                with self.assertRaises(RuntimeError):runtime.verify_runtime(runtime.CAPSULE_MOUNT,runtime.CAPSULE_SHA256)
            imports.assert_not_called()

    def test_pathfinder_rejects_wrong_module_or_selected_metadata_before_hash_acceptance(self):
        from types import SimpleNamespace
        module=SimpleNamespace(__file__='/other/pathfinder.py')
        distribution=SimpleNamespace(version='1.7.0',locate_file=lambda name: Path(runtime.PATHFINDER_METADATA))
        with (patch.object(runtime.importlib,'import_module',return_value=module),
              patch.object(runtime.metadata,'distribution',return_value=distribution)):
            with self.assertRaisesRegex(RuntimeError,'origin/version'):runtime.pathfinder_identity()


if __name__=='__main__':unittest.main()
