"""CUDA binding initialization failures stop before a PyTorch runtime context."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class InitTests(unittest.TestCase):
    def test_driver_first_and_failed_driver_status_prevents_torch_import(self):
        path=Path(__file__).resolve().parents[1]/'probes/glm53_moe_m64_sanitize.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        namespace=dict(argparse=argparse,__doc__='test',json=json)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        original=__import__
        for failure in (None,'init','version'):
            calls=[]
            def init(_):calls.append('driver_init');return (1 if failure=='init' else 0,)
            def version():calls.append('driver_version');return (1 if failure=='version' else 0,13000)
            driver=SimpleNamespace(cuInit=init,cuDriverGetVersion=version)
            def importing(name,*args,**kwargs):
                if name=='cuda.bindings':return SimpleNamespace(driver=driver)
                if name=='torch':
                    calls.append('torch_import');raise RuntimeError('stop before actual CUDA')
                return original(name,*args,**kwargs)
            with patch('sys.argv',['sanitize']),patch('builtins.__import__',side_effect=importing), \
                    self.assertRaises(AssertionError if failure else RuntimeError):namespace['main']()
            expected=['driver_version'] if failure=='version' else ['driver_version','driver_init']
            if failure is None:expected.append('torch_import')
            self.assertEqual(calls,expected)


if __name__=='__main__':unittest.main()
